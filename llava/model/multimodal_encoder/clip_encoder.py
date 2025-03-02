import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        # vision_tower 是一个str，表明所使用vision tower权重文件的路径
    
        super().__init__()

        self.is_loaded = False
        # 指明vision tower权重文件的路径
        self.vision_tower_name = vision_tower
        
        #self.select_layer = args.mm_vision_select_layer 
        #self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch') 

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False) 
        self.is_loaded = True
    
    # #原始token选择函数
    # def feature_select(self, image_forward_outs):
    #     image_features = image_forward_outs.hidden_states[self.select_layer]
    #     if self.select_feature == 'patch':
    #         image_features = image_features[:, 1:]
    #     elif self.select_feature == 'cls_patch':
    #         image_features = image_features
    #     else:
    #         raise ValueError(f'Unexpected select feature: {self.select_feature}')
    #     return image_features
    
    # def select_cls_and_patch_features_from_layers(self, outputs, aggregate_layers):
    #     """
    #     从CLIPVisionModel的输出中抽取特定层级的cls token和patch token特征
        
    #     参数:
    #         outputs: CLIPVisionModel的输出，包含hidden_states
    #         aggregate_layers: 列表，表示要抽取特征的层级索引
        
    #     返回:
    #         cls_features: 一个张量，形状为 (batch_size, num_selected_layers, feature_dim)
    #         patch_features: 一个张量，形状为 (batch_size, num_selected_layers, num_patch, feature_dim)
    #     """
    #     # 获取当前编码器的总层数 23
    #     last_idx = len(self.vision_tower.vision_model.encoder.layers) - 1
        
    #     # 将倒数第二层加进来
    #     aggregate_layers.append(last_idx - 1)
        
    #     # 提取所需层级的hidden_states
    #     selected_hidden_states = [outputs.hidden_states[layer] for layer in aggregate_layers]
        
    #     # 从每个选定的层中提取cls token的特征
    #     cls_features = torch.stack([hidden_state[:, 0, :] for hidden_state in selected_hidden_states], dim=1)
        
    #     # 从每个选定的层中提取patch特征，去除第一个cls token
    #     patch_features = torch.stack([hidden_state[:, 1:, :] for hidden_state in selected_hidden_states], dim=1)
        
    #     return cls_features, patch_features 

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            raise NotImplementedError("Not implemented for multi images")
        else: 
            # 返回所有层的特征 
            # 包含25个张量的tuple，每个张量的形状是 （batch_size, num_patches+1, hidden_dim）
            return self.vision_tower(images.to(dtype=self.dtype), output_hidden_states=True).hidden_states

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
    
    # 在长或宽任意一边的patch数量
    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    # patch总数
    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


# 并在此基础上添加了多尺度图像处理的功能
class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)
        
        # 图像每一侧的尺寸可以是336/672/1008的一种，因此最终可以接受6种分辨率
        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward 

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
