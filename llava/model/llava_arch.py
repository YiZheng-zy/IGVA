#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import ast

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .multimodal_aggregator.builder import build_vision_aggregator

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

# 负责初始化和管理视觉编码器和适配器以及相关配置
class LlavaMetaModel: 

    def __init__(self, config):
        #print("LlavaMetaModel is being initialized")
        super(LlavaMetaModel, self).__init__(config)
        # 若配置中包含 mm_vision_tower，则构建视觉塔（vision_tower）和投影器（mm_projector）。
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True) # 参数在build的过程中已经被冻结了
            self.mm_projector = build_vision_projector(config) 
            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                ) 
                
            # 根据情况来决定是否加载aggregator和embedder
            if not config.aggregate_by_average: 
                self.sentence_embedder = SentenceTransformer(config.sentence_embedder)
                self.sentence_embedder.requires_grad_(False) # 冻结句子嵌入模型的参数
                visual_feature_dim = self.vision_tower.hidden_size 
                sentence_embed_dim = config.sentence_embed_dim  
                self.vision_aggregator = build_vision_aggregator(config, 
                                                                 feature_dim = visual_feature_dim, 
                                                                 embed_dim = sentence_embed_dim)
    
    # 返回模型的视觉塔。如果视觉塔是列表，返回列表中的第一个视觉塔。
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    # 用于在训练开始前初始化与视觉相关的所有部分，在此之前模型只是通过LLM的配置文件进行了实例化
    def initialize_vision_modules(self, model_args, fsdp=None): 
        # 初始化视觉编码器
        vision_tower = model_args.vision_tower
        #mm_vision_select_layer = model_args.mm_vision_select_layer
        #mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type 
        self.config.mm_vision_tower = vision_tower
        if self.get_vision_tower() is None: 
            # 根据训练文件中的model_args加载vision_tower，此时并不会delay load
            vision_tower = build_vision_tower(model_args) 
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else: 
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()
        # 初始化多模态适配器
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size 
        #self.config.mm_vision_select_layer = mm_vision_select_layer
        #self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else: 
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
        # 如果是在进行微调，就要加载预训练后的适配器参数
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
        
        # 初始化aggregator
        self.config.aggregate_by_average = model_args.aggregate_by_average
        self.config.vit_aggregate_groups = model_args.vit_aggregate_groups
        self.config.vit_base_layer = model_args.vit_base_layer
        # self.vit_aggregate_groups = ast.literal_eval(self.config.vit_aggregate_groups)
        # self.vit_base_layer = self.config.vit_base_layer
        if not self.config.aggregate_by_average: 
            self.config.sentence_embedder = model_args.sentence_embedder
            self.config.sentence_embed_dim = model_args.sentence_embed_dim
            self.config.aggregator_num_transformers = model_args.aggregator_num_transformers
            self.config.aggregator_num_heads = model_args.aggregator_num_heads
            self.config.aggregator_hidden_dim = model_args.aggregator_hidden_dim         
            if getattr(self, 'vision_aggregator', None) is None:
                visual_feature_dim = self.vision_tower.hidden_size 
                sentence_embed_dim = self.config.sentence_embed_dim 
                self.vision_aggregator = build_vision_aggregator(self.config, 
                                                                 feature_dim = visual_feature_dim, 
                                                                 embed_dim = sentence_embed_dim)                    
            if model_args.pretrain_aggregator is not None: 
                aggregator_weights = torch.load(model_args.pretrain_aggregator, map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self.vision_aggregator.load_state_dict(get_w(aggregator_weights, 'vision_aggregator'))        
            

# 用于对填充的图像进行去填充操作，将图像恢复到原始尺寸
def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor

# 抽象基类 是不能直接实例化的类，其主要作用是作为其他类的基类，定义一组子类必须实现的方法或属性。
# 抽象基类本身并不提供这些方法的具体实现，而是规定这些方法应该存在于子类中。
class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    # 编码、整合、对齐
    def encode_images(self, images, sentence_embeds=None):    
        # 获取关键参数 
        aggregate_by_average = self.get_model().config.aggregate_by_average
        aggregate_groups = ast.literal_eval(self.get_model().config.vit_aggregate_groups)
        base_layer = self.get_model().config.vit_base_layer
        # 获取各个组件 
        vision_tower =  self.get_model().get_vision_tower()
        mm_projector = self.get_model().mm_projector
        # 对图像进行编码 
        clip_outputs = vision_tower(images) # 包含25个张量的tuple，每个张量的形状是 （batch_size, num_patches+1, hidden_dim）
        # 提取 base feature
        base_patch_features = clip_outputs[base_layer][:, 1:, :] #  (batch_size, num_patches, hidden_dim)
        # 按照分好的group去提取CLIP的特征
        grouped_patch_features = []  # 存储每组的 patch 特征平均值
        grouped_cls_features = []  # 存储每组的 cls 特征平均值
        for group in aggregate_groups:
            # 提取该组所有层的输出，并堆叠在一起
            group_outputs = torch.stack([clip_outputs[i] for i in group], dim=0) #  (num_layers_in_group, batch_size, num_patches + 1, hidden_dim)
            # 计算该组内的平均值
            group_mean = group_outputs.mean(dim=0) #  (batch_size, num_patches + 1, hidden_dim)
            # 分别提取 CLS 和 Patch 特征的组内平均
            cls_features = group_mean[:, 0, :]  # (batch_size, hidden_dim)
            patch_features = group_mean[:, 1:, :]  #  (batch_size, num_patches, hidden_dim)
            grouped_cls_features.append(cls_features)
            grouped_patch_features.append(patch_features)
        grouped_patch_features = torch.stack(grouped_patch_features, dim=0) #  (num_groups, batch_size, num_patches, hidden_dim)
        grouped_cls_features =  torch.stack(grouped_cls_features, dim=0)  #  (num_groups, batch_size, hidden_dim)
        if aggregate_by_average:
            # 组之间计算平均
            aggregated_patch_features =grouped_patch_features.mean(dim=0) # (batch_size, num_patches, hidden_dim)
            final_features = torch.cat((base_patch_features, aggregated_patch_features), dim=-1) # (batch_size, num_patches, hidden_dim * 2)
            return mm_projector(final_features) # (batch_size, num_patches, llm_hidden_dim)
        else: 
            vision_aggregator = self.get_model().vision_aggregator
            # 获取文本指令的语义嵌入  
            sentence_embeds = sentence_embeds.to(device = grouped_cls_features.device, dtype = grouped_cls_features.dtype) # (batch_size, embed_dim)
            # 调整cls feature的维度 
            grouped_cls_features = grouped_cls_features.permute(1, 0, 2) # (batch_size, num_groups, hidden_dim)
            weights = vision_aggregator(cls_features = grouped_cls_features, 
                                        sentence_embed = sentence_embeds) # (batch_size, num_groups）
            # 将 weights 的形状调整为 (num_groups, batch_size, 1, 1) 以进行广播
            weights = weights.permute(1, 0).unsqueeze(-1).unsqueeze(-1)  # (num_groups, batch_size, 1, 1)
            # 对每组的 patch 特征进行加权
            weighted_features = grouped_patch_features * weights  # (num_groups, batch_size, num_patches, hidden_dim)
            aggregated_patch_features = weighted_features.sum(dim=0)  # (batch_size, num_patches, hidden_dim)
            final_features = torch.cat((base_patch_features, aggregated_patch_features), dim=-1)  # (batch_size, num_patches, hidden_dim * 2)
            return mm_projector(final_features) # (batch_size, num_patches, llm_hidden_dim)
    
    # 根据输入的图像和文本，生成适合模型处理的多模态嵌入序列
    def prepare_inputs_labels_for_multimodal( 
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, sentence_embeds=None, image_sizes=None
    ): 
        vision_tower = self.get_vision_tower()

        # 如果没有图像输入，或 vision_tower 不存在，或 input_ids 只有一个 token，直接返回输入，不做进一步处理。
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # 分情况进行图像特征提取与处理 
        # type(images) is list 代表 images 中包含了多张图像，且每张图像都被进行了子图划分
        # ndim==5 代表 images 包含多张图像或者单张图像被划分为了多张子图
        if type(images) is list or images.ndim == 5: 
            if type(images) is list:
                # images变成了一个4维张量的列表（batch_size, C, H, W）
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            # 将列表拼成一个大张量
            concat_images = torch.cat([image for image in images], dim=0)
            # 将每一个图像/子图进行CLIP特征提取，得到的是3维张量 （batch_size, num_features, feature_dim)
            aggregate_layers = self.get_model().aggregate_layers 
            print("there")
            image_features = self.encode_images(concat_images, aggregate_layers, sentence_embeds)            
            # 获取每张图像被分成了几个子图
            split_sizes = [image.shape[0] for image in images] 
            # 将不同图像的特征进行划分，得到了4维张量 (n_images, n_sub_images, num_features, feature_dim)
            image_features = torch.split(image_features, split_sizes, dim=0) 
            # 获取图像特征的融合方法
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                # 将4维张量变成了3维，(total_n_sub_images, num_features, feature_dim) 
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = [] 
                # 逐个图像（非子图）进行处理
                for image_idx, image_feature in enumerate(image_features):
                    # 如果图像被划分成了多个子图 
                    if image_feature.shape[0] > 1: 
                        base_image_feature = image_feature[0] # 整图rescale后的特征 （num_features, feature_dim）
                        image_feature = image_feature[1:] # 各个子图的特征 （n_sub_images - 1, num_features, feature_dim）
                        height = width = self.get_vision_tower().num_patches_per_side # 竖直方向与水平方向的patch数量，注意不是子图数量
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres': 
                            # 获取图像在水平方向和垂直方向上被分割成多少个子图
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            # 按照空间进行排列 （竖直方向子图数量，水平方向子图数量，竖直方向patch数量，水平方向patch数量，特征维度）
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else: 
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0] 
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                # 返回一个列表，每个元素都是2维张量（num_features, feature_dim)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        # 如果images是一个4维张量（batch_size, C, H, W）
        else: 
            #################### 单图的普通模式 ##################### 
            image_features = self.encode_images(images, sentence_embeds) 
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError
        
        # 在推理的时候，下面的三个值应该都为none；训练的时候，就不是none了
        _labels = labels 
        _position_ids = position_ids
        _attention_mask = attention_mask 
        
        # 为 input_ids 生成一个全为 1 的掩码，表示所有 token 都是有效的,接下来可能会根据填充的位置更新这个掩码
        if attention_mask is None: 
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool) # (batch_size, seq_len)
        else:  
            attention_mask = attention_mask.bool()
        # 为 input_ids 生成每个输入序列的位置编号
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device) # (seq_len, )
        # 生成一个与 input_ids 形状相同的张量，并将所有元素设置为 IGNORE_INDEX。IGNORE_INDEX 是一个特定的值，用来标记在训练中不计算损失的 token（通常是填充 token）
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX) # (batch_size, seq_len)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids  
        # 依据attention mask的布尔值，提取出有效的input ids（将每个句子中无效的填充部分去除）
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        # 提取出有效的labels，原理同上
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        # 依次处理batch中的每个样本
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 计算该样本中的图像数量 
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() 
            # 如果当前输入不包含图像，代码直接将文本的嵌入向量和图像特征结合在一起（不过此处图像特征为空），并将其加入 new_input_embeds 列表中
            if num_images == 0: 
                cur_image_features = image_features[cur_image_idx] # 2维图像特征张量（num_features, feature_dim)
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids) # 得到了所有本文token的嵌入向量
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0) # cur_image_features[0:0]是一个空张量
                new_input_embeds.append(cur_input_embeds) # append进去的是（sample_len, hidden_dim）
                new_labels.append(labels[batch_idx]) # append进去的是（sample_len, hidden_dim）
                cur_image_idx += 1
                # 跳过当前循环中的剩余代码
                continue 
            # 当输入包含图像 token 时，首先找到所有图像 token 的索引，然后将这些 token 之前和之后的文本分离出来。
            # 将分离出来的文本转换为嵌入向量，并按原来的分割方式重新划分。
            
            # 返回一个列表，包含了输入序列中所有图像标记的位置索引，并在开头和结尾分别添加了 -1 和序列长度 cur_input_ids.shape[0] 作为边界
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            
            # 提取文本 token 和标签，noim代表no image
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                # 提取的范围是从当前图像标记的下一个位置（image_token_indices[i]+1）到下一个图像标记之前的位置（image_token_indices[i+1]）
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            # 记录每段文本的长度
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            # 将所有提取出的文本 token 拼接成一个序列，并将其转换为嵌入向量
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            # 将嵌入向量重新分割回原来的长度，输出的是一个张量列表
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            # 最终的文本和图像嵌入向量，以及对应的标签 
            cur_new_input_embeds = [] 
            cur_new_labels = [] 
            for i in range(num_images + 1):
                # 每次循环，先添加一段文本嵌入和对应标签
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                # 如果尚有图像 token，接着添加对应的图像嵌入，并用 IGNORE_INDEX 填充其标签，表示这些位置的标签应被忽略。
                if i < num_images:  
                    ##########################################        
                    # if cur_image_idx >= image_features.shape[0]:
                    #     print(f"Error: for sample idx {batch_idx}, cur_image_idx {cur_image_idx} exceeds image_features size {image_features.shape[0]},  the num_images is: {num_images}")
                    #     break
                    ####################################
                    cur_image_features = image_features[cur_image_idx] # （num_features, feature_dim）
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)) # (num_features,)
            
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            
            # 将所有嵌入向量和标签分别拼接成一个整体 
            cur_new_input_embeds = torch.cat(cur_new_input_embeds) # (seq_len, hidden_dim)
            cur_new_labels = torch.cat(cur_new_labels) # (seq_len, )
            # 放入batch数据的列表中
            new_input_embeds.append(cur_new_input_embeds) 
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        # 根据上下文窗口长度，对输入内容进行截断
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None: 
            # 如果超过了窗口长度，那么会对 new_input_embeds 和 new_labels 进行截断，只保留前 tokenizer_model_max_length 个元素 
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        # 找到当前batch中最大的序列长度
        max_len = max(x.shape[0] for x in new_input_embeds)
        # 找到batch size 
        batch_size = len(new_input_embeds)
        # 生成带有填充的张量 
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # 对每个输入样本进行填充，使其长度达到 max_len 
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                # 在输入数据的左侧进行填充
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0)) 
                # 更新labels、mask、position
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                # 在输入数据的右侧进行填充
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                # 更新labels、mask、position
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
         
        # 最后得到的嵌入张量，维度为（batch_size, max_seq_len, hidden_dim）
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # 如果是在进行推理，需要将labels\attention mask\position id 这些值重新设为none
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    # 初始化视觉标记器（tokenizer），根据模型参数决定是否添加新的图像标记，并调整嵌入层的大小。
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
