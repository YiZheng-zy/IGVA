from PIL import Image
from io import BytesIO
import base64
import torch
import math
import ast

from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX

# 从一组可能的分辨率中选择一个最适合原始图像尺寸的分辨率，该选择基于两个标准：有效分辨率的最大化和分辨率浪费的最小化
def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit 

# 将输入图像调整为目标分辨率
def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

# 将图像进行子图划分
def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches

# 计算给定图像在水平方向和垂直方向上被分割成多少个子图
def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size

# 将任意分辨率的图像进行分辨率调整和子图划分
# 若原始图像的分辨率很小，会按照最适合的目标分辨率进行放大
def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # 得到目标分辨率的列表 [(h1, w1), (h2, w2), ...]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
        
    # 从可能的分辨率列表中选择一个最适合当前图像尺寸的目标分辨率
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    # 将输入图像调整为目标分辨率。若图像的宽高比不符合目标分辨率，会填充空白区域（通常为黑色），以适应目标分辨率
    image_padded = resize_and_pad_image(image, best_resolution) 
    # 将调整后的图像划分为多个子图。每个子图的尺寸由 processor.crop_size['height'] 决定
    patches = divide_to_patches(image_padded, processor.crop_size['height'])
    # 将原始图像缩放到与子图相同的分辨率
    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))
    # 组合子图和整图
    image_patches = [image_original_resize] + patches
    # 使用CLIP自带的图像处理器对每个图像块（包括原始图像和所有子图）进行预处理
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

# 将图像按照background_color的像素值填充为正方形
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    # images: 一个图像对象的列表，每个图像对象是 PIL.Image.Image 类
    # image_processor: 一个图像处理器对象，是 CLIPImageProcessor 类
    # model_cfg: 模型配置参数，控制图像处理行为
    
    # 获取如何处理图像的宽高比, model_cfg来自模型权重文件夹中的config文件
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            # 用于将图像填充为正方形，填充的是clip配置文件中三个channel的像素均值，均值归一化后就变成0了
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            # 将填充后的图像进行CLIP自带的预处理流程
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # 将一个(3, 336, 336)的tensor加入列表 
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            # 进行子图划分
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            # 将一个(N, 3, 336, 336)的tensor加入列表，N代表子图个数
            new_images.append(image)
    else:
        # 如果 image_aspect_ratio 没有定义，则直接使用 image_processor 对图像列表进行预处理
        # 最后输出的是一个（N, 3, 336, 336）的tensor，N代表图像的数量
        return image_processor(images, return_tensors='pt')['pixel_values']
    # 把 new_images 中的tensor拼成一个大tensor
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images 

# 将包含特殊token的文本提示（prompt）转换为适合模型输入的 input_ids
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    # 将prompt按照 <image> 标签进行分块，第一个块其实就对应sys message
    # 对于每个块中的字符串，使用tokenizer对其进行分词，得到对应的input_ids列表
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
    
    # 将 sep 插入到列表 X 中的每两个元素之间，并去掉最后一个 sep
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    # 如果prompt_chunks中的第一个片段存在，且第一个 input_id 是 bos_token_id，那么 offset 设置为 1，并将该标记加入 input_ids 列表。
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
        
    # 将每个分割后的文本片段的 input_ids 插入到 input_ids 列表中，并在片段之间插入图像标记 image_token_index
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    
    # 如果 return_tensors 是 'pt'，则返回 PyTorch 的 LongTensor。否则，返回一个普通的列表 input_ids。
    if return_tensors is not None:
        if return_tensors == 'pt': 
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
