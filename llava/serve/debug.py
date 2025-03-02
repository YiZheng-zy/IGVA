# 确保项目根目录被加入到 sys.path 中
# import sys
# import os
# project_root = '/'#项目路径
# if project_root not in sys.path: 
#     sys.path.insert(0, project_root) 

from llava.model.language_model.llava_llama import LlavaLlamaModel, LlavaConfig, LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.serve.cli import load_image
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from transformers import AutoTokenizer
import torch

device = "cpu"
device_map = "auto" # 用于load vision tower
model_path = "/"#模型路径
conv_mode = "llava_v1"

# 从 config 文件去实例化模型
config_path = model_path + "/config.json"
config = LlavaConfig.from_json_file(config_path)
model = LlavaLlamaForCausalLM(config).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# 加载视觉编码器
vision_tower = model.get_vision_tower()
vision_tower.load_model(device_map=device_map)
image_processor = vision_tower.image_processor

# 加载图像
image_file = "llava/serve/examples/test_image.png"
image = load_image(image_file) # PIL.Image.Image 的实例
image_size = image.size
print(image_size)

image_tensor = process_images([image], image_processor, model.config)
if type(image_tensor) is list: 
    print("multiple images processed")
    image_tensor = [image.to(model.device, dtype=model.dtype) for image in image_tensor]
else: 
    image_tensor = image_tensor.to(model.device, dtype=model.dtype)
print(image_tensor.shape)

instruction = 'what is the brand logo shown in the image?'
conv = conv_templates[conv_mode].copy()
roles = conv.roles
inp = conv.roles[0] + ': ' + instruction
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
print(prompt)

# 将prompt转换为单词表中的id
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
print(input_ids.shape)

output = model(
        input_ids = input_ids, # 必须定义
        input_text = instruction,  # 必须定义
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = True,
        output_attentions = None, 
        output_hidden_states = None,
        images = image_tensor, 
        image_sizes = [image_size], # 必须定义
        return_dict = None)