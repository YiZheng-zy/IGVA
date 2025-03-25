from llava.model.language_model.llava_llama import LlavaLlamaModel, LlavaConfig, LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.serve.cli import load_image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from transformers import AutoTokenizer
import torch
import numpy as np

# ----------------------------Load Model----------------------------
device = "cpu"
device_map = "cpu" # load vision tower 
torch_dtype = torch.float32 # note：if device is cpu，has to be set as float32
model_path = "/data/proj903/lx-AvgEmbed"  # path to the model weights
conv_mode = "llava_v1" 
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False) 
model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, attn_implementation="eager").to(device)
sentence_embedder = model.get_model().sentence_embedder 
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
if device_map != 'auto': 
        vision_tower.to(device=device_map, dtype=torch_dtype)
image_processor = vision_tower.image_processor # 是CLIPImageProcessor类 

# ----------------------------image preprocessing----------------------------
image_file = "/home/lx/LLaVA/llava/serve/examples/ex3.jpg" # path to the input image
image = load_image(image_file) 
image_size = image.size 
image_tensor = process_images([image], image_processor, model.config) 
image_tensor = image_tensor.to(model.device, dtype=model.dtype) 
image_resized = image.resize((336, 336))
image_resized

# ----------------------------text preprocessing----------------------------
instruction = "Is there a toothbrush in the cup?" # textual instruction
instruction_extension = " Answer using a single word or phrase."
instruction_combined = instruction + instruction_extension
conv = conv_templates[conv_mode].copy() 
roles = conv.roles 
inp = conv.roles[0] + ': ' + instruction_combined
inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
print(prompt)

# ----------------------------sentence embedding----------------------------
sentence_embeds = sentence_embedder.encode([instruction], convert_to_tensor=True)

# ----------------------------inference----------------------------
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
with torch.inference_mode():
    output_ids = model.generate(
            inputs = input_ids,
            sentence_embeds = sentence_embeds, 
            images = image_tensor,
            image_sizes = [image_size]
        )
output_text = tokenizer.decode(output_ids[0])
print(output_text)