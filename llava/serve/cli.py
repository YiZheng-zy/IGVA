import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

# 从本地或网络中加载与对话相关的图像，输出是一个PIL.Image.Image的类
def load_image(image_file):
    
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # args代表在命令行开启推理时输入的配置参数
    
    # 禁止模型实例化时的参数初始化，减少加载模型时的计算开销
    disable_torch_init()
    
    # 从模型权重路径中解析出模型版本名字，如llava-v1.5-7b
    model_name = get_model_name_from_path(args.model_path)
    
    # 根据模型版本加载对应的预训练模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    # tokenizer - 对应预训练模型的分词器
    # model - 主要的多模态模型（视觉编码器+适配器+LLM），由LlavaLlamaForCausalLM.from_pretrained(）获得
    # image_processor - 图像预处理器，用于将图像加载并处理到CLIP可以接受的形式
    # context_len - llava的上下文窗口长度
    
    # 根据模型版本定义与用户的对话模型
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    
    # 若自动分配的对话模型与用户输入的不一致，以用户输入的为准
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode
        
    # 找到与对话模型对应的对话模版，输出的是一个Conversation的类
    conv = conv_templates[args.conv_mode].copy() # conv_templates是一个字典，定义了从conv_mode到conversation类的映射
    # 找到对话模型中AI和用户的名称，最常用的llava-v1当中 roles = ("USER", "ASSISTANT")
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    # 从本地或网络中加载与对话相关的图像，输出是一个PIL.Image.Image的类
    image = load_image(args.image_file)
    image_size = image.size
    
    # Similar operation in model_worker.py
    # 详细定义了如何对输入图像进行预处理，这里的预处理包括在输入image_processor前进行的处理，包括了子图划分
    image_tensor = process_images([image], image_processor, model.config) 
    if type(image_tensor) is list: 
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else: 
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    # 开始对话
    while True:
        try: 
            inp = input(f"{roles[0]}: ") # 返回的就是"role0的称谓："
        except EOFError:
            inp = ""
        if not inp: 
            print("exit...")
            break
        
        print(f"{roles[1]}: ", end="")
        
        # 确定为图像所设置的特殊token 
        if image is not None:
            # first message 
            if model.config.mm_use_im_start_end: 
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else: 
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None
        
        # 把用户的当前输入和AI的称谓输入对话记录
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        
        # 生成最新的上下文内容/提示，里面包含有和图像相关的特殊token
        prompt = conv.get_prompt() 
        
        # 将包含 <image> token的文本进行分词处理，转换为适合模型输入的 input id
        # 注意这里并不考虑图像内容！因为视觉特征是连续空间内的向量，在词汇表中并没有对应的token
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        # 根据对话的分隔符风格，选择合适的停止字符串 
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords 通常用于确定生成文本何时应该停止 
        keywords = [stop_str] 
        # 一个文本流处理器对象，用于在生成文本时逐步输出，提供实时的生成反馈。
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, # 经过了分词的文本上下文（也包含特殊的token）
                images=image_tensor, # 经过处理后的图像序列
                image_sizes=[image_size], # 原始图像的尺寸
                do_sample=True if args.temperature > 0 else False, # 每一步是否进行随机采样
                temperature=args.temperature, # 采样时的温度
                max_new_tokens=args.max_new_tokens, # 生成的最大新token数量
                streamer=streamer,
                use_cache=True)
            # output_ids的形状是(batch_size, seq_len)
            
        # 将生成的token序列转换为正常的文本，并去除首尾的空白字符
        outputs = tokenizer.decode(output_ids[0]).strip()
        # 更新对话历史 
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 通常只有model-path和image-file是需要指定的
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True) #可以是网站链接也可以是本地路径
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()  
    main(args) 

# git 上给出的 推理命令 python -m llava.serve.cli \
# --model-path liuhaotian/llava-v1.5-7b \
# --image-file "https://llava-vl.github.io/static/images/view.jpg" \
# --load-4bit 