# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
import sys
import os
project_root = '/home/lx/LYX903_balance'
if project_root not in sys.path: 
    sys.path.insert(0, project_root)

import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import numpy as np

import transformers
import tokenizers

from sentence_transformers import SentenceTransformer
import argparse

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image

# 在分布式训练中用于指示当前进程的设备索引
local_rank = None

# 用于在local_rank为0时打印信息，在分布式训练中，只有主进程（rank 0）会打印日志信息，以避免重复输出。
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

# 解析tokenizers库的版本，并将版本信息与0.14进行比较
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

# @dataclass装饰器用于简化类的定义，特别是那些主要用于存储数据的类。
# field 是 dataclasses 提供的一个函数，允许指定默认值、元数据以及字段初始化时的行为

# 这个类包含了与模型相关的配置参数，用于指定模型的路径、冻结策略、多模态处理等
@dataclass 
class ModelArguments: 
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # 指定llm模型的名称或路径
    version: Optional[str] = field(default="v0") # 指定模型的版本号
    freeze_backbone: bool = field(default=False) # 是否冻结模型的主干网络
    tune_mm_mlp_adapter: bool = field(default=False) # 是否仅调整多模态MLP适配器的参数
    vision_tower: Optional[str] = field(default=None) # 指定视觉编码器的路径或名称 
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None) # 指定预训练的多模态MLP适配器的路径
    mm_projector_type: Optional[str] = field(default='linear') # 适配器的类型
    mm_use_im_start_end: bool = field(default=False) 
    mm_use_im_patch_token: bool = field(default=True) 
    mm_patch_merge_type: Optional[str] = field(default='flat') 
    # -------------------------------------------------------------------
    aggregate_by_average: bool = False # 预训练阶段必须设为true，对除base feature外的视觉特征进行平均融合
    aggregator_num_transformers: Optional[int] = field(default=None)
    aggregator_num_heads: Optional[int] = field(default=None)
    aggregator_hidden_dim: Optional[int] = field(default=None)
    sentence_embedder: Optional[str] = field(default=None) # sentence model路径
    sentence_embed_dim: Optional[int] = field(default=None) # 语义嵌入的特征维度
    pretrain_aggregator: Optional[str] = field(default=None) # 如果有预训练aggregator，这个参数就对应他的权重地址
    #-------------------------------------------------------------------
    vit_base_layer: Optional[int] = field(default=-2) 
    vit_aggregate_groups: Optional[str] = field(default=None) #具体的分组情况,数据格式"[[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24]]"


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."}) # 指定训练数据的路径。这是一个必需的参数，通常指向包含训练数据的文件夹或文件。
    lazy_preprocess: bool = False # 如果设为True，数据在被使用时才会进行预处理，而不是在加载时全部处理好。可以节省初始的处理时间，但可能会增加训练时的延迟。
    is_multimodal: bool = False # 指定是否处理多模态数据
    image_folder: Optional[str] = field(default=None) # 图像文件夹的路径
    image_aspect_ratio: str = 'square' # 图像预处理的方法，"pad" "square"
    # --------------------------------------------------------------------------------------------
    sentence_embed_folder: Optional[str] = field(default=None) # 句子嵌入的路径


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None) # 指定缓存目录，用于存储模型和数据的缓存文件
    optim: str = field(default="adamw_torch") # 指定优化器类型
    remove_unused_columns: bool = field(default=False) 
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton") # 指定注意力机制的实现方式
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    ) # 指定模型输入序列的最大长度
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    ) # 是否使用双重量化技术
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    ) # 指定量化的数据类型
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    ) # 指定量化时使用的位数
    lora_enable: bool = False # 如果设为 True，模型训练过程中将使用LoRA进行权重调整。
    lora_r: int = 64 
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none" # 指定LoRA中的偏置项策略，可以选择 "none"、"all" 或 "lora_only"
    mm_projector_lr: Optional[float] = None # 多模态投影器的学习率
    group_by_modality_length: bool = field(default=False) # 是否根据模态的长度对数据进行分组，可用于优化训练中的批处理效率，特别是在处理多模态数据时。 
    #---------------------------------------------
    lambda_balance: Optional[float] = None # 负载均衡损失的系数

# 在分布式训练中处理模型参数，尤其是在使用 DeepSpeed 的 ZeRO 优化器时
def maybe_zero_3(param, ignore_status=False, name=None):
    # param：函数的主要输入参数，通常是模型中的某个权重或参数
    # ignore_status：布尔型参数，默认值为 False，用于控制是否忽略 param 的状态检查
    # name：可选的字符串参数，通常用于在日志中标识参数的名称，便于调试和追踪
    from deepspeed import zero 
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    # 检查 param 是否有 ds_id 属性。如果存在，说明 param 正在被 DeepSpeed 管理
    if hasattr(param, "ds_id"): 
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        # 当参数被分片时，这个操作会将参数从不同的设备聚合到当前设备上，以便进行后续操作
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
# 用于从模型的命名参数中提取特定参数集，尤其是与 LoRA 相关的参数
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        # 只提取包含 "lora_" 关键字的参数
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        # 提取所有包含 "lora_" 或 "bias" 关键字的参数
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        # 进一步区分处理方式
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

# 用于从模型的命名参数中提取非 LoRA（Low-Rank Adaptation）相关的参数
def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

# 用于从模型的命名参数中提取与多模态（MM）适配器相关的参数
def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

# 从给定的模型中找到所有线性层的名称。它会过滤掉与多模态相关的模块名称，并返回这些线性层名称的列表
def find_all_linear_names(model):
    cls = torch.nn.Linear
    # 用于存储找到的线性层的名称
    lora_module_names = set()
    # 跳过名称中多模态相关的模块
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    # 移除lm head，原因是 lm_head 通常是输出层，可能需要特殊处理
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
        
    return list(lora_module_names)

# 根据模型和训练的具体配置，选择性地保存不同的模型部分（例如，多模态适配器，或整个模型的状态字典）
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    # 如果多模态适配器参与微调，则只保存相关的适配器权重
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter 
        # 用于后续匹配模型参数名，以便找到需要保存的多模态适配器权重
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        # 提取模型中包含 keys_to_match 列表中关键字的参数
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        # 将模型的配置保存到指定的 output_dir 目录中
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        # 确保只有主进程（local_rank == 0）或非分布式训练（local_rank == -1）时才会保存模型
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            # 如果当前输出目录是一个检查点目录，则在父目录下创建一个名为 mm_projector 的文件夹，并将权重保存为 {current_folder}.bin。
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            # 如果不是检查点目录，则直接将权重保存为 mm_projector.bin 文件
            else:  
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        # 不再运行接下来的函数内容  
        return
    
    # 如果使用了deepspeed，则按照 DeepSpeed 的保存方式进行处理
    if trainer.deepspeed:
        # 同步 CUDA 操作
        torch.cuda.synchronize()
        # 用于保存当前的模型权重和配置文件，会调用_save() 方法
        # 如果使用了 DeepSpeed 进行分布式训练，会根据 DeepSpeed 的要求进行保存，以确保模型权重能够正确保存和恢复
        trainer.save_model(output_dir)
        return
    
    # 在普通模式下，保存整个模型的状态字典
    # 首先获取模型的所有参数和权重
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        # 先移到cpu上
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        } 
        # 从gpu上移除参数
        del state_dict
        # 保存
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

# 用于调整 tokenizer 和模型的 embedding 层，以适应新的特殊tokens
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # 统计新增加的token数量
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # 调整模型的 token embedding 层的大小
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        # 获取模型的输入与输出 embedding 权重
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        # 算了现有 embedding 的平均值（排除掉新添加的标记）
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        # 将平均值用于新标记的 embedding 初始化。避免了新标记的 embedding 被随机初始化，从而可以更稳定地融入到模型中
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

# 得到文本序列的token ID以及各序列的有效长度
def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    # 生成一个包含分词结果的列表
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    # 得到分词后的 token ID 序列
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    # 计算每个序列中实际有效 token 的长度，即不包括填充 token 的长度。
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

# 在目标序列中根据发言者和分词后的长度信息对特定部分进行掩码处理（忽略某些位置的损失计算）
def _mask_targets(target, tokenized_lens, speakers): 
    # cur_idx = 0 
    # 用于跟踪当前处理的位置索引。将其设置为第一个片段的长度，意味着最开始要跳过第一个片段。
    cur_idx = tokenized_lens[0]
    # 将 tokenized_lens 列表中的第一个元素移除，更新后的列表仅包含其余片段的长度信息
    tokenized_lens = tokenized_lens[1:]
    # 对目标序列的前 cur_idx 个元素进行掩码，将它们的值设置为 IGNORE_INDEX，用于在计算损失时忽略这些位置的预测
    target[:cur_idx] = IGNORE_INDEX
    # 将人类发言部分进行掩码处理
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

# 用于在对话数据的每一轮中添加发言者标签和开始/结束信号
def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    # header：表示对话的开头部分，通常是一些引导性文本或上下文信息
    # source：这是一个包含多个句子的列表，每个句子是一个字典，包含发言者和内容信息
    # get_conversation：布尔型参数，默认值为 True。如果为 True，函数会将生成的对话内容拼接成一个完整的字符串并返回
    
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n" 
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        # 修改用户和模型在对话中的称谓
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        # 将发言者标记和开始/结束信号添加到句子的内容中
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

# 用于处理多模态数据中与图像相关的token
def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict: 
    # 此时的输入：sources = [[{from: human, value:..}, {from: GPT, value:...}, ....]]
    
    # 如果数据不包含多模态内容，直接返回原始的sources，不进行任何处理
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    
    for source in sources:
        # 遍历 source 中的每个句子
        for sentence in source:
            # sentence 是一个字典 {from: ...., value: ....}
            if DEFAULT_IMAGE_TOKEN in sentence['value']: 
                # 先将 "<image>" 从句子中移除并进行清理 (首轮对话中已经包含了<image>，但是位置不定)
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                # 将 DEFAULT_IMAGE_TOKEN 插入到句子的开头并加上换行符，以确保图像标记与文本内容分隔开。
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                # 清理字符串两端的空格或换行符
                sentence['value'] = sentence['value'].strip()
                # 将 DEFAULT_IMAGE_TOKEN 包裹在 <Image> 标签中。也就是将'<image>'替换为'<Image><image></Image>'
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            # 根据图像token的预设模式，看看是否进一步替换原有的DEFAULT_IMAGE_TOKEN
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                # 将 <image> 替换为 <im_start><image><im_end>
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    
    # 初始化对话，默认的conv类型是conv_vicuna_v1
    conv = conversation_lib.default_conversation.copy()
    # 在数据中的角色称谓与conv中要求的称谓之间建立映射关系
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates 构建提示模版
    conversations = []
    for i, source in enumerate(sources):
        # source的样子 [{'from': 'human', 'value': ...}, {'from': 'gpt', 'value': ...}]
        # 如果对话的第一句不是由人类发起，则跳过这句话。因为模板要求对话由人类开始。
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        # 将每轮对话的发言者和内容按照模板添加到 conv.messages 中
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        # 生成完整的提示
        conversations.append(conv.get_prompt())

    # Tokenize conversations 对提示内容进行分词
    
    # 如果包含图像数据，则使用 tokenizer_image_token 进行分词，并将结果堆叠成张量
    if has_image: 
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    # 如果不包含图像数据，则使用 tokenizer 对纯文本内容进行分词
    else: 
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    # 将 input_ids 克隆一份，作为模型的标签
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets 对目标序列进行掩码处理
    # sep=" " sep2="</s>"
    sep = conv.sep + conv.roles[1] + ": " # 人类指令和机器回复之间的字符 " ASSISTANT: "
    for conversation, target in zip(conversations, targets):
        # 计算有效的token数量
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        
        # 将每轮对话中的指令部分进行掩码
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else: 
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        # source的样子 [{'from': 'human', 'value': ...}, {'from': 'gpt', 'value': ...}]
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        # 将 human 的文本替换为了 <image>
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    # 把机器生成的描述文本外的token id设为igore_index
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer)) # <image> 被分成了几个token
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together; 
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer) 
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

# 用于支持惰性加载的监督学习数据集，继承自torch.utils.data
# 懒惰是指数据的加载和预处理被延迟到每次调用 __getitem__ 时进行，以便处理大型数据集时减少内存消耗
class LazySupervisedDataset(Dataset): 
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        
        # 从文件中加载数据，返回一个列表，每个元素是一个字典
        list_data_dict = json.load(open(data_path, "r")) 
        
        # 每个元素的样子
        # {'id': '004539375',  
        #  'image': '00453/004539375.jpg', 
        #  'conversations': [{'from': 'human', 'value': 'Render a clear and concise summary of the photo.\n<image>'},{'from': 'gpt','value': 'select luxury furniture 3 - inch gel memory foam mattress topper'}]}
        
        # 由主进程打印信息，表示数据将在惰性加载时格式化
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict 
        self.data_args = data_args 
    
    # 训练数据集中的样本数量
    def __len__(self):
        return len(self.list_data_dict)

    # 计算每个样本的序列长度，同时考虑文本和图像的token数量。如果样本中包含图像，则增加 128 个 token
    @property
    def lengths(self):
        length_list = [] 
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list
    
    # 类似于上面的lengths，但如果样本包含图像，长度为正值，否则为负值。可能用于对样本按模态进行区分
    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list 

    # 获取单个样本的数据，最终得到一个字典，包含image、input ids、labels三部分
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # 得到第i个样本，是一个字典，包括id、image、conversations三个元素
        sources = self.list_data_dict[i]
        #print(sources["conversations"])
        if isinstance(i, int): 
            sources = [sources] 
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        # 处理多模态训练样本
        if 'image' in sources[0]: 
            # 获取图像文件名
            image_file = self.list_data_dict[i]['image']
            # 获取图像文件夹路径 
            image_folder = self.data_args.image_folder
            # 获取图像预处理器 （CLIP自带的）
            processor = self.data_args.image_processor
            # 加载图像并进行预处理
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
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
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:  
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            # 在第一轮对话的用户问题前加上“<image>/n”
            sources = preprocess_multimodal( 
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args) 
            # 此时的sources是[[{"from": ..., "value":... },..., {"from": ..., "value":... }]]
            
            # 接下来要得到当前样本的 input_ids 和 labels
            if self.data_args.version != 'plain':
                # 指令微调且为多模态样本
                data_dict = preprocess(sources, self.tokenizer, has_image=('image' in self.list_data_dict[i]))
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
                # ----------------------------
                sentence_embed_folder = self.data_args.sentence_embed_folder
                sentence_embed_file = self.list_data_dict[i]['sentence_embed']
                sentence_embed_tensor = torch.load(os.path.join(sentence_embed_folder, sentence_embed_file))
                data_dict['sentence_embed'] = sentence_embed_tensor 
                # -----------------------------
                data_dict['image'] = image 
            else:
                # 预训练（一定是多模态样本）
                data_dict = preprocess(sources, self.tokenizer, has_image=('image' in self.list_data_dict[i]))
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]) 
                data_dict['image'] = image           

        # 处理纯文本训练样本，注意：只有指令微调时才会出现纯文本样本
        else:  
            # 指令微调且为纯文本样本
            sources = copy.deepcopy([e["conversations"] for e in sources])
            # 此时的sources是[[{"from": ..., "value":... },..., {"from": ..., "value":... }]]
            # 得到第i个样本的 input_ids 和 labels
            data_dict = preprocess(sources, self.tokenizer, has_image=('image' in self.list_data_dict[i]))
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]) 
            # 则填充一个全零的句子嵌入，保持输入格式的一致性
            data_dict['sentence_embed'] = torch.zeros(self.data_args.sentence_embed_dim)
            if self.data_args.is_multimodal:
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
                
        # 最终得到一个字典，包含image、input ids、labels三部分，每部分的值都是一个tensor: 
        # data_dict["image"]   (3, 336, 336)
        # data_dict["input_ids"]  (seq_len, )
        # data_dict["labels"]  (seq_len, )
        # data_dict["sentence_embed"] (768, )
        
        # if (data_dict["input_ids"] == IMAGE_TOKEN_INDEX).sum() > 1:
        #     print(self.list_data_dict[i]['id'])
        
        return data_dict 

# 用于在监督微调过程中将多个样本整理成一个batch，batch形式是字典，包含images、input ids、labels、attention masks四部分
@dataclass 
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    
    # __call__ 方法使得这个类的实例可以像函数一样调用
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 提取batch当中的 input_ids 和 labels 
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        # 根据batch当中的最大序列长度填充 input_ids 和 labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id) # (batch_size, batch_max_seq_len)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX) # (batch_size, batch_max_seq_len)
        
        # 截断：为了确保输入序列和标签序列的长度不超过模型的最大长度 
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        # 构建批次字典，包含了 input_ids、labels 和 attention_mask。attention_mask是一个布尔掩码，指示哪些 token 是实际输入，哪些是填充值
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        # 将图像也加入到批次数据中
        if 'image' in instances[0]:
            # 如果所有图像的形状相同，则使用 torch.stack 将它们堆叠成一个批次张量
            images = [instance['image'] for instance in instances]
            # 如果所有图像的形状相同，则使用 torch.stack 将它们堆叠成一个批次张量
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            # 否则直接返回图像列表而不进行堆叠
            else: 
                batch['images'] = images 
        if 'sentence_embed' in instances[0]:
            batch['sentence_embeds'] = torch.stack([instance['sentence_embed'] for instance in instances]) # (batch_size, embed_dim)
        
        # batch 是一个字典，包含了 input_ids、labels、 attention_mask、images、sentence_embeds
        # input_ids (batch_size, batch_max_seq_len)
        # labels (batch_size, batch_max_seq_len)
        # attention_mask (batch_size, batch_max_seq_len)
        # sentence_embeds (batch_size, 768) 
        return batch

# 初始化数据集和整理器，并将它们打包成字典返回
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict: 
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator) 

###########################################################
###########################################################
###########################################################
# 主训练函数，输入的attn_implementation用于指定注意力机制的实现方式
def train(attn_implementation=None):

    # 将 local_rank 定义为全局变量，表示当前进程在分布式训练中的位置
    global local_rank
    
    # 使用 Hugging Face 的 HfArgumentParser 解析命令行参数
    parser = transformers.HfArgumentParser( 
        (ModelArguments, DataArguments, TrainingArguments))

    
    # 将命令行参数打包到之前定义的ModelArguments, DataArguments, TrainingArguments类中
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank 
    
    # 根据训练配置，设置计算精度
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    # 量化模型的配置 
    bnb_model_from_pretrained_args = {} 
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"], # 应该是因为适配器在低精度下表现不佳，因此保留其的高精度参数
                llm_int8_threshold=6.0, # 用来判断某些权重是否需要高精度存储
                llm_int8_has_fp16_weight=False, # 是否在使用 8-bit 量化时保留 16-bit 的权重副本
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant, # 是否使用双重量化技术
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'} 
            )
        ))

    # 加载模型
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else: 
            # 从LLM的模型文件中去实例化LlavaLlamaForCausalLM
            # 此时model.config里只有LLM config中的那些配置参数
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path, # LLM组件的模型文件
                cache_dir=training_args.cache_dir, 
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir, 
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    
    # 关闭模型的缓存功能：确保前向传播时重新计算模型的所有隐藏状态，而不使用之前计算的缓存
    model.config.use_cache = False

    # 是否冻结模型的LLM（此时只加载进来了LLM）
    if model_args.freeze_backbone:
        model.model.requires_grad_(False) 

    # 如果使用了 4-bit 或 8-bit 量化，使用 PEFT 库进行额外的配置，如梯度检查点和计算精度
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # 启用梯度检查点时，不会存储所有层的激活值，而是只存储少部分关键层的激活值。对于未存储激活值的层，在反向传播时需要重新计算
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads() 
        else: 
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True) 
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 如果启用了 LoRA，配置并添加 LoRA 适配器
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model), 
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # 加载分词器
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        # 采用的是LLM模型文件夹中的tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, 
            cache_dir=training_args.cache_dir, 
            model_max_length=training_args.model_max_length, # 对于vicuna1.5来讲就是2048
            padding_side="right",
            use_fast=False, # 快速分词器是用 Rust 语言实现的，具有更高的效率
        )
    
    # 根据对话版本设置特殊 token（如 [PAD]），并调整 tokenizer 和模型的嵌入层
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else: 
        # 使用tokenizer中自定义的<unk>来填充序列
        tokenizer.pad_token = tokenizer.unk_token # <UNK> 
        # 加载对话模版，预训练时 version 为 "plain"，微调时version 为 "v1"
        if model_args.version in conversation_lib.conv_templates:
            # 返回的是一个Conversation类
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else: 
            # 返回的是一个Conversation类
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # 如果训练文件中包含模块编码器的路径，则初始化视觉编码器和适配器
    if model_args.vision_tower is not None:
        
        model.get_model().initialize_vision_modules(model_args=model_args,fsdp=training_args.fsdp)
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        
        data_args.sentence_embed_dim = model_args.sentence_embed_dim
        data_args.version = model_args.version
        
        # 在模型配置文件中加入与数据处理相关的参数
        model.config.image_aspect_ratio = data_args.image_aspect_ratio # pad 或 直接resize到CLIP支持的分辨率
        model.config.tokenizer_padding_side = tokenizer.padding_side # 从左侧或右侧对batch中长度不一的序列进行拼接
        model.config.tokenizer_model_max_length = tokenizer.model_max_length # 上下文窗口
        
        # 是否仅调整mlp适配器的参数
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter: 
            # 仅仅解冻MLP模块，其它部分的参数要冻结上 
            model.requires_grad_(False) 
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True  
        
        # 是否冻结mlp适配器的参数 
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            # 冻结MLP模块的参数
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        # 根据模型配置参数决定是否添加新的图像token，并调整嵌入层的大小
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # 在量化训练中，调整特定层（如 LoRA 层、规范化层）需要特殊处理，以确保它们在正确的精度下运行
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # 初始化数据模块和训练器，返回一个字典，里面包括train_dataset，eval_dataset，data collator
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    # LLaVATraine r继承自 Trainer，是 Hugging Face transformers 库中用于简化模型训练的接口
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # 如果检测到已有的检查点，恢复训练；否则从头开始训练
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")): 
        print("=========================================================")
        print("resume from checkpoint")
        trainer.train(resume_from_checkpoint=True)  
    else:  
        trainer.train() 
    
    # 保存训练状态为 trainer_state.json 文件，其中包含：
    # 1. 优化器状态
    # 2. 学习率调度器状态
    # 3. 当前的随机数生成器状态
    trainer.save_state()
    
    # 重新打开缓存功能，以便在推理过程中提高生成速度
    model.config.use_cache = True

    # 如果启用了 LoRA，保存模型的 LoRA 参数和非 LoRA 参数；如果没有启用 LoRA，使用标准方法保存模型。
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        ) 
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else: 
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
