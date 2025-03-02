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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig
    
    def __init__(self, config: LlamaConfig):
        #print("LlavaLlamaModel is being initialized")
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        #print("LlavaLlamaForCausalLM is being initialized")
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        # 在模型的预训练过程中，Transformer 层的张量并行的程度
        self.pretraining_tp = config.pretraining_tp 
        self.vocab_size = config.vocab_size 
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        sentence_embeds: Optional[torch.FloatTensor] = None,  # 新增参数，用户指令的句子嵌入
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, 
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 生成用于模型推理的嵌入和标签
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids = input_ids,
                sentence_embeds = sentence_embeds, # 新增参数
                position_ids = position_ids,
                attention_mask = attention_mask,
                past_key_values = past_key_values,
                labels = labels,
                images = images,
                image_sizes = image_sizes
            )
        # 最终调用父类的 forward 方法进行标准的前向传播
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict 
        )
    
    # 用于生成文本输，使用 @torch.no_grad() 装饰器，以确保在推理时不计算梯度，减少内存使用
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        sentence_embeds: Optional[torch.Tensor] = None,  # 新增参数
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        # 获得位置编号和attention mask
        # pop() 是字典对象的一个方法，用于移除并返回字典中指定键的值。如果指定的键不存在，则返回一个默认值（这里是 None）。
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs: 
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        # 如果有图像输入，会将文本和图像输入整合为适合模型处理的嵌入（inputs_embeds）
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images,
                sentence_embeds = sentence_embeds,
                image_sizes=image_sizes
            )
        # 如果没有图像输入，则直接使用大语言模型对文本输入进行前乳
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        # 调用LlamaForCausalLM的生成函数
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

# 将 LlavaConfig 和 LlavaLlamaForCausalLM 注册到 transformers 库中，使得这些类可以通过配置文件中的model_type被自动加载和使用。
AutoConfig.register("llava_llama", LlavaConfig) 
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM) 
