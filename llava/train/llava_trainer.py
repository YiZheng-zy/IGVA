import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer 
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional

# 见 train.py
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# 见 train.py
def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

# 将一组索引按照长度进行合理分割，确保每个部分的总长度尽可能相等
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """
    # 如果 indices 的长度无法被 num_chunks 整除，那么使用简单的切片方法，将 indices 按照步长为 num_chunks 的方式分割
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)] 
    
    # 在可以均匀分割的情况下，计算每个部分应该包含的索引数量
    num_indices_per_chunk = len(indices) // num_chunks
    # 将索引分配到最短的部分
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        # 确保每个部分数量限制
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks

# 将包含不同模态的索引列表按照样本长度分组，并生成一个适合分布式训练的随机化索引列表
def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # lengths：这是一个列表，包含数据集中的样本长度
    # batch_size：每个批次的大小
    # world_size：指训练时分布式训练的总进程数，用于划分批次
    # generator：一个可选的随机数生成器，用于控制随机性
    
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    
    # 检查是否有长度为0的样本
    assert all(l != 0 for l in lengths), "Should not have zero length."
    
    # 如果所有样本的长度都大于 0 或都小于 0，表示样本都属于同一模态（多模态或语言模态）
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality 
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    
    # 提取多模态样本的索引及其对应的长度
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    # 提取语言模态样本的索引及其对应的长度
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    # 分别对多模态和语言模态的数据进行长度分组，并生成随机化后的索引列表
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    
    # 将分组后的索引按大批次（mega batch）划分
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    # 将最后一个多模态和语言模态的大批次合并成一个additional batch，以便后续处理
    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    # 将其余的大批次合并，去掉最后一个大批次
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    
    # 对大批次进行随机化，以打乱它们的顺序（注意是对大批次进行随机打乱，而不是对每个小批次中的样本顺序进行打乱）
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]
    
    # 如果 additional_batch 中有内容，则将其排序后附加到 megabatches 中 
    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))
    
    # 大批次展开为一个完整的索引列表
    return [i for megabatch in megabatches for i in megabatch]

# 将一个包含长度信息的索引列表随机化，并将其分组为适合分布式训练的批次
# 同时根据样本长度对每个批次内的样本进行排序，以便更好地平衡每个 GPU 或处理器的工作负载
# 最终返回一个经过随机化、分组和排序的索引列表。
def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    # 生成一个长度为 len(lengths) 的随机排列索引列表
    indices = torch.randperm(len(lengths), generator=generator)
    # 计算大批次的大小 （每个批次的大小*分布式训练中的进程数）
    megabatch_size = world_size * batch_size
    # 将随机化后的索引列表按 megabatch_size 划分为多个大批次，并将这些索引转换为列表形式
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    # 对于每个大批次内部的样本，根据样本的长度（lengths[i]）进行降序排序，有助于优化训练效率
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    # 将大批次进一步划分为 world_size 个小批次，确保每个小批次的长度尽可能均匀分布
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    # 将嵌套的 megabatches（大批次、小批次）展开为一个平坦的索引列表
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


# 用于在数据加载过程中，根据样本长度对数据进行分组采样，同时保持一定的随机性
class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size # 每个批次的样本数量
        self.world_size = world_size # 分布式训练中的进程数（通常对应 GPU 数量）
        self.lengths = lengths # 样本长度列表
        self.generator = generator # 可选的随机数生成器，用于控制随机性，确保在分布式环境中各进程的随机性一致
        self.group_by_modality = group_by_modality # 是否按照模态（纯语言 vs 多模态）分组

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:  
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices) 

# 这个 LLaVATrainer 类继承自 Hugging Face 的 Trainer 类，并对部分方法进行了自定义，以适应多模态模型训练的需求
class LLaVATrainer(Trainer): 

    # 用于获取训练数据集的采样器（sampler）
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # 如果 train_dataset 为空或没有长度信息，则返回 None
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None 
        # 如果设置了 group_by_modality_length 参数，则使用 LengthGroupedSampler 进行按模态分组采样，确保每个批次内的数据长度尽可能均衡
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size, 
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths, 
                group_by_modality=True,
            )
        # 否则，调用父类的 _get_train_sampler 方法，使用默认的采样器
        else:
            return super()._get_train_sampler()

    # 设置优化器，处理参数分组、权重衰减（weight decay）等配置
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """ 
        # 如果使用了 Amazon SageMaker 的模型并行，则调用父类的方法
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        # 对于不同的参数应用不同的训练配置
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            # optimizer_cls代表优化器的类（比如 AdamW、SGD 等）
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            # 初始化所选中的优化器 
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
            # 如果使用 Adam8bit 优化器（通过 bitsandbytes 库实现的低精度优化器），进一步配置优化器，特别是对嵌入层进行特殊处理
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    # 负责保存整个训练过程的检查点，包括模型的当前状态、优化器状态、调度器状态、随机数生成器状态等，为中途恢复训练设计的
    def _save_checkpoint(self, model, trial, metrics=None):
        # trial：一个可选的 trial 对象，用于超参数优化过程中标识当前试验的上下文信息
        
        # 只需要保存与多模态适配器相关的权重，而不需要保存整个模型
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # 将前缀与步数结合起来，生成检查点文件夹名称，如 "checkpoint-1000"
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            # 将检查点文件夹路径与试验目录结合，确定完整的输出路径
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            # 定义需要保存的参数关键字列表
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            # 在分布式训练环境中，只有主进程（local_rank == 0 或 local_rank == -1）负责保存检查点
            if self.args.local_rank == 0 or self.args.local_rank == -1: 
                # 将模型的配置文件（config）保存到 output_dir 中
                self.model.config.save_pretrained(output_dir) 
                # 将提取的多模态适配器相关权重（weight_to_save）保存为 mm_projector.bin 文件
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        # 如果并不是只微调多模态适配器，直接调用父类的 _save_checkpoint 方法，会按正常流程保存整个模型和优化器的状态
        else:
            # # 在模型保存之前，检查并调整 generation_config
            # if hasattr(model, 'generation_config'):
            #     print("yes 1111111111111111111111111111111")
            #     # 如果 do_sample 为 False，禁用 temperature 和 top_p
            #     if not model.generation_config.do_sample:
            #         model.generation_config.temperature = 1.0
            #         model.generation_config.top_p = 1.0
            #     else: 
            #         # 否则可以根据需要调整这些参数
            #         model.generation_config.temperature = 0.9  # 如果需要使用 temperature
            #         model.generation_config.top_p = 0.6  # 如果需要使用 top_p
            # else:
            #     print("no 1111111111111111111111111111111")        
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
    
    # 负责将模型的参数和权重保存到磁盘，用于模型的加载和推理
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            # 在模型保存之前，检查并调整 generation_config
            if hasattr(self.model, 'generation_config'):
                # 如果 do_sample 为 False，禁用 temperature 和 top_p
                if not self.model.generation_config.do_sample:
                    self.model.generation_config.temperature = 1.0
                    self.model.generation_config.top_p = 1.0
                else:  
                    # 否则可以根据需要调整这些参数
                    self.model.generation_config.temperature = 0.9  # 如果需要使用 temperature
                    self.model.generation_config.top_p = 0.6  # 如果需要使用 top_p      
            super(LLaVATrainer, self)._save(output_dir, state_dict)
