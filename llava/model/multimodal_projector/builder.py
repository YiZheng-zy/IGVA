import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    # 如果投影器类型是 'linear'，返回一个线性层
    if projector_type == 'linear':
        return nn.Linear(2*config.mm_hidden_size, config.hidden_size)
    
    # 如果投影器类型是'mlpNx_gelu'，返回一个由GELU激活的N层MLP
    # 首先检查投影器类型是否符合 'mlpNx_gelu' 字符串模式
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        # 提取 MLP 的深度
        mlp_depth = int(mlp_gelu_match.group(1))
        # 构建符合层数的MLP网络
        modules = [nn.Linear(2*config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    
    #如果投影器类型是 'identity'，返回一个IdentityMap，这个投影器不会改变输入特征，直接将输入作为输出。
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
