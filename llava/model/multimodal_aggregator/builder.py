import ast
from .aggregator import *

def build_vision_aggregator(config, **kwargs):
    
    # n_layers 代表需要融合的视觉特征的组数，注意是组数！！
    return Aggregator(n_layers =len(ast.literal_eval(config.vit_aggregate_groups)), # softmax映射到的维度
                      feature_dim = kwargs.get('feature_dim') ,  # 视觉特征的维度
                      embed_dim = kwargs.get('embed_dim'), # 语义嵌入的维度
                      num_transformers = config.aggregator_num_transformers, # transformer模块的数量
                      hidden_dim=config.aggregator_hidden_dim, # transformer进行计算的hidden_dim
                      num_heads=config.aggregator_num_heads # 每个transformer模块有多少个注意力头
                      )    
    