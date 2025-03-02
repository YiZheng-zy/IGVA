import torch
import torch.nn as nn
import torch.nn.functional as F

class Aggregator(nn.Module):
    def __init__(self, n_layers, feature_dim, embed_dim, hidden_dim, num_transformers, num_heads):
        super(Aggregator, self).__init__()
        
        # 将 sentence_embed 和 cls_features 分别映射到 hidden_dim
        self.sentence_proj = nn.Linear(embed_dim, hidden_dim)
        self.cls_proj = nn.Linear(feature_dim, hidden_dim)
        
        # 生成多个 QueryTransformer，每个都处理映射后的 hidden_dim
        self.transformers = nn.ModuleList([
            QueryTransformer(hidden_dim, num_heads) for _ in range(num_transformers)
        ])
        
        # 线性映射到 n_layers 维度，用于生成权重
        self.linear = nn.Linear(hidden_dim, n_layers)

    def forward(self, cls_features, sentence_embed): 
        # 首先对 sentence_embed 和 cls_features 进行映射
        sentence_embed_proj = self.sentence_proj(sentence_embed)  # (batch_size, hidden_dim)
        cls_features_proj = self.cls_proj(cls_features)  # (batch_size, n_layers, hidden_dim)
        #print(f"sentence_embed_proj shape: {sentence_embed_proj.shape}")
        #print(f"cls_features_proj shape: {cls_features_proj.shape}")
        # 使用映射后的特征进行 QueryTransformer 的计算
        x = sentence_embed_proj.unsqueeze(1)
        for transformer in self.transformers:
            x = transformer(x, cls_features_proj)
        
        # 映射到 n_layers 维度，并使用 softmax 进行归一化
        weights = self.linear(x)  # (batch_size, 1, n_layers)
        weights = F.softmax(weights, dim=-1)
        weights = weights.squeeze(1) # (batch_size, n_layers)

        return weights


class QueryTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(QueryTransformer, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be a multiple of num_heads"
        
        # Cross-attention 模块，输入和输出都是 hidden_dim
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # 前馈网络，用于进一步处理 Attention 输出
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # LayerNorm 层
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, sentence_embed, cls_features):
        # sentence_embed: (batch_size, hidden_dim)
        # cls_features: (batch_size, n_layers, hidden_dim)

        # 将 sentence_embed 作为 query
        query = sentence_embed  # (batch_size, 1, hidden_dim)

        # cls_features 作为 key 和 value
        key_value = cls_features  # (batch_size, n_layers, hidden_dim)

        # 执行 Cross-attention
        #print(f"query shape: {query.shape}")
        #print(f"key_value shape: {key_value.shape}")
        attn_output, _ = self.cross_attention(query, key_value, key_value)
        attn_output = attn_output.squeeze(1)  # (batch_size, hidden_dim)
        #print(f"attn_output shape: {attn_output.shape}")
        # 加入残差连接和 LayerNorm
        attn_output = self.norm1(attn_output + sentence_embed.squeeze(1))

        # 前馈网络处理
        ffn_output = self.ffn(attn_output)
        output = self.norm2(ffn_output + attn_output)
        #print(f"output shape: {output.shape}")
        return output.unsqueeze(1)
