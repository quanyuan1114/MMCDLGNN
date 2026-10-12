import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by number of heads"

        self.query_fc = nn.Linear(embed_size, embed_size)
        self.key_fc = nn.Linear(embed_size, embed_size)
        self.value_fc = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask=None):
        batch_size = query.shape[0]

        # 计算 Q, K, V 矩阵
        queries = self.query_fc(query).view(batch_size, self.num_heads, self.head_dim)  # (batch_size, num_heads, head_dim)
        keys = self.key_fc(keys).view(batch_size, self.num_heads, self.head_dim)        # (batch_size, num_heads, head_dim)
        values = self.value_fc(values).view(batch_size, self.num_heads, self.head_dim)  # (batch_size, num_heads, head_dim)

        # 计算注意力分数
        energy = torch.einsum("bhd,bhd->bh", [queries, keys])  # (batch_size, num_heads)
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)  # (batch_size, num_heads)

        # 应用注意力权重
        out = torch.einsum("bh,bhd->bhd", [attention, values]).contiguous()
        out = out.view(batch_size, -1)  # (batch_size, embed_size)

        return self.fc_out(out)


class CoAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CoAttention, self).__init__()
        self.multihead_attention = MultiHeadAttention(embed_size, num_heads)

    def forward(self, text_features, visual_features):
        # Textual multi-head self-attention
        Zt = self.multihead_attention(text_features, text_features, text_features)  # (batch_size, embed_size)

        # Visual multi-head self-attention
        Zv = self.multihead_attention(visual_features, visual_features, visual_features)  # (batch_size, embed_size)

        # Text-Visual Cross-modal Co-attention
        Qiv = self.multihead_attention.query_fc(Zv)  # (batch_size, embed_size)
        Ki_t = self.multihead_attention.key_fc(Zt)   # (batch_size, embed_size)
        V_i_t = self.multihead_attention.value_fc(Zt)  # (batch_size, embed_size)
        Zivt = self.multihead_attention(V_i_t, Ki_t, Qiv)  # (batch_size, embed_size)

        # Visual-Textual Cross-modal Co-attention
        Qit = self.multihead_attention.query_fc(Zt)  # (batch_size, embed_size)
        Ki_v = self.multihead_attention.key_fc(Zv)   # (batch_size, embed_size)
        V_i_v = self.multihead_attention.value_fc(Zv)  # (batch_size, embed_size)
        Zitv = self.multihead_attention(V_i_v, Ki_v, Qit)  # (batch_size, embed_size)

        return Zivt, Zitv


if __name__ == '__main__':
    # 示例用法
    embed_size = 4
    num_heads = 2
    batch_size = 2

    # 假设输入的文本特征和视觉特征
    text_features = torch.randn(batch_size, embed_size)  # (batch_size, embed_size)
    visual_features = torch.randn(batch_size, embed_size)  # (batch_size, embed_size)

    # 初始化 Co-attention 层
    co_attention = CoAttention(embed_size, num_heads)

    # 前向传播
    Zivt, Zitv = co_attention(text_features, visual_features)
    print("Zivt:", Zivt)
    print("Zivt shape:", Zivt.shape)  # (batch_size, embed_size)
    print("Zitv shape:", Zitv.shape)  # (batch_size, embed_size)
