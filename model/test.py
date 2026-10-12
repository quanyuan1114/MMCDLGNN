import torch
import torch.nn as nn
import torch.nn.functional as F

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

        queries = self.query_fc(query)  # 形状应为 (batch_size, seq_length, embed_size)
        keys = self.key_fc(keys)  # 形状应为 (batch_size, seq_length, embed_size)
        values = self.value_fc(values)  # 形状应为 (batch_size, seq_length, embed_size)

        queries = queries.view(batch_size, -1, self.num_heads,
                               self.head_dim)  # (batch_size, seq_length, num_heads, head_dim)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim)  # (batch_size, seq_length, num_heads, head_dim)
        values = values.view(batch_size, -1, self.num_heads,
                             self.head_dim)  # (batch_size, seq_length, num_heads, head_dim)

        queries = queries.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
        keys = keys.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
        values = values.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)

        energy = torch.einsum("bhqd,bhkd->bhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)
        out = torch.einsum("bhqk,bhvd->bhqd", [attention, values])

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.fc_out(out)

        return out


class CoAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CoAttention, self).__init__()
        self.multihead_attention = MultiHeadAttention(embed_size, num_heads)

    def forward(self, text_features, visual_features):
        # Textual multi-head self-attention
        Q_t = self.multihead_attention.query_fc(text_features)  # (batch_size, seq_length, embed_size)
        K_t = self.multihead_attention.key_fc(text_features)  # (batch_size, seq_length, embed_size)
        V_t = self.multihead_attention.value_fc(text_features)  # (batch_size, seq_length, embed_size)
        Zt = self.multihead_attention(V_t, K_t, Q_t)  # 执行自注意力操作

        # Visual multi-head self-attention
        Q_v = self.multihead_attention.query_fc(visual_features)  # (batch_size, seq_length, embed_size)
        K_v = self.multihead_attention.key_fc(visual_features)  # (batch_size, seq_length, embed_size)
        V_v = self.multihead_attention.value_fc(visual_features)  # (batch_size, seq_length, embed_size)
        Zv = self.multihead_attention(V_v, K_v, Q_v)  # 执行自注意力操作

        # Text-Visual Cross-modal Co-attention (replace Ri_t with Zi_v for Qiv)
        # 通过替换 Ri_t 为 Zi_v 来计算 Qiv
        Qiv = self.multihead_attention.query_fc(Zv)  # Qiv = W_Qv * Zi_v
        Ki_t = self.multihead_attention.key_fc(Zt)  # Ki_t = W_Kt * Zi_t
        V_i_t = self.multihead_attention.value_fc(Zt)  # Ki_t = W_Kv * Zi_t

        # Cross-modal Co-attention (Visual-Textual)
        Zivt = self.multihead_attention(V_i_t, Ki_t, Qiv)

        # Visual-Textual Cross-modal Co-attention (replace Ri_v with Zi_t for Qit)
        # 通过替换 Ri_v 为 Zi_t 来计算 Qit
        Qit = self.multihead_attention.query_fc(Zt)  # Qit = W_Qt * Zi_t
        Ki_v = self.multihead_attention.key_fc(Zv)  # Ki_k = W_Kv * Zi_v
        V_i_v = self.multihead_attention.value_fc(Zv)  # Ki_v = W_Kv * Zi_v

        Zitv = self.multihead_attention(V_i_v, Ki_v, Qit)

        return Zivt, Zitv
if __name__ == '__main__':
    # 示例用法
    embed_size = 4
    num_heads = 2
    batch_size = 2
    seq_length = 3

    # 假设输入的文本特征和视觉特征
    text_features = torch.randn(batch_size, seq_length, embed_size)  # (batch_size, seq_length, embed_size)
    visual_features = torch.randn(batch_size, seq_length, embed_size)  # (batch_size, seq_length, embed_size)
    print(text_features)
    # 初始化 Co-attention 层
    co_attention = CoAttention(embed_size, num_heads)

    # 前向传播
    Zivt, Zitv = co_attention(text_features, visual_features)
    print(Zivt)
    print(Zivt.shape)  # (batch_size, seq_length, embed_size)
    print(Zitv.shape)  # (batch_size, seq_length, embed_size)