import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalCoAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(CrossModalCoAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        # 线性变换
        self.W_Q_t = nn.Linear(input_dim, self.hidden_dim)
        self.W_K_t = nn.Linear(input_dim, self.hidden_dim)
        self.W_V_t = nn.Linear(input_dim, self.hidden_dim)
        self.W_O_t = nn.Linear(hidden_dim, input_dim)

        self.W_Q_v = nn.Linear(input_dim, self.hidden_dim)
        self.W_K_v = nn.Linear(input_dim, self.hidden_dim)
        self.W_V_v = nn.Linear(input_dim, self.hidden_dim)
        self.W_O_v = nn.Linear(hidden_dim, input_dim)

        self.W_O_vt = nn.Linear(self.hidden_dim, input_dim)
        self.W_O_tv = nn.Linear(self.hidden_dim, input_dim)

    def multi_head_attention(self, Q, K, V):
        # 计算多头注意力
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

    def forward(self, text_features, visual_features):
        # text_features: (batch_size, n, input_dim) -> 文本特征
        # visual_features: (batch_size, n, input_dim) -> 视觉特征

        # 第一步：多头自注意力增强文本特征
        Q_t = self.W_Q_t(text_features)  # (batch_size, n, hidden_dim)
        K_t = self.W_K_t(text_features)  # (batch_size, n, hidden_dim)
        V_t = self.W_V_t(text_features)  # (batch_size, n, hidden_dim)
        Z_t = self.multi_head_attention(Q_t, K_t, V_t)  # (batch_size, n, hidden_dim)

        # 第二步：多头自注意力增强视觉特征
        Q_v = self.W_Q_v(visual_features)  # (batch_size, n, hidden_dim)
        K_v = self.W_K_v(visual_features)  # (batch_size, n, hidden_dim)
        V_v = self.W_V_v(visual_features)  # (batch_size, n, hidden_dim)
        Z_v = self.multi_head_attention(Q_v, K_v, V_v)  # (batch_size, n, hidden_dim)

        # 第三步：文本-视觉协同注意力
        Q_vt = self.W_Q_t(Z_v)   # Visual feature as query
        K_t =  self.W_K_t(Z_t)  # Text feature as key
        V_t =  self.W_V_t(Z_t)  # Text feature as value
        Z_vt = self.multi_head_attention(Q_vt, K_t, V_t)  # (batch_size, n, hidden_dim)
        Z_vt = self.W_O_vt(Z_vt)  # (batch_size, n, input_dim)

        # 第四步：视觉-文本协同注意力
        Q_tv = self.W_Q_v(Z_t)   # Text feature as query
        K_v =  self.W_K_v(Z_v)   # Visual feature as key
        V_v =  self.W_V_v(Z_v)  # Visual feature as value
        Z_tv = self.multi_head_attention(Q_tv, K_v, V_v)  # (batch_size, n, hidden_dim)
        Z_tv = self.W_O_tv(Z_tv)  # (batch_size, n, input_dim)

        return Z_vt, Z_tv

# # 示例输入
# batch_size = 4
# n = 10  # 假设节点的数量为 10
# input_dim = 256  # 输入特征的维度
# hidden_dim = 256*4  # 隐藏层的维度
# num_heads = 4
#
# # 随机生成模拟的输入数据
# text_features = torch.randn(batch_size, n, input_dim)  # 文本特征
# visual_features = torch.randn(batch_size, n, input_dim)  # 视觉特征
#
# # 创建Cross-modal Co-attention模型
# model = CrossModalCoAttention(input_dim, hidden_dim, num_heads)
#
# # 前向传播
# Z_vt, Z_tv = model(text_features, visual_features)
#
# # 输出的增强特征
# print(Z_vt.shape)  # 增强后的文本特征，形状为 (batch_size, n, input_dim)
# print(Z_tv.shape)  # 增强后的视觉特征，形状为 (batch_size, n, input_dim)
