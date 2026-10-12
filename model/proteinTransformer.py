import torch
import torch.nn as nn
import torch.optim as optim


# 假设你输入的蛋白质序列已经是一个 n * 256 的张量
# n 是批次大小，256 是每个序列的特征维度
# 现在我们将直接使用这些输入

class ProteinTransformer(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, dim_ff, num_classes, dropout=0.1):
        super(ProteinTransformer, self).__init__()

        # Transformer 编码器
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,  # 输入的维度就是蛋白质序列的特征维度
            nhead=num_heads,  # 自注意力的头数
            dim_feedforward=dim_ff,  # 前馈网络的维度
            dropout=dropout  # dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

        # 分类层：将 Transformer 输出的最后一层映射到最终的类别
        self.fc = nn.Linear(input_dim, num_classes)  # 假设我们做的是分类任务

    def forward(self, x):
        # 输入 x 形状为 [batch_size, input_dim]
        # 添加序列维度，使其变为 [batch_size, 1, input_dim]
        x = x.unsqueeze(1)  # 形状变为 [batch_size, 1, input_dim]

        # 将形状从 [batch_size, 1, input_dim] 转换为 [1, batch_size, input_dim]
        x = x.permute(1, 0, 2)  # 变换后的形状为 [seq_length, batch_size, input_dim]

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)
        x = torch.max(x, dim=0)[0]  # [batch_size, input_dim]，max池化后，池化沿着 seq_length 维度

        return x


# 假设你已经有了一个 n * 256 的蛋白质序列信息
# 例如，我们随机生成一个 32 * 256 的张量作为输入（32 是批次大小）
input_tensor = torch.randn(32, 256)  # 32个样本，每个样本256维

# 创建模型
model = ProteinTransformer(
    input_dim=256,  # 每个序列的特征维度是256
    num_layers=4,  # Transformer 编码器的层数
    num_heads=8,  # 多头注意力的头数
    dim_ff=512,  # 前馈网络的维度
    num_classes=10,  # 假设是一个10类分类任务
    dropout=0.1  # dropout比率
)

# 假设目标标签是10类分类
labels = torch.randint(0, 10, (32,))  # 随机生成目标标签，大小为32（与批次大小相同）

# 进行前向传播
outputs = model(input_tensor)

# 输出分类结果
print(outputs.shape)  # 输出形状应该是 [batch_size, num_classes]，即 [32, 10]

# 如果需要，可以使用交叉熵损失进行训练
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
print(f"Loss: {loss.item()}")
