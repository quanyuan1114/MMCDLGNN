import dgl
import torch
from torch import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax


class MetapathAgg(nn.Module):
    def __init__(self,
                 out_dim,
                 num_heads,
                 attn_drop=0.5,
                 alpha=0.01,
                 ):
        super(MetapathAgg, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        # self.rnn = nn.Linear(out_dim // 2, num_heads * out_dim)
        self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        # Xavier均匀分布初始化
        nn.init.xavier_uniform_(self.attn, gain=1.387)  #1.387 是通过数学推导和实验优化后，为了使得 LeakyReLU 激活函数在初始化时具有较好的性能，确保每层的梯度能够有效传播，特别是在深度网络中
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        self.attn_drop = nn.Dropout(attn_drop)
        # self.liner = nn.Linear(out_dim//2,out_dim)
        # self.norml = nn.BatchNorm2d(out_dim)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def forward(self, g, features, edge_metapath_indices):
        # 获取边的嵌入特征
        edata = F.embedding(edge_metapath_indices, features)
        # 源节点特征 中间节点特征 目标节点特征
        src = edata[:, 0, :]
        mid = edata[:, 1, :]
        dst = edata[:, 2, :]
        # 计算源节点和目标节点/中间节点和目标节点，之间的余弦相似度
        d1 = cosine_similarity(src, dst)
        d2 = cosine_similarity(mid, dst)
        d = F.softmax(torch.stack([d1, d2], dim=1), dim=1)
        # 按相似度加权源/中间节点特征
        src = src * d[:, 0].unsqueeze(-1)
        mid = mid * d[:, 1].unsqueeze(-1)
        # edata = torch.cat([src.unsqueeze(1),mid.unsqueeze(1),dst.unsqueeze(1)],dim=1)
        # 组合特征
        edata = torch.cat([src.unsqueeze(1), mid.unsqueeze(1)], dim=1)
        # edata = torch.cat([src.unsqueeze(1)], dim=1)
        # edata = edata[:, 0, :].unsqueeze(-1)
        # edata = self.liner(edata)
        # edata = self.norml(edata)
        # edata = self.leaky_relu(edata)
        # edata = self.norml(edata)
        # 平均并通过线性层转换 添加额外的维度
        hidden = self.rnn(torch.mean(edata, dim=1))
        hidden = hidden.unsqueeze(dim=0)
        # 重新排列形状以适应注意力机制
        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)
        # 计算注意力得分
        a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)
        a = self.leaky_relu(a)
        # 更新图的边数据
        g.edata.update({'eft': eft, 'a': a})
        self.edge_softmax(g)
        # 执行消息传递
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        # 获取节点的聚合特征
        rst = g.ndata['ft']

        return rst


if __name__ == '__main__':
    net = MetapathAgg(8, 4)
    edge_metapath_indices = torch.tensor([[0, 1, 0],
                                          [3, 1, 0],
                                          [0, 1, 1],
                                          [1, 1, 1],
                                          [2, 1, 1],
                                          [2, 1, 2],
                                          [3, 1, 2],
                                          [0, 1, 3],
                                          [1, 1, 4],
                                          [2, 1, 4]])

    u = edge_metapath_indices[:, 0]
    v = edge_metapath_indices[:, -1]
    graph = dgl.graph((u, v))
    feat = torch.randn(5, 8)
    print(feat)
    logits = net(graph, feat, edge_metapath_indices)

    print(logits)
