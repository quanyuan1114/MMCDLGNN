from .co_attention import CoAttention
from .model_hetero import MPA
from utills.utils import *
from .engyge_constrained_attention import MutilAttention
import torch.nn.functional as F
from .crossNodeCoAttention import CrossModalCoAttention

def sim(h1, h2):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())


def contrastive_loss_cross_view(h1, h2):
    f = lambda x: torch.exp(x)
    cross_sim = f(sim(h1, h2))
    return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))

# 定义一个全连接网络模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # ReLU激活函数
            layers.append(nn.Dropout(0.5))  # Dropout 防止过拟合
            prev_dim = hidden_dim

        # 最后一层输出
        layers.append(nn.Linear(prev_dim, output_dim))

        # 将所有层组合成模型
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# # 定义模型输入、隐藏层维度和输出维度
# input_dim = 512  # Z_rp 和 Z_pr 拼接后的维度
# hidden_dims = [512, 256, 128]  # 多层隐藏层，每层维度逐渐减小
# output_dim = 1  # 输出是一个标量得分

# 初始化模型
# model = MLP(input_dim, hidden_dims, output_dim)


class Model(nn.Module):
    def __init__(self, feat_dims, in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout, device, meta_paths):
        super().__init__()
        self.mlp = MLP(1024, [512, 256, 128], 1)
        self.co_att = CoAttention(256, num_heads)
        self.trans_list = nn.ModuleList([nn.Linear(feat_dim, in_dim, bias=True) for feat_dim in feat_dims])
        self.map1 = MPA(meta_paths=meta_paths[0],
                        in_size=in_dim,
                        hidden_size=hidden_dim,
                        out_size=out_dim,
                        num_heads=[num_heads],
                        dropout=dropout
                        )
        self.map2 = MPA(meta_paths=meta_paths[1],
                        in_size=in_dim,
                        hidden_size=hidden_dim,
                        out_size=out_dim,
                        num_heads=[num_heads],
                        dropout=dropout
                        )

        self.act = nn.ELU()
        self.fc_list = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(len(feat_dims))])
        self.pred = HeteroDotProductPredictor()
        self.device = device
        if num_layers == 1:
            self.sq1 = MutilAttention(in_dim, out_dim, num_heads, dropout)
            self.sq2 = MutilAttention(in_dim, out_dim, num_heads, dropout)
        else:
            self.sq1 = nn.Sequential()
            self.sq2 = nn.Sequential()
            self.sq1.add_module(f'layer{1}', MutilAttention(in_dim, hidden_dim, num_heads, dropout))
            self.sq2.add_module(f'layer{1}', MutilAttention(in_dim, hidden_dim, num_heads, dropout))
            for i in range(1, num_layers):
                if i + 1 == num_layers:
                    self.sq1.add_module(f'layer{i + 1}', MutilAttention(hidden_dim, out_dim, num_heads, dropout))
                    self.sq2.add_module(f'layer{i + 1}', MutilAttention(hidden_dim, out_dim, num_heads, dropout))
                else:
                    self.sq1.add_module(f'layer{i + 1}', MutilAttention(hidden_dim, hidden_dim, num_heads, dropout))
                    self.sq2.add_module(f'layer{i + 1}', MutilAttention(hidden_dim, hidden_dim, num_heads, dropout))

    def forward(self, g_list, pos_g, neg_g, x, sequence_feature,etype, edge_idx_list):
        pos_g = pos_g.to(self.device)
        neg_g = neg_g.to(self.device)
        x1 = x['protein'].to(self.device)
        x2 = x['rna'].to(self.device)
        g_p = g_list[0]
        g_r = g_list[1]
        g_p1 = []
        g_r1 = []
        for g in g_p:
            g = g.to(self.device)
            g_p1.append(g)
        for g in g_r:
            g = g.to(self.device)
            g_r1.append(g)
        neg_g = neg_g.to(self.device)

        # nodes transform
        x1 = self.trans_list[0](x1)
        x1 = self.act(x1)
        x2 = self.trans_list[1](x2)
        x2 = self.act(x2)

        # mpa
        edge_idx0 = edge_idx_list[0].to(self.device)
        edge_idx1 = edge_idx_list[1].to(self.device)
        h1 = self.map1(g_p1, x1, edge_idx0)
        h2 = self.map2(g_r1, x2, edge_idx1)

        # add attention to get dominant
        seq1 = sequence_feature['protein'].to(self.device)
        seq2 = sequence_feature['rna'].to(self.device)
        trans_list = nn.ModuleList([nn.Linear(feat_dim, 256, bias=True) for feat_dim in [49, 64]]).to(self.device)
        seq1 = trans_list[0](seq1)
        seq2 = trans_list[1](seq2)
        a1 = self.sq1(seq1)
        a2 = self.sq2(seq2)

        # add cmc loss to get co-dominant
        cmc_loss_protein = 0.5 * contrastive_loss_cross_view(a1, h1) + 0.5 * contrastive_loss_cross_view(h1,a1)
        cmc_loss_protein = cmc_loss_protein.mean()
        cmc_loss_rna = 0.5 * contrastive_loss_cross_view(a2, h2) + 0.5 * contrastive_loss_cross_view(h2,a2)
        cmc_loss_rna = cmc_loss_rna.mean()

        # get the co-dominant
        emb1 = torch.cat((h1, a1), dim=1)
        emb2 = torch.cat((h2, a2), dim=1)
        # linear projection
        emb_list = nn.Linear(512, 256, bias=True).to(self.device)
        emb1 = emb_list(emb1)
        emb2 = emb_list(emb2)

        # get the predict node pairs
        src_dst_pos = pos_g.edges(etype='link')
        src_dst_neg = neg_g.edges(etype='link')
        # construct representations of the predict node pairs"
        edges_pos = torch.stack((src_dst_pos[0], src_dst_pos[1]), dim=1)
        edges_neg = torch.stack((src_dst_neg[0], src_dst_neg[1]), dim=1)
        # 正样本预测
        protein_emb = [] #容器清空
        rna_emb = []
        src_seq_cat = []
        dst_seq_cat = []
        for edge in edges_pos:
            src_pos = emb1[edge[0]]
            dst_pos = emb2[edge[1]]
            src_seq = seq1[edge[0]]
            dst_seq = seq2[edge[1]]
            protein_emb.append(src_pos)
            rna_emb.append(dst_pos)
            src_seq_cat.append(src_seq)
            dst_seq_cat.append(dst_seq)
            # pos_score.append(torch.dot(Z_rp, Z_pr))
        protein_emb_pos = torch.stack(protein_emb)  # 将 protein_emb 堆叠成一个二维张量
        rna_emb_pos = torch.stack(rna_emb)  # 将 rna_emb 堆叠成一个二维张量
        #原始序列特征
        src_seq_cat = torch.stack(src_seq_cat)  # 将 rna_emb 堆叠成一个二维张量
        dst_seq_cat = torch.stack(dst_seq_cat)  # 将 rna_emb 堆叠成一个二维张量
        # 协同注意力
        Z_rp, Z_pr = self.co_att(protein_emb_pos, rna_emb_pos)
        # 残差链接
        pos_concat = torch.cat((Z_rp, Z_pr, src_seq_cat, dst_seq_cat), dim=1)
        # 使用mlp进行得分预测
        mlp = self.mlp  # 输出是 n*1 的得分
        pos_score = mlp(pos_concat)

        # 负样本预测
        protein_emb = [] #容器清空
        rna_emb = []
        src_seq_cat = []
        dst_seq_cat = []
        for edge in edges_neg:
            src_pos = emb1[edge[0]]
            dst_pos = emb2[edge[1]]
            src_seq = seq1[edge[0]]
            dst_seq = seq2[edge[1]]
            protein_emb.append(src_pos)
            rna_emb.append(dst_pos)
            src_seq_cat.append(src_seq)
            dst_seq_cat.append(dst_seq)
            # pos_score.append(torch.dot(Z_rp, Z_pr))
        protein_emb_neg = torch.stack(protein_emb)  # 将 protein_emb 堆叠成一个二维张量
        rna_emb_neg = torch.stack(rna_emb)  # 将 rna_emb 堆叠成一个二维张量

        src_seq_cat = torch.stack(src_seq_cat)  # 将 rna_emb 堆叠成一个二维张量
        dst_seq_cat = torch.stack(dst_seq_cat)  # 将 rna_emb 堆叠成一个二维张量

        Z_rp, Z_pr = self.co_att(protein_emb_neg, rna_emb_neg)
        neg_concat = torch.cat((Z_rp, Z_pr, src_seq_cat, dst_seq_cat), dim=1)
        # 使用mlp进行得分预测
        mlp = self.mlp  # 输出是 n*1 的得分
        neg_score = mlp(neg_concat)
        #
        # for edge in edges_neg:
        #     src_pos = emb1[edge[0]]
        #     dst_pos = emb2[edge[1]]
        #     Z_rp, Z_pr = self.cross_attn(src_pos, dst_pos)
            # neg_score.append(torch.dot(Z_rp, Z_pr))

        # pos_score = torch.tensor(pos_score)
        # neg_score = torch.tensor(neg_score)

            # 协同注意力
        # 输入：将共显性节点表示emb、预测节点对pos_g输入协同注意力模块计算
        # 流程：1、将预测节点对按索引取出，按照索引构造节点特征

        # Z_rp, Z_pr = self.cross_attn(pos_g,h)
        # Z_rp, Z_pr = cross_attn(neg_g,h)




        # residual connection x1 初始结构信息 h1 元路径聚合结构信息 a1 注意力计算序列特征 seq1 序列初始特征
        # emb1 = x1 + h1 + a1
        # emb2 = x2 + h2 + a2
        # emb1 = self.fc_list[0](emb1)
        # emb2 = self.fc_list[1](emb2)

        # h = {'protein': emb1, 'rna': emb2}

        # linear projection
        emb1 = self.fc_list[0](emb1)
        emb2 = self.fc_list[1](emb2)

        h = {'protein': emb1, 'rna': emb2}


        return cmc_loss_protein, cmc_loss_rna, pos_score, neg_score, h
