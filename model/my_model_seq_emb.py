from .model_hetero import MPA
from utills.utils import *
from .engyge_constrained_attention import MutilAttention


class Model(nn.Module):
    def __init__(self, feat_dims, in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout, device, meta_paths):
        super().__init__()
        # 对每种特征维度的线性变换
        self.trans_list_emb = nn.ModuleList([nn.Linear(feat_dim, in_dim, bias=True) for feat_dim in feat_dims])
        # 序列特征唯独转换
        self.trans_list_seq = nn.ModuleList([nn.Linear(feat_dim, in_dim, bias=True) for feat_dim in [49,64]])
        # 使用MPA（Meta - Path Aggregation）模块来处理不同的元路径
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
        # 对每种特征的线性投影层。
        self.fc_list = nn.ModuleList([nn.Linear(out_dim*3, out_dim) for _ in range(len(feat_dims))])
        #连接预测的模块，使用HeteroDotProductPredictor
        self.pred = HeteroDotProductPredictor()
        self.device = device
        # 加注意力
        self.sq1=MutilAttention(49, 128, num_heads=8, dropout=0.2)
        self.sq2=MutilAttention(64, 128, num_heads=8, dropout=0.2)

    def forward(self, g_list, pos_g, neg_g, x, etype, edge_idx_list,sequence_features):
        pos_g = pos_g.to(self.device)
        neg_g = neg_g.to(self.device)
        x1 = x['protein'].to(self.device)
        x2 = x['rna'].to(self.device)

        seq1 = sequence_features['protein'].to(self.device)
        seq2 = sequence_features['rna'].to(self.device)
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
        x1 = self.trans_list_emb[0](x1)
        x1 = self.act(x1)
        x2 = self.trans_list_emb[1](x2)
        x2 = self.act(x2)

        # ncRNA/protein nodes (sequence feature + embed feature) transform
        # torch.cat((s2, x2), dim=1)
        # s1 = self.trans_list[0](s1)
        # s1 = self.act(s1)
        # s2 = self.trans_list[1](s2)
        # s2 = self.act(s2)

        # mpa 元路径聚合出节点结构特征表示
        edge_idx0 = edge_idx_list[0].to(self.device)
        edge_idx1 = edge_idx_list[1].to(self.device)
        h1 = self.map1(g_p1, x1, edge_idx0)
        h2 = self.map2(g_r1, x2, edge_idx1)

        # add attention
        a1 = self.sq1(seq1)
        a2 = self.sq2(seq2)

        # ncRNA/protein nodes sequence feature transform
        seq1 = self.trans_list_seq[0](seq1)
        seq1 = self.act(seq1)
        seq2 = self.trans_list_seq[1](seq2)
        seq2 = self.act(seq2)

        # residual connection
        # emb1 = x1 + h1 +a1
        # emb2 = x2 + h2 +a2
        emb1 = torch.cat((seq1, h1,a1), dim=1)
        emb2 = torch.cat((seq2, h2,a2), dim=1)
        # emb1 = torch.cat((x1, h1), dim=1)
        # emb2 = torch.cat((x2, h2), dim=1)

        # linear projection
        emb1 = self.fc_list[0](emb1)
        emb2 = self.fc_list[1](emb2)

        h = {'protein': emb1, 'rna': emb2}

        return self.pred(pos_g, h, etype), self.pred(neg_g, h, etype), h
