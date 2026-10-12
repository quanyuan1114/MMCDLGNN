import dgl
import torch

# 创建一个简单的 DGL 图，包含 3 个节点和 3 条边
src_nodes = torch.tensor([0, 1, 2])  # 边的起点
dst_nodes = torch.tensor([1, 2, 0])  # 边的终点
graph = dgl.graph((src_nodes, dst_nodes))

# 1. 提取边的起点和终点
src, dst = graph.edges()
print("Edges (source nodes):", src)
print("Edges (destination nodes):", dst)

# 2. 如果有边特征，可以通过 `edata` 提取
# 假设我们给每条边添加了一个权重
graph.edata['weight'] = torch.tensor([0.1, 0.2, 0.3])
print("Edge weights:", graph.edata['weight'])
