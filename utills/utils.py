import random
import torch
import dgl
import dgl.function as fn
import numpy as np
import torch.nn as nn
import pickle
seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


def compute_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


def split_train_test(data, ratio):
    n_samples = data.shape[0]
    n_test = int(n_samples * ratio)
    ridx = np.random.choice(n_samples, n_test, replace=False)

    train = np.delete(data, ridx, axis=0)

    n_train, p_train = np.unique(train[:, 0]), np.unique(train[:, 1])

    f_test_idx = []
    for i in ridx:
        line = data[i]
        if line[0] in n_train and line[1] in p_train:
            f_test_idx.append(i)
    f_test = data[f_test_idx]
    f_train = np.delete(data, f_test_idx, axis=0)

    return f_train, f_test


# load dataset
def load_dataset(data_path):
    data = open(data_path, 'r')
    links = data.readlines()
    l = []
    for link in links:
        link = link.strip()
        link = link.split(' ')
        link = [eval(ll) for ll in link]
        l.append(link)
    links = np.array(l)
    return links


def get_test_data(links, test, negative_graph):
    t_link_src = test[:, 1]
    t_link_dst = test[:, 0]
    t_hetero_graph = dgl.heterograph({
        ('protein', 'link', 'rna'): (t_link_src, t_link_dst),
        ('rna', 'link_by', 'protein'): (t_link_dst, t_link_src),
    })
    t_negative_graph = construct_negative_graph(t_hetero_graph, 2, ('protein', 'link', 'rna'))
    src_dst1 = t_negative_graph.edges(etype='link')

    protein_id1 = src_dst1[0]
    rna_id1 = src_dst1[1]
    # print(protein_id)
    # combined_tensor = torch.stack([tensor1, tensor2], dim=0)
    neg = torch.stack([protein_id1, rna_id1], dim=1).numpy()
    neg1 = torch.stack([rna_id1, protein_id1], dim=1).numpy()
    ll = []
    for index, i in enumerate(neg1):
        dd = (i == links)
        d = np.sum(dd, axis=1)
        if 2 in d:
            ll.append(index)

    neg = np.delete(neg, ll, axis=0)
    # negative_graph = construct_negative_graph(hetero_graph, k, ('protein', 'link', 'rna'))
    test_samples = dict()

    src_dst_neg = negative_graph.edges(etype='link')
    # print(src_dst_neg)
    protein_id_neg = src_dst_neg[0]
    rna_id_neg = src_dst_neg[1]

    neg11 = torch.stack([protein_id_neg, rna_id_neg], dim=1).numpy()
    lll = []
    for index, i in enumerate(neg):
        dd = (i == neg11)
        d = np.sum(dd, axis=1)
        if 2 in d:
            lll.append(index)
    # print(len(lll))
    neg_test = np.delete(neg, lll, axis=0)
    test_samples['pos_samples'] = np.concatenate((np.array(test[:, 1])[:, None], np.array(test[:, 0])[:, None]), axis=1)
    test_samples['neg_samples'] = neg_test
    return test_samples




def parse_minibatch_LastFM(adjlists_pr, edge_metapath_indices_list_pr, pro_rna_links, device, samples=None,
                           offset=None,num_nodes=None):
    g_lists = [[], []]
    result_indices_lists = [[], []]

    for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(adjlists_pr, edge_metapath_indices_list_pr)):
        for adjlist, indices in zip(adjlists, edge_metapath_indices_list):

            edges, result_indices = parse_adjlist_LastFM(
                adjlist, indices, samples,
                pro_rna_links, offset, mode)
            # adjlist_LastFM(
            #     [adjlist[row[mode]] for row in pro_rna_links], [indices[row[mode]] for row in pro_rna_links], samples,
            #     pro_rna_links, offset, mode)
            if mode == 0:
                g = dgl.graph([],num_nodes=num_nodes[0])
            else:
                g = dgl.graph([], num_nodes=num_nodes[1])
            if len(edges) > 0:
                sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
                g.add_edges(*list(zip(*[(edges[i][0], edges[i][1]) for i in sorted_index])))
                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
                print(mode)
                print(g)
            else:
                result_indices = torch.LongTensor(result_indices).to(device)
            # print('start')
            # print(g)
            # print(result_indices.shape)
            # print('end')
            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)


    return g_lists, result_indices_lists



def load_LastFM_data(prefix='E:\graphs\MyProject\dataset\\NPR2241_processed'):

    in_file = open(prefix + '/0/0-1-0-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()



    in_file = open(prefix + '/1/1-0-1-0-1.adjlist', 'r')
    adjlist12 = [line.strip() for line in in_file]
    adjlist12 = adjlist12
    in_file.close()

    in_file = open(prefix + '/0/0-1-0-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()



    in_file = open(prefix + '/1/1-0-1-0-1_idx.pickle', 'rb')
    idx12 = pickle.load(in_file)
    in_file.close()



    return [[ adjlist01],[ adjlist12]],[[idx01], [idx12]]

if __name__ == '__main__':
    f = open('20_r_p.txt', 'r')
    l = []
    data = f.readlines()
    for d in data:
        d = d.strip()
        d = d.split(' ')
        d = [eval(dd) for dd in d]
        l.append(d)
    print(len(l))
    l = np.array(l)
    n = l[:, 0]
    p = l[:, 1]
    train, test = split_train_test(np.array(l), 0.8)
    print(train)
    print(test.shape)
