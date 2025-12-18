
import time
from utlis import *
import run_grana
import math
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def get_embedding(graphfile1, graphfile2, anchorfile, attributefile,
                  training_ratio, directed1, directed2, d=100, attr_K=10, t=0.4,
                  epoch=500):
    t1 = time.time()
    nodedict1, edgedict1, edgedict_reverse1 = read_edge(graphfile1, directed1)
    nodedict2, edgedict2, edgedict_reverse2 = read_edge(graphfile2, directed2)
    anchor = read_anchor(nodedict1, nodedict2, anchorfile)
    attr_matrix1, attr_matrix2, attributed = read_attr(nodedict1, nodedict2, attributefile)
    node_size1 = len(nodedict1.keys())
    node_size2 = len(nodedict2.keys())
    print('SN1 node number:', node_size1, 'SN2 node number:', node_size2)
    print('negative sampling......')
    sampling_table1, sampling_table2 = gen_sampling_table(edgedict1, edgedict2, node_size1, node_size2)

    # generating alignment matrix S
    if attributed:
        sim_matrix = cosine_similarity(attr_matrix1, attr_matrix2)
        sim_matrix = np.where(sim_matrix >= t, sim_matrix, 0)
        identical_rate = len(np.argwhere(sim_matrix == 1)) / min(node_size1, node_size2)
    else:
        sim_matrix = []
        identical_rate = 1e8
    train_list, test_list = anchor_split(anchor, training_ratio)
    attr_matrix1, attr_matrix2, S1, S2 = alignment_sampling(
        attr_matrix1, attr_matrix2, attr_K, sim_matrix, train_list,
        node_size1, node_size2, attributed)


    # node preprocess
    node_dict, edge_index1, edge_index2, cross_edge_num, degree = node_preprocess(S1[0], S2[0], nodedict1,
                                                                                  edgedict1, edgedict_reverse1,
                                                                                  nodedict2, edgedict2,
                                                                                  edgedict_reverse2, train_list)
    # weight initialization
    weight = weight_init(degree / (node_size1 + node_size2), cross_edge_num / (len(train_list) * 2),
                         identical_rate, attributed)

    # model
    print('model: GRANA')
    sn1_emb, sn2_emb = run_grana.grana(node_size1, node_size2, weight, edge_index1, edge_index2,
                                       S1, S2, node_dict, test_list, attributed, attr_matrix1, attr_matrix2,
                                       sampling_table1, sampling_table2, rep_dim=d, epoch=epoch, directed=False)


    print("GRANA Running time is", time.time() - t1)
    return sn1_emb, sn2_emb


def node_preprocess(S1, S2, nodedict1, edgedict1, edgedict_reverse1, nodedict2, edgedict2, edgedict_reverse2, anchor_list):
    S1 = [int(x) for x in list(set(S1))]
    S2 = [int(x) for x in list(set(S2))]
    sn1node_dict = defaultdict(dict)
    sn2node_dict = defaultdict(dict)
    an1 = [x[0] for x in anchor_list]
    an2 = [x[1] for x in anchor_list]
    cross_edge_num = 0
    edge_index1 = []
    edge_index2 = []
    degree = 0
    for key in nodedict1.keys():
        i = nodedict1[key]
        sn1node_dict[i]['neighbor'] = []
        sn1node_dict[i]['cross_edge'] = []
        sn1node_dict[i]['alignment'] = []
        if i in edgedict1.keys():
            neighbor = edgedict1[i]
            sn1node_dict[i]['neighbor'] = neighbor
            degree += len(neighbor)
            for n in neighbor:
                edge_index1.append([i, n])
        partner_edge = []
        if i in an1:
            partner = an2[an1.index(i)]
            partner_neighbor = []
            if partner in edgedict2.keys():
                partner_neighbor = edgedict2.get(partner)
            if partner in edgedict_reverse2.keys():
                partner_neighbor = partner_neighbor + edgedict_reverse2.get(partner)

            if partner_neighbor != []:
                for n in partner_neighbor:
                    partner_edge.append(n)
                    if n in an2:
                        cross_edge_num += 1
        if i in S1:
            sn1node_dict[i]['alignment'] = True
        else:
            sn1node_dict[i]['alignment'] = False
        sn1node_dict[i]['cross_edge'] = partner_edge

    for key in nodedict2.keys():
        i = nodedict2[key]
        sn2node_dict[i]['neighbor'] = []
        sn2node_dict[i]['cross_edge'] = []
        sn2node_dict[i]['alignment'] = []
        if i in edgedict2.keys():
            neighbor = edgedict2[i]
            sn2node_dict[i]['neighbor'] = neighbor
            degree += len(neighbor)
            for n in neighbor:
                edge_index2.append([i, n])
        partner_edge = []
        if i in an2:
            partner = an1[an2.index(i)]
            partner_neighbor = []
            if partner in edgedict1.keys():
                partner_neighbor = edgedict1.get(partner)
            if partner in edgedict_reverse1.keys():
                partner_neighbor = partner_neighbor + edgedict_reverse1.get(partner)

            if partner_neighbor != []:
                for n in partner_neighbor:
                    partner_edge.append(n)
                    if n in an1:
                        cross_edge_num +=1
        if i in S2:
            sn2node_dict[i]['alignment'] = True
        else:
            sn2node_dict[i]['alignment'] = False
        sn2node_dict[i]['cross_edge'] = partner_edge

    node_dict={}
    node_dict[0]=sn1node_dict
    node_dict[1]=sn2node_dict

    return node_dict, edge_index1, edge_index2, cross_edge_num, degree


def alignment_sampling(attr_matrix1, attr_matrix2, attr_K, sim_matrix, anchor_list,
                       node_size1, node_size2, attributed):
    anchor_matrix1 = np.zeros([node_size1, len(anchor_list)], dtype=np.float32)
    anchor_matrix2 = np.zeros([node_size2, len(anchor_list)], dtype=np.float32)
    for i in range(len(anchor_list)):
        n1 = anchor_list[i][0]
        n2 = anchor_list[i][1]
        anchor_matrix1[n1][i] = 1
        anchor_matrix2[n2][i] = 1
    print(np.sum(cosine_similarity(anchor_matrix1, anchor_matrix2)))

    if attributed:
        attr_matrix1 = np.concatenate((attr_matrix1, anchor_matrix1), axis=1)
        attr_matrix2 = np.concatenate((attr_matrix2, anchor_matrix2), axis=1)
    else:
        attr_matrix1 = np.concatenate((np.eye(node_size1, dtype=np.float32), anchor_matrix1), axis=1)
        attr_matrix2 = np.concatenate((np.eye(node_size2, dtype=np.float32), anchor_matrix2), axis=1)

    S1 = []
    S2 = []

    for an in anchor_list:
        index1 = an[0]
        index2 = an[1]
        S1.append([index1, index2, 1])
        S2.append([index2, index1, 1])

    if attributed:
        sn_sim, sn_index = get_sorted_top_k(sim_matrix, top_k=attr_K, axis=1, reverse=True)
        anchor = [x[0] for x in anchor_list]
        for i in range(node_size1):
            if i not in anchor:
                pos = len(np.argwhere(sn_sim[i] > 0).tolist())
                if pos < len(sn_sim[i]):
                    partner = sn_index[i][0:pos]
                    sim = sn_sim[i][0:pos] / sum(sn_sim[i][0:pos])
                else:
                    partner = sn_index[i]
                    sim = sn_sim[i] / np.sum(sn_sim[i])
                for j in range(len(partner)):
                    S1.append([i, partner[j], sim[j]])

        sn_sim, sn_index = get_sorted_top_k(sim_matrix.T, top_k=attr_K, axis=1, reverse=True)
        anchor = [x[1] for x in anchor_list]
        for i in range(node_size2):
            if i not in anchor:
                pos = len(np.argwhere(sn_sim[i] > 0).tolist())
                if pos < len(sn_sim[i]):
                    partner = sn_index[i][0:pos]
                    sim = sn_sim[i][0:pos] / sum(sn_sim[i][0:pos])
                else:
                    partner = sn_index[i]
                    sim = sn_sim[i] / np.sum(sn_sim[i])
                for j in range(len(partner)):
                    S2.append([i, partner[j], sim[j]])
                    # S2[i][partner[j]] = sim[j]

    # The row of S1 is sn1 node and its column is sn2 node
    # The row of S2 is sn2 node and its column is sn1 node

    S_node = [x[0] for x in S1]
    if (node_size1 - 1) not in S_node:
        S1.append([node_size1 - 1, 0, 0])

    S_node = [x[0] for x in S2]
    if (node_size2 - 1) not in S_node:
        S2.append([node_size2 - 1, 0, 0])

    return attr_matrix1, attr_matrix2, np.array(S1).T, np.array(S2).T


def gen_sampling_table(edgedict1, edgedict2, numNodes1, numNodes2):
    # generate sampling table for negative sampling
    t = time.time()
    table_size = 1e8
    power = 0.75

    node_degree1 = np.zeros(numNodes1)  # out degree

    for key in edgedict1.keys():
        node_degree1[key] = len(edgedict1[key])

    norm1 = sum([math.pow(node_degree1[i], power) for i in range(numNodes1)])

    sampling_table1 = np.zeros(int(table_size), dtype=np.uint32)

    p = 0
    i = 0
    for j in range(numNodes1):
        p += float(math.pow(node_degree1[j], power)) / norm1
        while i < table_size and float(i) / table_size < p:
            sampling_table1[i] = j
            i += 1

    node_degree2 = np.zeros(numNodes2)
    for key in edgedict2.keys():
        node_degree2[key] = len(edgedict2[key])

    norm2 = sum([math.pow(node_degree2[i], power) for i in range(numNodes2)])

    sampling_table2 = np.zeros(int(table_size), dtype=np.uint32)

    p = 0
    i = 0
    for j in range(numNodes2):
        p += float(math.pow(node_degree2[j], power)) / norm2
        while i < table_size and float(i) / table_size < p:
            sampling_table2[i] = j
            i += 1

    print('negative sampling finished, time=',time.time()-t)
    return sampling_table1.tolist(), sampling_table2.tolist()


if __name__ == '__main__':
    graphfile1 = 'data/douban/douban1.txt'
    graphfile2 = 'data/douban/douban2.txt'
    anchorfile = 'data/douban/douban-anchor.txt'
    attributefile = ['data/douban/douban-attr1.txt',
                     'data/douban/douban-attr2.txt']

    d = 100  # dimension of node embeddings
    tr = 0.1  # training ratio
    K = 3  # number of nodes considered in cross alignment matrix S, which is used in cross convolution
    epoch = 300  # number of epochs

    print('training ratio=', tr, 'embedding dim=', d, 'N=', K)
    get_embedding(graphfile1, graphfile2, anchorfile, attributefile, tr,
                  d=d, epoch=epoch, attr_K=K, directed1=False, directed2=False)

