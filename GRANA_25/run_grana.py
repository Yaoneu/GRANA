import time
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree
from utlis import *
from GRANA import *
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def embedding_look_up(data, v, v_attr, attr, v_partner, v_stru, node_dict,
                      node_size1, negative_ratio, sampling_table1,
                      sampling_table2, device, attributed):
    # generate samples in each batch
    data = data.tolist()
    table_size = 1e8

    intra_emb1 = []
    intra_emb2 = []
    intra_sign = []

    cross_edge_emb1 = []
    cross_edge_emb2 = []
    cross_edge_sign = []

    emb = []
    emb_neighbor = []

    emb_attr = []
    attr_vec = []

    stru_n1 = []
    stru_n2 = []
    cross_edge_n1 = []
    cross_edge_n2 = []
    align_n = []
    attr_n = []

    for i in range(len(data)):
        node = data[i][0]
        sn_label = data[i][1]
        neighbor = node_dict[sn_label][node]['neighbor']
        if neighbor != []:
            if sn_label == 0:
                n1 = [node] * (len(neighbor) * (negative_ratio + 1))
                stru_n1 = stru_n1 + n1
                neg = random.sample(sampling_table1, len(neighbor) * negative_ratio)
                n2 = neighbor + neg
                stru_n2 = stru_n2 + n2

            else:
                n1 = [node + node_size1] * (len(neighbor) * (negative_ratio + 1))
                stru_n1 = stru_n1 + n1
                neg = random.sample(sampling_table2, len(neighbor) * negative_ratio)
                n2 = [(x +node_size1) for x in (neighbor + neg)]
                stru_n2 = stru_n2 + n2
            for j in range(len(n2)):
                if j < len(neighbor):
                    intra_sign.append(1)
                else:
                    intra_sign.append(-1)

        cross_neighbor = node_dict[sn_label][node]['cross_edge']
        if cross_neighbor != []:
            neighbor = cross_neighbor
            if sn_label == 0:
                n1 = [node] * (len(neighbor) * (negative_ratio + 1))
                cross_edge_n1 = cross_edge_n1 + n1
                neg = random.sample(sampling_table2, negative_ratio * len(neighbor))

                n2 = [(x + node_size1) for x in (neighbor + neg)]
                cross_edge_n2 = cross_edge_n2 + n2
            else:
                n1 = [node + node_size1] * (len(neighbor) * (negative_ratio + 1))
                cross_edge_n1 = cross_edge_n1 + n1
                neg = random.sample(sampling_table1, negative_ratio * len(neighbor))
                n2 = neighbor + neg
                cross_edge_n2 = cross_edge_n2 + n2
            for j in range(len(n2)):
                if j < len(neighbor):
                    cross_edge_sign.append(1)
                else:
                    cross_edge_sign.append(-1)

        alignments = node_dict[sn_label][node]['alignment']
        if alignments == True:
            if sn_label == 0:
                align_n.append(node)
            else:
                align_n.append(node + node_size1)

        if attributed:
            if sn_label == 0:
                attr_n.append(node)
            else:
                attr_n.append(node + node_size1)

    if len(stru_n1) > 0:
        stru_n1 = torch.tensor(stru_n1).to(device)
        stru_n2 = torch.tensor(stru_n2).to(device)
        intra_emb1 = v_stru[stru_n1]
        intra_emb2 = v_stru[stru_n2]
        intra_sign = torch.tensor(intra_sign).to(device)

    if len(cross_edge_n1) > 0:
        cross_edge_n1 = torch.tensor(cross_edge_n1).to(device)
        cross_edge_n2 = torch.tensor(cross_edge_n2).to(device)
        cross_edge_emb1 = v[cross_edge_n1]
        cross_edge_emb2 = v_stru[cross_edge_n2]
        cross_edge_sign = torch.tensor(cross_edge_sign).to(device)

    if len(align_n) > 0:
        align_n = torch.tensor(align_n).to(device)
        emb = v[align_n]
        emb_neighbor = v_partner[align_n]

    if len(attr_n) > 0:
        emb_attr = v_attr[attr_n]
        attr_vec = attr[attr_n]

    return intra_emb1, intra_emb2, intra_sign, cross_edge_emb1, cross_edge_emb2,\
           cross_edge_sign, emb, emb_neighbor, emb_attr, attr_vec


def grana(node_size1, node_size2, weight, edge_index1, edge_index2, S1, S2, node_dict, test_list,
          attributed, attr_matrix1, attr_matrix2, sampling_table1, sampling_table2, rep_dim=100,
          batch_size=200, negative_ratio=10, epoch=500, directed=True):
    dataset = []
    for i in range(node_size1):
        dataset.append([i,0])
    for i in range(node_size2):
        dataset.append([i,1])

    device = 'cpu'
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        device = 'cuda:%d' % 0

    best_pre = []
    max_pre_30, max_epoch = 0, 0
    sn1_emb = []
    sn2_emb = []
    max_MAP, max_AUC = 0, 0
    t = time.time()

    # model initializing
    MTL_weight = torch.FloatTensor(weight).to(device)
    model = GRANA(attr_matrix1.shape[1], attr_matrix2.shape[1],
                  rep_dim, attribute_dim=attr_matrix1.shape[1], MTL_weight=MTL_weight,
                  sn_model_type='GCN', attributed=attributed).to(device)

    edge_index1 = torch.tensor(edge_index1).T.to(device)
    edge_index2 = torch.tensor(edge_index2).T.to(device)

    S1 = torch.from_numpy(S1).to(device)
    S2 = torch.from_numpy(S2).to(device)

    attr_matrix1 = torch.from_numpy(attr_matrix1).to(device)
    attr_matrix2 = torch.from_numpy(attr_matrix2).to(device)

    dataset = torch.tensor(dataset).to(device).type(torch.long)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    paras = []
    loss_weight = []
    for name, p in model.named_parameters():
        if name == 'MTL_weight':
            loss_weight.append(p)
            continue
        paras.append(p)
    optimizer1 = torch.optim.Adam(paras, lr=0.001, weight_decay=1e-5)
    optimizer2 = torch.optim.Adam(loss_weight, lr=0.001, weight_decay=1e-5)

    for e in range(epoch+1):
        epoch_time=time.time()
        model.train()
        loss_sum = torch.FloatTensor([0, 0, 0, 0]).to(device)
        for i, data in enumerate(data_loader):
            v1, v2, v_attr, v_stru, v_partner = model(attr_matrix1, edge_index1, attr_matrix2,
                                                      edge_index2, S1, S2)

            v = torch.cat((v1, v2), dim=0)
            if attributed:
                attr = torch.cat((attr_matrix1, attr_matrix2), dim=0)
            else:
                v_attr = []
                attr = []

            intra_emb1, intra_emb2, intra_sign, cross_edge_emb1, cross_edge_emb2, \
            cross_edge_sign, emb, emb_neighbor, emb_attr, attr_vec = embedding_look_up(
                data, v, v_attr, attr, v_partner, v_stru, node_dict,
                node_size1, negative_ratio, sampling_table1,
                sampling_table2, device, attributed)

            structure_loss = model.stru_loss(intra_emb1, intra_emb2, intra_sign)
            cross_loss = model.stru_loss(cross_edge_emb1, cross_edge_emb2, cross_edge_sign)
            embedding_loss = model.emb_loss(emb, emb_neighbor)
            attribute_loss = model.emb_loss(emb_attr, attr_vec)

            loss_stack = torch.stack([structure_loss, cross_loss, embedding_loss, attribute_loss])
            loss_sum = loss_sum + loss_stack.data

            # backward pass
            optimizer1.zero_grad()
            loss = model.loss(loss_stack, state='in')

            loss.backward()
            optimizer1.step()
        optimizer2.zero_grad()
        loss2 = model.loss(loss_sum, state='out')
        loss2.backward()
        optimizer2.step()

        if e%50 == 0:
            print(model.MTL_weight.data)
            print('epoch:{}, epoch time:{}'.format(e, time.time() - epoch_time))
            if e > 0:
                v1, v2, pre, MAP, AUC = test_kdtree(model, test_list,
                                                    attr_matrix1, edge_index1,
                                                    attr_matrix2, edge_index2,
                                                    S1, S2)
                print("kdtree: Epoch:{}, Test_Hits:{}, Time:{}, MAP:{}, AUC:{}".format(
                    e, pre, time.time() - t, MAP, AUC))
                if sum(pre) >= max_pre_30:
                    best_pre = pre
                    max_epoch = e
                    max_pre_30 = sum(pre)
                    max_MAP = MAP
                    max_AUC = AUC
                    sn1_emb = v1
                    sn2_emb = v2
    print('best hits:', best_pre, 'max epoch=', max_epoch, 'max MAP =', max_MAP, 'max AUC =', max_AUC)
    return sn1_emb, sn2_emb


def test_kdtree(model, test_list, attr_matrix1, edge_index1,
                attr_matrix2, edge_index2, S1, S2, k=30):
    model.eval()
    with torch.no_grad():
        v1, v2, _, _, _ = model(attr_matrix1, edge_index1, attr_matrix2,
                                edge_index2, S1, S2)

        v1 = F.normalize(v1, p=2, dim=1, eps=1e-12, out=None)
        v2 = F.normalize(v2, p=2, dim=1, eps=1e-12, out=None)

        an1 = [x[0] for x in test_list]
        an2 = [x[1] for x in test_list]
        emb1 = v1[an1].cpu().detach().numpy()
        emb2 = v2[an2].cpu().detach().numpy()
        v1 = v1.cpu().detach().numpy()
        v2 = v2.cpu().detach().numpy()

        align_matrix = cosine_similarity(v1, v2)
        MAP, AUC = compute_MAP_AUC(align_matrix, test_list)

        # MAP, _, AUC = compute_MAP_Hit_AUC(v1, v2, test_list)

        kd_tree1 = KDTree(v2, metric="euclidean")
        kd_tree2 = KDTree(v1, metric="euclidean")

        ind1 = kd_tree1.query(emb1, k=k, return_distance=False)
        ind2 = kd_tree2.query(emb2, k=k, return_distance=False)

        pre = [0]*k
        for i in range(len(test_list)):
            # sn1 node candidates
            candidates = ind1[i]
            for j in range(len(candidates)):
                if candidates[j] == an2[i] and j <= k:
                    for p in range(j, k):
                        pre[p] = pre[p] + 1
            # sn2 node candidates
            candidates = ind2[i]
            for j in range(len(candidates)):
                if candidates[j] == an1[i] and j <= k:
                    for p in range(j, k):
                        pre[p] = pre[p] + 1
    pre = np.array(pre) / (2 * len(test_list))
    return v1, v2, pre, MAP, AUC


def compute_MAP_AUC(alignment_matrix, test_list):
    gt = test_to_dict(test_list)
    MAP = 0
    AUC = 0
    for key, value in gt.items():
        ele_key = alignment_matrix[key].argsort()[::-1]
        for i in range(len(ele_key)):
            if ele_key[i] == value:
                ra = i + 1 # r1
                MAP += 1/ra
                AUC += (alignment_matrix.shape[1] - ra) / (alignment_matrix.shape[1] - 1)
                break

        ele_key = alignment_matrix.T[value].argsort()[::-1]
        for i in range(len(ele_key)):
            if ele_key[i] == key:
                ra = i + 1  # r1
                MAP += 1 / ra
                AUC += (alignment_matrix.shape[0] - ra) / (alignment_matrix.shape[0] - 1)
                break
    n_nodes = len(gt) * 2
    MAP /= n_nodes
    AUC /= n_nodes
    return MAP, AUC
