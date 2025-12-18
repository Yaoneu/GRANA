import numpy as np
import random
from collections import defaultdict


def get_sorted_top_k(array, top_k=1, axis=1, reverse=True):
    """
    多维数组排序
    Args:
        array: 多维数组
        top_k: 取数
        axis: 轴维度
        reverse: 是否倒序

    Returns:
        top_sorted_scores: 值
        top_sorted_indexes: 位置
    """
    if reverse:
        # argpartition分区排序，在给定轴上找到最小的值对应的idx，partition同理找对应的值
        # kth表示在前的较小值的个数，带来的问题是排序后的结果两个分区间是仍然是无序的
        # kth绝对值越小，分区排序效果越明显
        axis_length = array.shape[axis]
        partition_index = np.take(np.argpartition(array, kth=-top_k, axis=axis),
                                  range(axis_length - top_k, axis_length), axis)
    else:
        partition_index = np.take(np.argpartition(array, kth=top_k, axis=axis), range(0, top_k), axis)
    top_scores = np.take_along_axis(array, partition_index, axis)
    # 分区后重新排序
    sorted_index = np.argsort(top_scores, axis=axis)
    if reverse:
        sorted_index = np.flip(sorted_index, axis=axis)
    top_sorted_scores = np.take_along_axis(top_scores, sorted_index, axis)
    top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
    return top_sorted_scores, top_sorted_indexes


def anchor_split(anchor_list, train_ratio):
    train_list = random.sample(anchor_list, int(train_ratio * len(anchor_list)))
    test_list = []
    for an in anchor_list:
        if an not in train_list:
            test_list.append(an)

    return train_list, test_list


def weight_init(degree, cross_degree, identical_rate, attributed):
    print('degree:', degree, 'cross degree:', cross_degree, 'identical attribute rate:', identical_rate)
    weight = []
    if degree < 7:
        weight.append(0.1)
    else:
        weight.append(1)

    if cross_degree < 0.5:
        weight.append(0.1)
    else:
        weight.append(1)

    if identical_rate < 3:
        weight.append(1)
        weight.append(1)
    else:
        if attributed:
            weight.append(0.1)
            weight.append(0.1)
        else:
            weight.append(1)
            weight.append(0)
    print(weight)
    return weight


def read_edge(graphfile, directed):
    edgedict = defaultdict(list)
    edgedict_reverse = defaultdict(list)
    nodedict = {}
    i = 0
    f = open(graphfile, 'r', encoding='utf-8')
    while 1:
        l=f.readline().rstrip('\n')
        if l=='':
            break
        src, dst = l.split()
        if src not in nodedict.keys():
            nodedict[src] = i
            i = i + 1
        if dst not in nodedict.keys():
            nodedict[dst] = i
            i = i + 1
        if directed:
            edgedict[nodedict[src]].append(nodedict[dst])
            edgedict_reverse[nodedict[dst]].append(nodedict[src])
        else:
            edgedict[nodedict[src]].append(nodedict[dst])
            edgedict[nodedict[dst]].append(nodedict[src])
            edgedict_reverse[nodedict[src]].append(nodedict[dst])
            edgedict_reverse[nodedict[dst]].append(nodedict[src])
    return nodedict, edgedict, edgedict_reverse


def read_attr(nodedict1, nodedict2, attributefile):
    if attributefile != []:
        attributed = True
        attr1 = {}
        attr2 = {}
        f = open(attributefile[0], 'r', encoding='utf-8')
        while 1:
            l = f.readline().rstrip('\n')
            if l == '':
                break
            node, vec = l.split()
            vec = list(map(float, vec.split(',')))
            attr1[node] = vec
        f.close()

        f = open(attributefile[1], 'r', encoding='utf-8')
        while 1:
            l = f.readline().rstrip('\n')
            if l == '':
                break
            node, vec = l.split()
            vec = list(map(float, vec.split(',')))
            attr2[node] = vec
        f.close()

        attr_matrix1 = [[]] * len(nodedict1.keys())
        attr_matrix2 = [[]] * len(nodedict2.keys())

        for key in nodedict1.keys():
            i = nodedict1[key]
            attr_matrix1[i] = attr1[key]
        for key in nodedict2.keys():
            i = nodedict2[key]
            attr_matrix2[i] = attr2[key]
        I1 = np.array(attr_matrix1, dtype=np.float32)
        I2 = np.array(attr_matrix2, dtype=np.float32)
    else:
        I1 = []
        I2 = []
        attributed = False
    return I1, I2, attributed


def read_anchor(nodedict1, nodedict2, anchorfile):
    anchor = []
    f = open(anchorfile, 'r', encoding='utf-8')
    while 1:
        l = f.readline().rstrip('\n')
        if l == '':
            break
        src, dst = l.split()
        i = nodedict1[src]
        j = nodedict2[dst]
        anchor.append([i, j])
    return anchor


def test_to_dict(test_list):
    test_dict = {}
    for an in test_list:
        test_dict[an[0]] = an[1]

    return test_dict