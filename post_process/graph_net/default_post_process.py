#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:weishu
# datetime:2022/11/8 11:34 上午
# software: PyCharm
import copy
import math
import os
from collections import Counter

import numpy as np
import networkx
import torch


class DefaultPostProcessor():

    def __init__(self, **kwargs):
        self.delete_dif_cls = kwargs.get('delete_dif_cls', True)
        #'Text'0, 'Title'1, 'Header'2, 'Footer'3, 'Figure'4, 'Table'5, 'List'6
        self.label_priority_list = kwargs.get('label_priority_list', [1, 5, 6, 4, 0, 3, 2])
        self.pair_score_threshold = kwargs.get('pair_score_threshold', 0.0)
        self.combine_threshold = kwargs.get('combine_threshold', 1.0)
        self.delete_pair_flag = kwargs.get('delete_pair_flag', False)

    def __call__(self, result_data, cell_box, transform_cell_box=None, text=None, **kwargs):
        node_pred = result_data['node_pred']
        pair_cell = result_data['pair_cell']
        adj = np.zeros([len(cell_box), len(cell_box)], dtype='int')
        node_score_list = torch.nn.functional.softmax(result_data['node_score_list'], dim=1)
        if len(pair_cell) > 0:
            pair_cell_pred = list(map(int, result_data['pair_cell_pred'].cpu().numpy().tolist()))
            pair_cell_score_list = torch.nn.functional.softmax(result_data['pair_cell_score_list'], dim=1)
            all_cell_pairs_score = [
                round(float(pair_cell_score_list[i][1].cpu()), 2)
                for i, pair in enumerate(pair_cell)
                if pair_cell_pred[i] == 1
            ]
            all_cell_pairs = [
                pair for i, pair in enumerate(pair_cell)
                if pair_cell_pred[i] == 1 and pair_cell_score_list[i][1] > self.pair_score_threshold
            ]

            for pair in all_cell_pairs:
                if self.delete_dif_cls:
                    if node_pred[pair[0]] == node_pred[pair[1]]:
                        adj[pair[0], pair[1]], adj[pair[1], pair[0]] = 1, 1
                else:
                    adj[pair[0], pair[1]], adj[pair[1], pair[0]] = 1, 1
        nodenum = adj.shape[0]
        edge_temp = np.where(adj != 0)
        edge = list(zip(edge_temp[0], edge_temp[1]))
        layout_graph = networkx.Graph()
        layout_graph.add_nodes_from(list(range(nodenum)))
        layout_graph.add_edges_from(edge)

        od_label_list = []
        for c in networkx.connected_components(layout_graph):
            # 得到不连通的子集
            subgraph = layout_graph.subgraph(c)
            od_label = get_od_label(subgraph, cell_box, node_pred, node_score_list, self.label_priority_list)
            od_label_list.append(od_label)

        result_data['od_label_list'] = od_label_list
        return result_data


def delete_pair(od_label, cell_box, node_pred, node_score_list, label_priority_list):
    graph = networkx.Graph(od_label['subgraph'])
    pairs = graph.edges()
    pairs = [pair for pair in pairs]
    if len(pairs) == 1:
        graph.remove_edge(pairs[0][0], pairs[0][1])
    else:
        for pair in pairs:
            pair_x1 = min(cell_box[pair[0]][0], cell_box[pair[1]][0])
            pair_x2 = max(cell_box[pair[0]][2], cell_box[pair[1]][2])
            pair_y1 = min(cell_box[pair[0]][3], cell_box[pair[1]][3])
            pair_y2 = max(cell_box[pair[0]][1], cell_box[pair[1]][1])
            for node_index, box in enumerate(cell_box):
                if node_index not in pair:
                    union_x1 = max(pair_x1, box[0])
                    union_x2 = min(pair_x2, box[2])
                    union_y1 = max(pair_y1, box[1])
                    union_y2 = min(pair_y2, box[3])
                    if union_x2 - union_x1 > (box[2] - box[0]) * 0.2 and union_y2 - union_y1 > (box[3] - box[1]) * 0.2:
                        graph.remove_edge(pair[0], pair[1])
                        break
    new_od_label_list = []
    for c in networkx.connected_components(graph):
        # 得到不连通的子集
        nodeSet = graph.subgraph(c).nodes()
        subgraph = graph.subgraph(c)
        cell_box_list = []
        label_list = []
        one_node_score_list = []
        for node in nodeSet:
            box = cell_box[node]
            cell_box_list.append(box)
            label_list.append(int(node_pred[node].cpu()))
            one_node_score_list.append(float(node_score_list[node][node_pred[node]].cpu()))
        cell_box_array = np.array(cell_box_list)
        label_counters = Counter(label_list)
        # 如果有两个label的count数一样，按照预定义的优先级也决定最终label
        if len(label_counters) > 1 and label_counters.most_common(1)[0][1] == label_counters.most_common(2)[1][1]:
            two_label_list = [label_counters.most_common(1)[0][0], label_counters.most_common(2)[1][0]]
            two_label_priority = [label_priority_list.index(label) for label in two_label_list]
            label = two_label_list[two_label_priority.index(min(two_label_priority))]
        else:
            label = label_counters.most_common(1)[0][0]
        points = [
            float(min(cell_box_array[:, 0])),
            float(min(cell_box_array[:, 1])),
            float(max(cell_box_array[:, 2])),
            float(max(cell_box_array[:, 3]))
        ]
        new_od_label_list.append({
            'label': label,
            'points': points,
            'nodeSet': nodeSet,
            'subgraph': subgraph,
            "node_score": sum(one_node_score_list) / len(one_node_score_list),
        })

    return new_od_label_list


def get_od_label(subgraph, cell_box, node_pred, node_score_list, label_priority_list):
    nodeSet = subgraph.nodes()
    cell_box_list = []
    label_list = []
    one_node_score_list = []
    for node in nodeSet:
        box = cell_box[node]
        cell_box_list.append(box)
        label_list.append(int(node_pred[node].cpu()))
        one_node_score_list.append(float(node_score_list[node][node_pred[node]].cpu()))
    cell_box_array = np.array(cell_box_list)
    label_counters = Counter(label_list)
    # 如果有两个label的count数一样，按照预定义的优先级也决定最终label
    if len(label_counters) > 1 and label_counters.most_common(1)[0][1] == label_counters.most_common(2)[1][1]:
        two_label_list = [label_counters.most_common(1)[0][0], label_counters.most_common(2)[1][0]]
        two_label_priority = [label_priority_list.index(label) for label in two_label_list]
        label = two_label_list[two_label_priority.index(min(two_label_priority))]
    else:
        label = label_counters.most_common(1)[0][0]
    points = [
        float(min(cell_box_array[:, 0])),
        float(min(cell_box_array[:, 1])),
        float(max(cell_box_array[:, 2])),
        float(max(cell_box_array[:, 3]))
    ]
    od_label = {
        'label': label,
        'points': points,
        'nodeSet': nodeSet,
        'subgraph': subgraph,
        "node_score": sum(one_node_score_list) / len(one_node_score_list),
    }
    return od_label
