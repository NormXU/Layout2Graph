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

from base.driver import logger
from scripts.preprocess_data import get_iou


class DefaultPostProcessor():

    def __init__(self, **kwargs):
        self.delete_dif_cls = kwargs.get('delete_dif_cls', True)
        self.pair_score_threshold = kwargs.get('pair_score_threshold', 0.0)

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
            od_label = get_od_label(subgraph, cell_box, node_pred, node_score_list)
            od_label_list.append(od_label)

        result_data['od_label_list'] = od_label_list
        return result_data

def get_od_label(subgraph, cell_box, node_pred, node_score_list):
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
