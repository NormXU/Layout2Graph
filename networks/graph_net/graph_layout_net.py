#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:weishu
# datetime:2023/2/8 11:36 上午
# software: PyCharm
import copy
import math
import torch.nn.functional as F
import random
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from torchvision import ops
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import itertools

from .graph_net import FeatureFusionForFPN, GraphBase, _get_nearest_pair_custom, polar, to_bin, _get_nearest_pair_knn, \
    _get_nearest_pair_beta_skeleton


class GraphLayoutNet(nn.Module):
    '''
        对于每个node（文本框）预测一个分类类别，比如['Text', 'Title', 'Header', 'Footer', 'Figure', 'Table', 'List', ‘Seal’]
        node之间k最近邻/全连接，两两配对去预测关系，是否属于一个实例
    '''

    def __init__(self, **kwargs):
        super(GraphLayoutNet, self).__init__()
        self.num_classes = kwargs['num_classes']
        self.focal_loss_flag = kwargs['focal_loss_flag']
        self.max_pair_num = kwargs['max_pair_num']
        self.k_nearest_num = kwargs['k_nearest_num']
        self.sampling_strategy = kwargs['sampling_strategy']
        self.fc_flag = kwargs['fc_flag']
        self.gnn_res_flag = kwargs['gnn_res_flag']
        self.relation_flag = kwargs['relation_flag']
        self.rope_flag = kwargs['rope_flag']
        self.rope_max_length = kwargs['rope_max_length']
        self.polar_flag = kwargs['polar_flag']
        self.num_polar_bins = kwargs['num_polar_bins']
        self.polar_emb_feat = kwargs['polar_emb_feat']
        self.node_class_flag = kwargs['node_class_flag']
        if self.rope_flag:
            self.rope_emb_feat = kwargs['rope_emb_feat']
            pe = torch.zeros(self.rope_max_length, self.rope_emb_feat)
            position = torch.arange(0, self.rope_max_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.rope_emb_feat, 2).float() * (-math.log(10000.0) / self.rope_emb_feat))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.position_emb = pe
        else:
            self.rope_max_length = 100000000
        if kwargs.get('cnn_flag', True):
            if kwargs['cnn_backbone_type'].startswith('res'):
                self.backbone = resnet_fpn_backbone(kwargs['cnn_backbone_type'], pretrained=True)
            elif kwargs['cnn_backbone_type'] == 'fasterrcnn_resnet50_fpn_v2':
                self.backbone = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).backbone
            self.smooth_layer = FeatureFusionForFPN()
        else:
            self.backbone = None
        self.graph = GraphBase(**kwargs)
        gcn_out_feat = kwargs['gcn_out_feat']
        linear_cell_feat = gcn_out_feat * 2
        if self.relation_flag:
            linear_cell_feat += 18 + kwargs['cnn_emb_feat'] // 2
        if self.rope_flag:
            linear_cell_feat += kwargs['rope_emb_feat']
        if self.polar_flag:
            linear_cell_feat += kwargs['polar_emb_feat']
        if self.node_class_flag:
            linear_cell_feat += self.num_classes * 2
        self.linear_cell = nn.Sequential(nn.Linear(linear_cell_feat, gcn_out_feat), nn.LeakyReLU(inplace=True))
        self.cls_cell = nn.Sequential(nn.Linear(gcn_out_feat, 2), nn.LeakyReLU(inplace=True))
        # self.linear_cell = nn.Sequential(nn.Linear(linear_cell_feat, gcn_out_feat), nn.LayerNorm(gcn_out_feat),
        #                                  nn.LeakyReLU(inplace=True))
        # self.cls_cell = nn.Sequential(nn.Linear(gcn_out_feat, 2), nn.Dropout(0.2))
        if self.fc_flag:
            self.linear_node = nn.Sequential(nn.Linear(gcn_out_feat, gcn_out_feat), nn.LeakyReLU(inplace=True))
        self.cls_node = nn.Sequential(nn.Linear(gcn_out_feat, self.num_classes), nn.LeakyReLU(inplace=True))
        # self.cls_node = nn.Sequential(nn.Linear(gcn_out_feat, self.num_classes), nn.LayerNorm(self.num_classes))
        # self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, images, cell_boxes, targets=None, texts=None, linkings=None):
        if self.backbone:
            inter_feat = self.backbone(images)
            inter_feat = self.smooth_layer(images, inter_feat)
            feat, graphs, fusion_feat, cnn_decode_feat = self.graph(input=inter_feat,
                                                                    cell_boxes=copy.deepcopy(cell_boxes),
                                                                    targets=targets,
                                                                    texts=texts)
        else:
            feat, graphs, fusion_feat, cnn_decode_feat = self.graph(input=None,
                                                                    cell_boxes=copy.deepcopy(cell_boxes),
                                                                    targets=targets,
                                                                    texts=texts)
        # 正负样本采样
        pair_cell_target, pair_cell, pair_relation_feat_list, pair_rope_feat_list, pair_polar_feat_list = self.get_sampling(
            cell_boxes, targets, graphs, cnn_decode_feat, self.focal_loss_flag, images.device, linkings)

        if self.graph.graph_type == 'GCN':
            feat = torch.mean(torch.cat([i.unsqueeze(0) for i in feat], 0), 0)
        if self.gnn_res_flag == 'add':
            feat = feat + fusion_feat
        elif self.gnn_res_flag:
            # TODO
            feat = torch.maximum(feat, fusion_feat)
        # node 分类
        if self.fc_flag:
            node_feat = self.linear_node(feat)
            node_score_list = self.cls_node(node_feat)
        else:
            node_score_list = self.cls_node(feat)

        pair_cell_score_list = torch.empty(0).to(images.device)
        if len(pair_cell) > 0:
            # edge分类
            for i in range(0, pair_cell.shape[0] // self.max_pair_num + 1):
                pair_feat = torch.cat(
                    (torch.as_tensor(feat[pair_cell[i * self.max_pair_num:(i + 1) * self.max_pair_num, 0]]),
                     torch.as_tensor(feat[pair_cell[i * self.max_pair_num:(i + 1) * self.max_pair_num, 1]])),
                    dim=1)
                if self.relation_flag:
                    pair_feat = torch.cat(
                        [pair_feat, pair_relation_feat_list[i * self.max_pair_num:(i + 1) * self.max_pair_num]], dim=1)
                if self.rope_flag:
                    pair_feat = torch.cat(
                        [pair_feat, pair_rope_feat_list[i * self.max_pair_num:(i + 1) * self.max_pair_num]], dim=1)
                if self.polar_flag:
                    pair_feat = torch.cat(
                        [pair_feat, pair_polar_feat_list[i * self.max_pair_num:(i + 1) * self.max_pair_num]], dim=1)
                if self.node_class_flag:
                    node_class_feat = torch.cat(
                        (F.softmax(node_score_list[pair_cell[i * self.max_pair_num:(i + 1) * self.max_pair_num, 0]],
                                   dim=1),
                         F.softmax(node_score_list[pair_cell[i * self.max_pair_num:(i + 1) * self.max_pair_num, 1]],
                                   dim=1)),
                        dim=1)
                    pair_feat = torch.cat([pair_feat, node_class_feat], dim=1)
                linear_cell = self.linear_cell(pair_feat)
                cls_score_cell = self.cls_cell(linear_cell)
                pair_cell_score_list = torch.cat([pair_cell_score_list, cls_score_cell], dim=0)
        data = {
            'node_feat':
                feat,
            'pair_cell':
                pair_cell,
            'pair_cell_score_list':
                pair_cell_score_list,
            'pair_cell_pred':
                torch.argmax(pair_cell_score_list, dim=1)
                if len(pair_cell_score_list) > 0 else torch.empty(0).to(images.device),
            'pair_cell_target':
                pair_cell_target,
            'node_score_list':
                node_score_list,
            'node_pred':
                torch.argmax(node_score_list, dim=1),
        }
        return data

    def get_sampling(self, cell_boxes, targets, graphs, cnn_decode_feat, focal_loss_flag, device, linkings):
        #预测时k最近邻/全连接，两两配对去预测关系
        pair_cell_target = []
        if self.sampling_strategy:
            pair_cell_list = self._get_nearest_pair(graphs, cell_boxes)
        else:
            pair_cell_list = []
            for cell_box in cell_boxes:
                box_num = cell_box.shape[0]
                pair_cell_list.append(list(itertools.combinations(range(0, box_num), r=2)))
        if targets is not None:
            # 训练时正负样本均衡采样
            train_cell_pair_list = []
            for index, target in enumerate(targets):
                if linkings is not None:
                    positive_cell_pair = linkings[index]
                    positive_cell_pair = [tuple(cell_pair) for cell_pair in positive_cell_pair]
                else:
                    positive_cell_pair = []
                    for i, loc1 in enumerate(target):
                        for j, loc2 in enumerate(target[i + 1:]):
                            if loc1 == loc2:
                                positive_cell_pair.append((i, j + i + 1))
                negative_cell_pair = list(set(pair_cell_list[index]).difference(set(positive_cell_pair)))
                assert len(set(positive_cell_pair).difference(set(pair_cell_list[index]))) == 0, "!!Lack of pair Set:{}".format(set(positive_cell_pair).difference(set(pair_cell_list[index])))
                positive_cell_pair = list(set(pair_cell_list[index]).intersection(set(positive_cell_pair)))
                if focal_loss_flag is False:
                    random.shuffle(negative_cell_pair)
                    train_cell_pair_list.append(positive_cell_pair + negative_cell_pair[:len(positive_cell_pair)])
                    train_pair_cell_target = [1] * len(positive_cell_pair) + [0] * len(
                        negative_cell_pair[:len(positive_cell_pair)])
                else:
                    train_cell_pair_list.append(positive_cell_pair + negative_cell_pair)
                    train_pair_cell_target = [1] * len(positive_cell_pair) + [0] * len(negative_cell_pair)
                randnum = random.randint(0, 100)
                random.seed(randnum)
                random.shuffle(train_cell_pair_list[-1])
                random.seed(randnum)
                random.shuffle(train_pair_cell_target)
                pair_cell_target.extend(train_pair_cell_target)
            pair_cell_list = train_cell_pair_list
        pair_relation_feat_list, pair_rope_feat_list, pair_polar_feat_list = [], [], []
        if sum([len(pair_cell) for pair_cell in pair_cell_list]) != 0:
            if self.relation_flag:
                pair_relation_feat_list = self.get_relation_feature(pair_cell_list, cell_boxes, cnn_decode_feat)
            if self.rope_flag:
                pair_rope_feat_list = self.get_rope_feature(pair_cell_list).to(cnn_decode_feat.device)
            if self.polar_flag:
                pair_polar_feat_list = self.get_polar_feature(pair_cell_list, cell_boxes).to(cnn_decode_feat.device)
        pair_cell = []
        start_index = 0
        for index, edge_index_list in enumerate(pair_cell_list):
            edge_index_list = [(item[0] + start_index, item[1] + start_index) for item in edge_index_list]
            pair_cell.extend(edge_index_list)
            start_index += graphs[index].num_nodes

        return torch.from_numpy(np.array(pair_cell_target)).to(device), np.array(
            pair_cell), pair_relation_feat_list, pair_rope_feat_list, pair_polar_feat_list

    def get_relation_feature(self, pair_cell_list, cell_boxes, cnn_decode_feat):
        pair_relation_feat_list = []
        out_boxes = []
        for i, pair_cell in enumerate(pair_cell_list):
            if len(pair_cell) == 0:
                continue
            pair_cell = np.array(pair_cell)
            cell_box = cell_boxes[i].cpu().numpy()
            two_cell_box = np.concatenate([cell_box[pair_cell[:, 0]], cell_box[pair_cell[:, 1]]], axis=1)
            out_box = np.array([
                np.min(two_cell_box[:, [0, 2, 4, 6]], axis=1),
                np.min(two_cell_box[:, [1, 3, 5, 7]], axis=1),
                np.max(two_cell_box[:, [0, 2, 4, 6]], axis=1),
                np.max(two_cell_box[:, [1, 3, 5, 7]], axis=1)
            ]).T
            relative_feature = np.array([(two_cell_box[:, 0]-two_cell_box[:, 4])/(two_cell_box[:, 2]-two_cell_box[:, 0]),\
                                         (two_cell_box[:, 1] - two_cell_box[:, 5]) / (two_cell_box[:, 3] - two_cell_box[:, 1]), \
                                         (two_cell_box[:, 4] - two_cell_box[:, 0]) / (two_cell_box[:, 6] - two_cell_box[:, 4]), \
                                         (two_cell_box[:, 5] - two_cell_box[:, 1]) / (two_cell_box[:, 7] - two_cell_box[:, 5]), \
                                         np.log((two_cell_box[:, 2] - two_cell_box[:, 0]) / (two_cell_box[:, 6] - two_cell_box[:, 4])), \
                                         np.log((two_cell_box[:, 3] - two_cell_box[:, 1]) / (two_cell_box[:, 7] - two_cell_box[:, 5])), \

                                         (two_cell_box[:, 0] - out_box[:, 0]) / (two_cell_box[:, 2] - two_cell_box[:, 0]), \
                                         (two_cell_box[:, 1] - out_box[:, 1]) / (two_cell_box[:, 3] - two_cell_box[:, 1]), \
                                         (out_box[:, 0] - two_cell_box[:, 0]) / (out_box[:, 2] - out_box[:, 0]), \
                                         (out_box[:, 1] - two_cell_box[:, 1]) / (out_box[:, 3] - out_box[:, 1]), \
                                         np.log((two_cell_box[:, 2] - two_cell_box[:, 0]) / (out_box[:, 2] - out_box[:, 0])), \
                                         np.log((two_cell_box[:, 3] - two_cell_box[:, 1]) / (out_box[:, 3] - out_box[:, 1])), \

                                         (two_cell_box[:, 4] - out_box[:, 0]) / (two_cell_box[:, 6] - two_cell_box[:, 4]), \
                                         (two_cell_box[:, 5] - out_box[:, 1]) / (two_cell_box[:, 7] - two_cell_box[:, 5]), \
                                         (out_box[:, 0] - two_cell_box[:, 4]) / (out_box[:, 2] - out_box[:, 0]), \
                                         (out_box[:, 1] - two_cell_box[:, 5]) / (out_box[:, 3] - out_box[:, 1]), \
                                         np.log((two_cell_box[:, 6] - two_cell_box[:, 4]) / (out_box[:, 2] - out_box[:, 0])), \
                                         np.log((two_cell_box[:, 7] - two_cell_box[:, 5]) / (out_box[:, 3] - out_box[:, 1])), \
                                         ]).T
            pair_relation_feat_list.extend(relative_feature)
            out_boxes.append(torch.as_tensor(out_box).to(cnn_decode_feat.device))
        cnn_feat = ops.roi_align(cnn_decode_feat, out_boxes, 1)
        cnn_feat = cnn_feat.view(cnn_feat.size()[0], cnn_feat.size()[1])
        pair_relation_feat_list = torch.as_tensor(np.array(pair_relation_feat_list)).to(cnn_decode_feat.device)
        pair_relation_feat_list = torch.cat([pair_relation_feat_list, cnn_feat], dim=1)
        return pair_relation_feat_list

    def get_rope_feature(self, pair_cell_list):
        pair_rope_feat_list = torch.empty(0)
        for pair_cell in pair_cell_list:
            if len(pair_cell) == 0:
                continue
            pair_rope_feat_list = torch.cat(
                [pair_rope_feat_list, self.position_emb[np.array(pair_cell)[:, 1] - np.array(pair_cell)[:, 0]]], dim=0)
        return pair_rope_feat_list

    def get_polar_feature(self, pair_cell_list, cell_boxes):
        pair_polar_feat_list = torch.empty(0)
        for i, pair_cell in enumerate(pair_cell_list):
            if len(pair_cell) == 0:
                continue
            pair_cell = np.array(pair_cell)
            cell_box = cell_boxes[i].cpu().numpy()
            distances = []
            angles = []
            for pair in pair_cell:
                dist, angle = polar(cell_box[pair[0]], cell_box[pair[1]])
                distances.append(dist)
                angles.append(angle)
            # m = max(distances)
            polar_coordinates = to_bin(distances, angles, self.num_polar_bins)
            pair_polar_feat_list = torch.cat([pair_polar_feat_list, polar_coordinates], dim=0)
        return pair_polar_feat_list

    def _get_nearest_pair(self, graphs, cell_boxes):
        pair_cell_list = []
        for index in range(len(graphs)):
            # TODO GCN graph KNNGraph 跑不通
            if graphs[index].num_nodes == 1 and self.graph.graph_type == 'GCN':
                edge_index_list = []
            elif self.sampling_strategy == 'KNN':
                edge_index_list = _get_nearest_pair_knn(cell_boxes[index], self.k_nearest_num)
            elif self.sampling_strategy == 'BetaSkeleton':
                edge_index_list = _get_nearest_pair_beta_skeleton(cell_boxes[index])
            else:
                cell_boxes_array = cell_boxes[index].cpu().numpy()
                edge_index_list = _get_nearest_pair_custom(cell_boxes_array, self.rope_max_length)
            pair_cell_list.append(edge_index_list)
        return pair_cell_list
