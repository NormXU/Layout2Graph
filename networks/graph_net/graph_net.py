# encoding: utf-8
'''
@email: weishu@datagrand.com
@software: Pycharm
@time: 2021/11/23 2:09 下午
@desc:
'''
# encoding: utf-8
import copy
import math
from typing import Tuple
import torch.nn.functional as F
import random
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from torch_geometric import transforms
from torchvision import ops
import itertools

from .gcn.dgcnn import DGCNNModule
from .gcn.gravnet import GravnetModule
# from .gcn.garnet import GarnetModule
from torch_geometric.data import Data as GraphData


class GraphBase(nn.Module):

    def __init__(self,
                 height,
                 width,
                 norm_box_flag=False,
                 in_channels=1024,
                 cnn_emb_feat=512,
                 box_emb_feat=256,
                 gcn_out_feat=512,
                 vocab_size=None,
                 text_emd_feat=512,
                 text_hidden_dim=256,
                 bidirectional=False,
                 graph_type='GCN',
                 graph_layer_num=1,
                 **kwargs):
        super(GraphBase, self).__init__()
        self.height = height
        self.width = width
        self.norm_box_flag = norm_box_flag
        if kwargs.get('cnn_flag', True):
            self.decode_out = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(256,affine=False),
                nn.ReLU(inplace=True))
            self.cnn_emb = nn.Sequential(
                nn.Linear(256 * 2 * 2, cnn_emb_feat),
                # nn.BatchNorm1d(cnn_emb_feat,affine=False),
                nn.ReLU(inplace=True))
        else:
            self.cnn_emb = None
            cnn_emb_feat = 0
        self.encode_text_type = kwargs['encode_text_type']
        if self.encode_text_type == 'rnn':
            self.text_emb = nn.Embedding(vocab_size, text_emd_feat)
            self.rnn = nn.GRU(text_emd_feat, text_hidden_dim, bidirectional=bidirectional, batch_first=True)
            if bidirectional:
                text_hidden_dim *= 2
        elif self.encode_text_type == 'spacy':
            text_hidden_dim = 300
        else:
            text_hidden_dim = 0
        emb_feat = cnn_emb_feat + box_emb_feat + text_hidden_dim

        if self.norm_box_flag:
            self.box_emb = nn.Sequential(
                nn.Linear(8, box_emb_feat),
                # nn.BatchNorm1d(box_emb_feat,affine=False),
                nn.ReLU(inplace=True))
        else:
            self.box_emb = nn.Sequential(
                nn.Linear(4, box_emb_feat),
                # nn.BatchNorm1d(box_emb_feat,affine=False),
                nn.ReLU(inplace=True))
        self.fusion_linear = nn.Linear(emb_feat, gcn_out_feat)
        self.graph_type = graph_type
        # self.graph_module = self._get_graph_module(gcn_out_feat, gcn_out_feat, **kwargs)
        self.graph_modules = nn.ModuleList(
            [self._get_graph_module(gcn_out_feat, gcn_out_feat, **kwargs) for i in range(graph_layer_num)])
        # self._init_params()

    def _get_graph_module(self, emb_feat, gcn_out_feat, **kwargs):
        graph_module = None
        if self.graph_type == 'DGCNN':
            graph_module = DGCNNModule(emb_feat, gcn_out_feat, **kwargs)
        elif self.graph_type == 'GravNet':
            graph_module = GravnetModule(emb_feat, gcn_out_feat, **kwargs)
        return graph_module

    def forward(self, input=None, cell_boxes=None, targets=None, texts=None):
        #box 位置 feature
        box_feat = self.get_box_feat(copy.deepcopy(cell_boxes))
        fusion_feat = self.box_emb(box_feat)
        cnn_decode_feat = None
        if self.cnn_emb:
            #图像 feature
            cnn_decode_feat = self.decode_out(input)
            cnn_feat = ops.roi_align(cnn_decode_feat, cell_boxes, 2)  # [num_node, 256, 2, 2]
            cnn_feat = self.cnn_emb(cnn_feat.view(cnn_feat.size(0), -1))
            fusion_feat = torch.cat([fusion_feat, cnn_feat], dim=1)
        #text feature
        if self.encode_text_type == 'rnn':
            text_feat = torch.empty(0)
            for text in texts:
                text_embedding_feature = self.text_emb(text)
                textout, _ = self.rnn(torch.unsqueeze(text_embedding_feature, dim=0))
                textout = textout[:, -1, :]  # Take the output feature at last time sequence
                text_feat = torch.cat([text_feat, textout], dim=0)
            fusion_feat = torch.cat([fusion_feat, text_feat], dim=1)
        elif self.encode_text_type == 'spacy':
            text_feat = torch.cat([torch.as_tensor(np.array(text)) for text in texts]).to(fusion_feat.device)
            fusion_feat = torch.cat([fusion_feat, text_feat], dim=1)
        fusion_feat = self.fusion_linear(fusion_feat)
        input_feat = fusion_feat
        #图神经网络更新feature
        # feat, graphs = self.graph_module(cell_boxes, feat)
        for graph_module in self.graph_modules:
            feat, graphs = graph_module(cell_boxes, input_feat)
            input_feat = torch.maximum(feat, input_feat)
            # input_feat = feat+input_feat
        return feat, graphs, fusion_feat, cnn_decode_feat

    def get_box_feat(self, cell_boxes):
        if self.norm_box_flag:
            for cell_box in cell_boxes:
                min_x = cell_box[:, [0, 2]].min()
                min_y = cell_box[:, [1, 3]].min()
                cell_box_w = cell_box[:, [0, 2]].max() - min_x
                cell_box_h = cell_box[:, [1, 3]].max() - min_y
                cell_box[:, [0, 2]] = (cell_box[:, [0, 2]] - min_x) / cell_box_w
                cell_box[:, [1, 3]] = (cell_box[:, [1, 3]] - min_y) / cell_box_h
            boxes = torch.cat(cell_boxes, dim=0)
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            ctr_x = (boxes[:, 2] + boxes[:, 0]) / 2
            ctr_y = (boxes[:, 3] + boxes[:, 1]) / 2
            boxes_feat = torch.stack((boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], ctr_x, ctr_y, box_w, box_h),
                                     dim=1)
        else:
            boxes = torch.cat(cell_boxes, dim=0)
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            ctr_x = (boxes[:, 2] + boxes[:, 0]) / 2
            ctr_y = (boxes[:, 3] + boxes[:, 1]) / 2
            # rel_x = torch.log(ctr_x/self.width)
            # rel_y = torch.log(ctr_y/self.height)
            # rel_w = torch.log(box_w/self.width)
            # rel_h = torch.log(box_h/self.height)
            rel_x = ctr_x / self.width
            rel_y = ctr_y / self.height
            rel_w = box_w / self.width
            rel_h = box_h / self.height
            boxes_feat = torch.stack((rel_x, rel_y, rel_w, rel_h), dim=1)
        return boxes_feat

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


class FeatureFusionForFPN(nn.Module):

    def __init__(self):
        super(FeatureFusionForFPN, self).__init__()

        self.layer1_bn_relu = nn.Sequential(
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.layer2_bn_relu = nn.Sequential(
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.layer3_bn_relu = nn.Sequential(
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.layer4_bn_relu = nn.Sequential(
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.smooth1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True))

        self.smooth2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True))

        self.smooth3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True))
        # self._init_params()

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')
        return nn.functional.interpolate(x, size=(H // scale, W // scale), mode='bilinear', align_corners=False)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H, W), mode='bilinear') + y
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, images, fpn_outputs):
        # print(fpn_outputs['0'].shape,fpn_outputs['1'].shape,fpn_outputs['2'].shape)
        # the output of a group of fpn feature:
        # [('0', torch.Size([1, 256, 128, 128])),
        #  ('1', torch.Size([1, 256, 64, 64])),
        #  ('2', torch.Size([1, 256, 32, 32])),
        #  ('3', torch.Size([1, 256, 16, 16]))]
        layer1 = self.layer1_bn_relu(fpn_outputs['0'])
        layer2 = self.layer2_bn_relu(fpn_outputs['1'])
        layer3 = self.layer3_bn_relu(fpn_outputs['2'])
        layer4 = self.layer4_bn_relu(fpn_outputs['3'])

        fusion4_3 = self.smooth1(self._upsample_add(layer4, layer3))
        fusion4_2 = self.smooth2(self._upsample_add(fusion4_3, layer2))
        fusion4_1 = self.smooth3(self._upsample_add(fusion4_2, layer1))

        fusion4_2 = self._upsample(fusion4_2, fusion4_1)
        fusion4_3 = self._upsample(fusion4_3, fusion4_1)
        layer4 = self._upsample(layer4, fusion4_1)
        # fusion4_3 = self._upsample(fusion4_3, fusion4_2)
        # layer4 = self._upsample(layer4, fusion4_2)

        inter_feat = torch.cat((fusion4_1, fusion4_2, fusion4_3, layer4), 1)  # [N, 1024, H, W]
        inter_feat = self._upsample(inter_feat, images)  # [N, 1024, x_h, x_w]
        # inter_feat = torch.cat((fusion4_2, fusion4_3, layer4), 1) # [N, 1024, H, W]
        # inter_feat = self._upsample(inter_feat, x) # [N, 1024, x_h, x_w]

        return inter_feat


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






def polar(rect_src: list, rect_dst: list) -> Tuple[int, int]:
    """Compute distance and angle from src to dst bounding boxes (poolar coordinates considering the src as the center)
    Args:
        rect_src (list) : source rectangle coordinates
        rect_dst (list) : destination rectangle coordinates

    Returns:
        tuple (ints): distance and angle
    """

    # check relative position
    left = (rect_dst[2] - rect_src[0]) <= 0
    bottom = (rect_src[3] - rect_dst[1]) <= 0
    right = (rect_src[2] - rect_dst[0]) <= 0
    top = (rect_dst[3] - rect_src[1]) <= 0

    vp_intersect = (rect_src[0] <= rect_dst[2]
                    and rect_dst[0] <= rect_src[2])  # True if two rects "see" each other vertically, above or under
    hp_intersect = (rect_src[1] <= rect_dst[3]
                    and rect_dst[1] <= rect_src[3])  # True if two rects "see" each other horizontally, right or left
    rect_intersect = vp_intersect and hp_intersect

    center = lambda rect: ((rect[2] + rect[0]) / 2, (rect[3] + rect[1]) / 2)

    #  evaluate reciprocal position
    sc = center(rect_src)
    ec = center(rect_dst)
    new_ec = (ec[0] - sc[0], ec[1] - sc[1])
    angle = int(math.degrees(math.atan2(new_ec[1], new_ec[0])) % 360)

    if rect_intersect:
        return 0, angle
    elif top and left:
        a, b = (rect_dst[2] - rect_src[0]), (rect_dst[3] - rect_src[1])
        return int(math.sqrt(a**2 + b**2)), angle
    elif left and bottom:
        a, b = (rect_dst[2] - rect_src[0]), (rect_dst[1] - rect_src[3])
        return int(math.sqrt(a**2 + b**2)), angle
    elif bottom and right:
        a, b = (rect_dst[0] - rect_src[2]), (rect_dst[1] - rect_src[3])
        return int(math.sqrt(a**2 + b**2)), angle
    elif right and top:
        a, b = (rect_dst[0] - rect_src[2]), (rect_dst[3] - rect_src[1])
        return int(math.sqrt(a**2 + b**2)), angle
    elif left:
        return (rect_src[0] - rect_dst[2]), angle
    elif right:
        return (rect_dst[0] - rect_src[2]), angle
    elif bottom:
        return (rect_dst[1] - rect_src[3]), angle
    elif top:
        return (rect_src[1] - rect_dst[3]), angle


def to_bin(dist: int, angle: int, b=8) -> torch.Tensor:
    """ Discretize the space into equal "bins": return a distance and angle into a number between 0 and 1.

    Args:
        dist (int): distance in terms of pixel, given by "polar()" util function
        angle (int): angle between 0 and 360, given by "polar()" util function
        b (int): number of bins, MUST be power of 2

    Returns:
        torch.Tensor: new distance and angle (binary encoded)

    """

    def isPowerOfTwo(x):
        return (x and (not (x & (x - 1))))

    # dist
    assert isPowerOfTwo(b)
    m = max(dist) / b
    new_dist = []
    for d in dist:
        bin = int(d / m)
        if bin >= b:
            bin = b - 1
        bin = [int(x) for x in list('{0:0b}'.format(bin))]
        while len(bin) < math.sqrt(b):
            bin.insert(0, 0)
        new_dist.append(bin)

    # angle
    amplitude = 360 / b
    new_angle = []
    for a in angle:
        bin = (a - amplitude / 2)
        bin = int(bin / amplitude)
        bin = [int(x) for x in list('{0:0b}'.format(bin))]
        while len(bin) < math.sqrt(b):
            bin.insert(0, 0)
        new_angle.append(bin)

    return torch.cat([torch.tensor(new_dist, dtype=torch.float32), torch.tensor(new_angle, dtype=torch.float32)], dim=1)


def _get_nearest_pair_custom(cell_boxes_array, rope_max_length):
    cell_boxes_num = len(cell_boxes_array)
    eye_matrix = np.eye(cell_boxes_num)
    cell_boxes_height = cell_boxes_array[:, 3] - cell_boxes_array[:, 1]
    center_y_box_array = (cell_boxes_array[:, 1] + cell_boxes_array[:, 3]) / 2
    # find all i where mean(y0_i, y1_i) < y0_j, or mean(y0_i, y1_i) > y1_j for all j
    y_center_flag = np.logical_or(
        center_y_box_array.repeat(cell_boxes_num).reshape((cell_boxes_num, -1)).T <
        (cell_boxes_array[:, 1] - cell_boxes_height * 0.3).repeat(cell_boxes_num).reshape((cell_boxes_num, -1)),
        center_y_box_array.repeat(cell_boxes_num).reshape((cell_boxes_num, -1)).T >
        (cell_boxes_array[:, 3] + cell_boxes_height * 0.3).repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))
    center_x_box_array = (cell_boxes_array[:, 0] + cell_boxes_array[:, 2]) / 2
    x_center_flag = np.logical_or(
        center_x_box_array.repeat(cell_boxes_num).reshape(
            (cell_boxes_num, -1)).T < cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)),
        center_x_box_array.repeat(cell_boxes_num).reshape(
            (cell_boxes_num, -1)).T > cell_boxes_array[:, 2].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))
    # find all i where x1_i <= x1_j for all j
    left_dis_flag = np.logical_or(
        y_center_flag,
        np.logical_or(
            eye_matrix, cell_boxes_array[:, 2].repeat(cell_boxes_num).reshape(
                (cell_boxes_num, -1)).T <= cell_boxes_array[:, 2].repeat(cell_boxes_num).reshape((cell_boxes_num, -1))))
    # find all i where x0_i >= x0_j for all j
    right_dis_flag = np.logical_or(
        y_center_flag,
        np.logical_or(
            eye_matrix, cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape(
                (cell_boxes_num, -1)).T >= cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape((cell_boxes_num, -1))))
    # find all i where y1_i <= y1_j for all j
    up_dis_flag = np.logical_or(
        eye_matrix, cell_boxes_array[:, 3].repeat(cell_boxes_num).reshape(
            (cell_boxes_num, -1)).T <= cell_boxes_array[:, 3].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))
    # find all i where y0_i >= y0_j for all j
    down_dis_flag = np.logical_or(
        eye_matrix, cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape(
            (cell_boxes_num, -1)).T >= cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))
    # euclidean(x0_i - y1_j)
    hor_x_dis_matrix = (cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape(
        (cell_boxes_num, -1)).T - cell_boxes_array[:, 2].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))**2
    # euclidean(y0_i - y0_j)
    hor_y_dis_matrix = (cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape(
        (cell_boxes_num, -1)).T - cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))**2
    # euclidean(x0_i - x0_j)
    ver_x_dis_matrix = (cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape(
        (cell_boxes_num, -1)).T - cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))**2
    # euclidean(y0_i - y1_j)
    ver_y_dis_matrix = (cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape(
        (cell_boxes_num, -1)).T - cell_boxes_array[:, 3].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))**2

    left_dis_matrix = hor_x_dis_matrix + hor_y_dis_matrix
    left_dis_matrix[left_dis_flag] = math.inf
    right_dis_matrix = hor_x_dis_matrix + hor_y_dis_matrix
    right_dis_matrix[right_dis_flag] = math.inf
    up_dis_matrix = ver_x_dis_matrix + ver_y_dis_matrix
    up_dis_matrix[up_dis_flag] = math.inf
    down_dis_matrix = ver_x_dis_matrix + ver_y_dis_matrix
    down_dis_matrix[down_dis_flag] = math.inf
    down_y_dis_matrix = ver_y_dis_matrix
    down_y_dis_matrix[up_dis_flag] = math.inf
    down_y_dis_matrix[x_center_flag] = math.inf
    left_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                  np.argmin(left_dis_matrix, axis=1)]).T)[np.min(left_dis_matrix, axis=1) != math.inf]
    right_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                   np.argmin(right_dis_matrix,
                                             axis=1)]).T)[np.min(right_dis_matrix, axis=1) != math.inf]
    up_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                np.argmin(up_dis_matrix, axis=1)]).T)[np.min(up_dis_matrix, axis=1) != math.inf]
    down_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                  np.argmin(down_dis_matrix, axis=1)]).T)[np.min(down_dis_matrix, axis=1) != math.inf]
    down_y_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                    np.argmin(down_y_dis_matrix,
                                              axis=1)]).T)[np.min(down_y_dis_matrix, axis=1) != math.inf]

    # for vertical direction, we pick up at most two nodes w.r.t each node
    up_dis_matrix[np.array(range(cell_boxes_num)), np.argmin(up_dis_matrix, axis=1)] = math.inf
    down_dis_matrix[np.array(range(cell_boxes_num)), np.argmin(down_dis_matrix, axis=1)] = math.inf
    second_up_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                       np.argmin(up_dis_matrix, axis=1)]).T)[np.min(up_dis_matrix, axis=1) != math.inf]
    second_down_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                         np.argmin(down_dis_matrix,
                                                   axis=1)]).T)[np.min(down_dis_matrix, axis=1) != math.inf]

    # bring all selected neighbor node together
    ori_edge_index_list = left_index_list.tolist() + right_index_list.tolist() + up_index_list.tolist(
    ) + down_index_list.tolist() + second_up_index_list.tolist() + second_down_index_list.tolist(
    ) + down_y_index_list.tolist()
    ori_edge_index_list = list(
        set([(item[0], item[1]) if item[0] < item[1] else (item[1], item[0]) for item in ori_edge_index_list]))
    edge_index_list = [item for item in ori_edge_index_list if item[1] - item[0] < rope_max_length]
    # if len(edge_index_list) < len(ori_edge_index_list):
    #     logger.warning("delete edge:{}".format(set(ori_edge_index_list).difference(set(edge_index_list))))
    return edge_index_list


def _get_nearest_pair_beta_skeleton(cell_boxes_array):
    beta = 0.9

    costheta = math.sqrt(1 - 1 / (beta ** 2)) if beta > 1 else -math.sqrt(1 - beta ** 2)
    points = cell_boxes_array[:, [0, 1]].tolist()
    # + cell_boxes_array[:, [2, 1]].tolist()
    # ((cell_boxes_array[:, [0, 1]] + (
    #             cell_boxes_array[:, [2, 1]] - cell_boxes_array[:, [0, 1]]) / 4 * 1)).tolist() + \
    # ((cell_boxes_array[:, [0, 1]] + (
    #             cell_boxes_array[:, [2, 1]] - cell_boxes_array[:, [0, 1]]) / 4 * 2)).tolist() + \
    # ((cell_boxes_array[:, [0, 1]] + (
    #             cell_boxes_array[:, [2, 1]] - cell_boxes_array[:, [0, 1]]) / 4 * 3)).tolist()
    # ((cell_boxes_array[:, [0, 3]] + (
    #             cell_boxes_array[:, [2, 3]] - cell_boxes_array[:, [0, 3]]) / 6 * 1)).tolist() + \
    # ((cell_boxes_array[:, [0, 3]] + (
    #             cell_boxes_array[:, [2, 3]] - cell_boxes_array[:, [0, 3]]) / 6 * 2)).tolist() + \
    # ((cell_boxes_array[:, [0, 3]] + (
    #             cell_boxes_array[:, [2, 3]] - cell_boxes_array[:, [0, 3]]) / 6 * 3)).tolist() + \
    # ((cell_boxes_array[:, [0, 3]] + (
    #             cell_boxes_array[:, [2, 3]] - cell_boxes_array[:, [0, 3]]) / 6 * 4)).tolist() + \
    # ((cell_boxes_array[:, [0, 3]] + (
    #             cell_boxes_array[:, [2, 3]] - cell_boxes_array[:, [0, 3]]) / 6 * 5)).tolist()
    # cell_boxes_array[:, [0, 3]].tolist() + cell_boxes_array[:, [2, 3]].tolist() + \
    # ((cell_boxes_array[:, [0, 1]] + cell_boxes_array[:, [0, 3]]) / 2).tolist() + \
    # ((cell_boxes_array[:, [2, 1]] + cell_boxes_array[:, [2, 3]]) / 2).tolist() + \
    # ((cell_boxes_array[:, [0, 1]] + cell_boxes_array[:, [2, 1]]) / 2).tolist() +\
    # ((cell_boxes_array[:, [0, 3]] + cell_boxes_array[:, [2, 3]]) / 2).tolist() +\
    graph_neighbors = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if check_edge(i, j, points, costheta):
                graph_neighbors.append([i, j])
    # import nglpy
    # relaxed = True
    # p = 1.0
    # discrete_steps = -1
    # k_nearest_num = -1
    # graph_rep = nglpy.EmptyRegionGraph(
    #     max_neighbors=k_nearest_num,
    #     beta=beta,
    #     relaxed=relaxed,
    #     p=p,
    #     discrete_steps=discrete_steps,
    # )
    # graph_rep.build(points)
    # graph_neighbors = graph_rep.neighbors()
    # pair_cell_img = cv2.imread(img_path)
    # pair_cell_img1 = cv2.imread(img_path)
    # for key, neighbors in graph_neighbors.items():
    #     for neighbor in neighbors:
    #         # if key < box_num * 2 and neighbor < box_num * 2 and not (key > box_num and neighbor > box_num):
    #         cv2.line(pair_cell_img, list(np.array(points[key], int)), list(np.array(points[neighbor], int)),
    #                  (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)
    #         cv2.line(pair_cell_img1, list(np.array(points[key % box_num], int)),
    #                  list(np.array(points[neighbor % box_num], int)),
    #                  (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)
    # cv2.imwrite(os.path.join('/home/self_1_{}'.format(os.path.basename(img_path))), pair_cell_img)
    # cv2.imwrite(os.path.join('/home/self_2_{}'.format(os.path.basename(img_path))), pair_cell_img1)
    return graph_neighbors

def _get_nearest_pair_knn(cell_box, k_nearest_num):
    graph_transform = transforms.KNNGraph(k=k_nearest_num)
    # 尝试1
    # edge_index_list = graph_transform(graphs[index]).edge_index
    # 尝试2
    # pos = torch.as_tensor([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in cell_boxes[index]])
    # 尝试3
    # pos = torch.as_tensor(cell_box)
    # 尝试4
    pos = torch.as_tensor([box[0] for box in cell_box])
    graph = GraphData(pos=pos)
    edge_index_list = graph_transform(graph).edge_index
    pos = torch.as_tensor([box[1] for box in cell_box])
    graph = GraphData(pos=pos)
    edge_index_list = torch.cat((edge_index_list, graph_transform(graph).edge_index), dim=1)
    edge_index_list = edge_index_list.cpu().numpy().T.tolist()
    edge_index_list = list(
        set([(item[0], item[1]) if item[0] < item[1] else (item[1], item[0]) for item in edge_index_list]))
    return edge_index_list

    # 尝试6 nglpy的beta-skelton，难控制
    # import nglpy
    # self.beta = 1.0
    # self.relaxed = True
    # self.p = 2.0
    # self.discrete_steps = -1
    # graph_rep = nglpy.EmptyRegionGraph(
    #     max_neighbors=self.k_nearest_num,
    #     beta=self.beta,
    #     relaxed=self.relaxed,
    #     p=self.p,
    #     discrete_steps=self.discrete_steps,
    # )
    # cell_boxes_array = cell_boxes[index].cpu().numpy()
    # box_num = len(cell_boxes_array)
    # points = cell_boxes_array[:, [0, 1]].tolist() + cell_boxes_array[:, [2, 3]].tolist() + \
    #          cell_boxes_array[:, [0, 3]].tolist() + cell_boxes_array[:, [2, 1]].tolist() + \
    #          ((cell_boxes_array[:, [0, 1]] + cell_boxes_array[:, [0, 3]]) / 2).tolist() + \
    #          ((cell_boxes_array[:, [2, 1]] + cell_boxes_array[:, [2, 3]]) / 2).tolist() + \
    #          ((cell_boxes_array[:, [0, 1]] + (
    #                      cell_boxes_array[:, [2, 1]] - cell_boxes_array[:, [0, 1]]) / 6 * 1)).tolist() + \
    #          ((cell_boxes_array[:, [0, 1]] + (
    #                      cell_boxes_array[:, [2, 1]] - cell_boxes_array[:, [0, 1]]) / 6 * 2)).tolist() + \
    #          ((cell_boxes_array[:, [0, 1]] + (
    #                      cell_boxes_array[:, [2, 1]] - cell_boxes_array[:, [0, 1]]) / 6 * 3)).tolist() + \
    #          ((cell_boxes_array[:, [0, 1]] + (
    #                      cell_boxes_array[:, [2, 1]] - cell_boxes_array[:, [0, 1]]) / 6 * 4)).tolist() + \
    #          ((cell_boxes_array[:, [0, 1]] + (
    #                      cell_boxes_array[:, [2, 1]] - cell_boxes_array[:, [0, 1]]) / 6 * 5)).tolist() + \
    #          ((cell_boxes_array[:, [0, 3]] + (
    #                      cell_boxes_array[:, [2, 3]] - cell_boxes_array[:, [0, 3]]) / 6 * 1)).tolist() + \
    #          ((cell_boxes_array[:, [0, 3]] + (
    #                      cell_boxes_array[:, [2, 3]] - cell_boxes_array[:, [0, 3]]) / 6 * 2)).tolist() + \
    #          ((cell_boxes_array[:, [0, 3]] + (
    #                      cell_boxes_array[:, [2, 3]] - cell_boxes_array[:, [0, 3]]) / 6 * 3)).tolist() + \
    #          ((cell_boxes_array[:, [0, 3]] + (
    #                      cell_boxes_array[:, [2, 3]] - cell_boxes_array[:, [0, 3]]) / 6 * 4)).tolist() + \
    #          ((cell_boxes_array[:, [0, 3]] + (
    #                      cell_boxes_array[:, [2, 3]] - cell_boxes_array[:, [0, 3]]) / 6 * 5)).tolist()
    # # ((cell_boxes_array[:, [0, 1]] + cell_boxes_array[:, [2, 1]]) / 2).tolist() +\
    # # ((cell_boxes_array[:, [0, 3]] + cell_boxes_array[:, [2, 3]]) / 2).tolist() +\
    # graph_rep.build(points)
    # graph_neighbors = graph_rep.neighbors()
    # edge_index_list = [[key % box_num, neighbor % box_num]
    #                    for key, neighbors in graph_neighbors.items()
    #                    for neighbor in neighbors
    #                    if abs(key - neighbor) % box_num != 0]
    # 尝试5 太慢了 修改为numpy方式
    # cell_boxes_array = cell_boxes[index].cpu().numpy()
    # cell_boxes_num = len(cell_boxes_array)
    # eye_matrix = np.eye(cell_boxes_num)
    # center_box_array = (cell_boxes_array[:, 1] + cell_boxes_array[:, 3]) / 2
    # # find all i where mean(y0_i, y1_i) < y0_j, or mean(y0_i, y1_i) > y1_j for all j
    # y_center_flag = np.logical_or(
    #     center_box_array.repeat(cell_boxes_num).reshape(
    #         (cell_boxes_num, -1)).T < cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape(
    #         (cell_boxes_num, -1)),
    #     center_box_array.repeat(cell_boxes_num).reshape(
    #         (cell_boxes_num, -1)).T > cell_boxes_array[:, 3].repeat(cell_boxes_num).reshape(
    #         (cell_boxes_num, -1)))
    # # find all i where x1_i <= x1_j for all j
    # left_dis_flag = np.logical_or(
    #     y_center_flag,
    #     np.logical_or(
    #         eye_matrix, cell_boxes_array[:, 2].repeat(cell_boxes_num).reshape(
    #             (cell_boxes_num, -1)).T <= cell_boxes_array[:, 2].repeat(cell_boxes_num).reshape(
    #             (cell_boxes_num, -1))))
    # # find all i where x0_i >= x0_j for all j
    # right_dis_flag = np.logical_or(
    #     y_center_flag,
    #     np.logical_or(
    #         eye_matrix, cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape(
    #             (cell_boxes_num, -1)).T >= cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape(
    #             (cell_boxes_num, -1))))
    # # find all i where y1_i <= y1_j for all j
    # up_dis_flag = np.logical_or(
    #     eye_matrix, cell_boxes_array[:, 3].repeat(cell_boxes_num).reshape(
    #         (cell_boxes_num, -1)).T <= cell_boxes_array[:, 3].repeat(cell_boxes_num).reshape(
    #         (cell_boxes_num, -1)))
    # # find all i where y0_i >= y0_j for all j
    # down_dis_flag = np.logical_or(
    #     eye_matrix, cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape(
    #         (cell_boxes_num, -1)).T >= cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape(
    #         (cell_boxes_num, -1)))
    # # euclidean(x0_i - y1_j)
    # hor_x_dis_matrix = (cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape(
    #     (cell_boxes_num, -1)).T - cell_boxes_array[:, 2].repeat(cell_boxes_num).reshape(
    #     (cell_boxes_num, -1))) ** 2
    # # euclidean(y0_i - y0_j)
    # hor_y_dis_matrix = (cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape(
    #     (cell_boxes_num, -1)).T - cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape(
    #     (cell_boxes_num, -1))) ** 2
    # # euclidean(x0_i - x0_j)
    # ver_x_dis_matrix = (cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape(
    #     (cell_boxes_num, -1)).T - cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape(
    #     (cell_boxes_num, -1))) ** 2
    # # euclidean(y0_i - y1_j)
    # ver_y_dis_matrix = (cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape(
    #     (cell_boxes_num, -1)).T - cell_boxes_array[:, 3].repeat(cell_boxes_num).reshape(
    #     (cell_boxes_num, -1))) ** 2
    #
    # left_dis_matrix = hor_x_dis_matrix + hor_y_dis_matrix
    # left_dis_matrix[left_dis_flag] = math.inf
    # right_dis_matrix = hor_x_dis_matrix + hor_y_dis_matrix
    # right_dis_matrix[right_dis_flag] = math.inf
    # up_dis_matrix = ver_x_dis_matrix + ver_y_dis_matrix
    # up_dis_matrix[up_dis_flag] = math.inf
    # down_dis_matrix = ver_x_dis_matrix + ver_y_dis_matrix
    # down_dis_matrix[down_dis_flag] = math.inf
    # left_index_list = (np.vstack([np.array(range(cell_boxes_num)),
    #                               np.argmin(left_dis_matrix,
    #                                         axis=1)]).T)[np.min(left_dis_matrix, axis=1) != math.inf]
    # right_index_list = (np.vstack([np.array(range(cell_boxes_num)),
    #                                np.argmin(right_dis_matrix,
    #                                          axis=1)]).T)[np.min(right_dis_matrix, axis=1) != math.inf]
    # up_index_list = (np.vstack([np.array(range(cell_boxes_num)),
    #                             np.argmin(up_dis_matrix,
    #                                       axis=1)]).T)[np.min(up_dis_matrix, axis=1) != math.inf]
    # down_index_list = (np.vstack([np.array(range(cell_boxes_num)),
    #                               np.argmin(down_dis_matrix,
    #                                         axis=1)]).T)[np.min(down_dis_matrix, axis=1) != math.inf]
    #
    # # for vertical direction, we pick up at most two nodes w.r.t each node
    # up_dis_matrix[np.array(range(cell_boxes_num)), np.argmin(up_dis_matrix, axis=1)] = math.inf
    # down_dis_matrix[np.array(range(cell_boxes_num)), np.argmin(down_dis_matrix, axis=1)] = math.inf
    # second_up_index_list = (np.vstack([np.array(range(cell_boxes_num)),
    #                                    np.argmin(up_dis_matrix,
    #                                              axis=1)]).T)[np.min(up_dis_matrix, axis=1) != math.inf]
    # second_down_index_list = (np.vstack(
    #     [np.array(range(cell_boxes_num)),
    #      np.argmin(down_dis_matrix, axis=1)]).T)[np.min(down_dis_matrix, axis=1) != math.inf]
    #
    # # bring all selected neighbor node together
    # edge_index_list = left_index_list.tolist() + right_index_list.tolist() + up_index_list.tolist(
    # ) + down_index_list.tolist() + second_up_index_list.tolist() + second_down_index_list.tolist()


def check_angle(P, Q, R, costheta):
    '''判断角度条件'''
    RP = P - R
    RQ = Q - R
    cosalpha = np.dot(RP, RQ) / math.sqrt(np.dot(RP, RP) * np.dot(RQ, RQ))
    return True if cosalpha <= costheta else False


def check_edge(i, j, points, costheta):
    '''检查边是否满足空区域条件'''
    for k in range(len(points)):
        if i != k and j != k and check_angle(np.array(points[i]), np.array(points[j]), np.array(points[k]), costheta):
            return False
    return True