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
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import itertools

from .gcn.dgcnn import DGCNNModule
from .gcn.gravnet import GravnetModule
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


class GraphTableNet(nn.Module):

    def __init__(self, **kwargs):
        super(GraphTableNet, self).__init__()
        self.focal_loss_flag = kwargs['focal_loss_flag']
        self.max_pair_num = kwargs['max_pair_num']
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
        self.linear_row = nn.Sequential(nn.Linear(gcn_out_feat * 2, gcn_out_feat), nn.LeakyReLU(inplace=True))
        self.pair_row_pred = nn.Sequential(nn.Linear(gcn_out_feat, 2), nn.LeakyReLU(inplace=True))
        self.linear_col = nn.Sequential(nn.Linear(gcn_out_feat * 2, gcn_out_feat), nn.LeakyReLU(inplace=True))
        self.pair_col_pred = nn.Sequential(nn.Linear(gcn_out_feat, 2), nn.LeakyReLU(inplace=True))
        self.linear_cell = nn.Sequential(nn.Linear(gcn_out_feat * 2, gcn_out_feat), nn.LeakyReLU(inplace=True))
        self.cls_cell = nn.Sequential(nn.Linear(gcn_out_feat, 2), nn.LeakyReLU(inplace=True))
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

    def forward(self, images, cell_boxes, targets=None, texts=None):
        if self.backbone:
            inter_feat = self.backbone(images)
            inter_feat = self.smooth_layer(images, inter_feat)
            feat, graphs, fusion_feat, _ = self.graph(input=inter_feat,
                                                      cell_boxes=cell_boxes,
                                                      targets=targets,
                                                      texts=texts)
        else:
            feat, graphs, fusion_feat, _ = self.graph(input=None, cell_boxes=cell_boxes, targets=targets, texts=texts)
        # 正负样本采样
        pair_row_target, pair_row, pair_col_target, pair_col, pair_cell_target, pair_cell = self.get_sampling(
            cell_boxes, targets, graphs, self.focal_loss_flag, images.device)
        if self.graph.graph_type == "GCN":
            feat_row, feat_col, feat_cell = feat
        else:
            feat_row = feat
            feat_col = feat
            feat_cell = feat
        pair_row_score_list, pair_col_score_list, pair_cell_score_list = torch.empty(0).to(
            images.device), torch.empty(0).to(images.device), torch.empty(0).to(images.device)
        # 分类
        flag = random.random() < 0.5
        for i in range(0, max(pair_row.shape[0], pair_col.shape[0], pair_cell.shape[0]) // self.max_pair_num + 1):
            if flag:
                pair_feat_row = torch.cat(
                    (torch.as_tensor(feat_row[pair_row[i * self.max_pair_num:(i + 1) * self.max_pair_num, 0]]),
                     torch.as_tensor(feat_row[pair_row[i * self.max_pair_num:(i + 1) * self.max_pair_num, 1]])),
                    dim=1)
                pair_feat_col = torch.cat(
                    (torch.as_tensor(feat_col[pair_col[i * self.max_pair_num:(i + 1) * self.max_pair_num, 0]]),
                     torch.as_tensor(feat_col[pair_col[i * self.max_pair_num:(i + 1) * self.max_pair_num, 1]])),
                    dim=1)
                pair_feat_cell = torch.cat(
                    (torch.as_tensor(feat_cell[pair_cell[i * self.max_pair_num:(i + 1) * self.max_pair_num, 0]]),
                     torch.as_tensor(feat_cell[pair_cell[i * self.max_pair_num:(i + 1) * self.max_pair_num, 1]])),
                    dim=1)
            else:
                pair_feat_row = torch.cat(
                    (torch.as_tensor(feat_row[pair_row[i * self.max_pair_num:(i + 1) * self.max_pair_num, 1]]),
                     torch.as_tensor(feat_row[pair_row[i * self.max_pair_num:(i + 1) * self.max_pair_num, 0]])),
                    dim=1)
                pair_feat_col = torch.cat(
                    (torch.as_tensor(feat_col[pair_col[i * self.max_pair_num:(i + 1) * self.max_pair_num, 1]]),
                     torch.as_tensor(feat_col[pair_col[i * self.max_pair_num:(i + 1) * self.max_pair_num, 0]])),
                    dim=1)
                pair_feat_cell = torch.cat(
                    (torch.as_tensor(feat_cell[pair_cell[i * self.max_pair_num:(i + 1) * self.max_pair_num, 1]]),
                     torch.as_tensor(feat_cell[pair_cell[i * self.max_pair_num:(i + 1) * self.max_pair_num, 0]])),
                    dim=1)
            linear_row = self.linear_row(pair_feat_row)
            cls_score_row = self.pair_row_pred(linear_row)
            linear_col = self.linear_col(pair_feat_col)
            cls_score_col = self.pair_col_pred(linear_col)
            linear_cell = self.linear_cell(pair_feat_cell)
            cls_score_cell = self.cls_cell(linear_cell)
            pair_row_score_list = torch.cat([pair_row_score_list, cls_score_row], dim=0)
            pair_col_score_list = torch.cat([pair_col_score_list, cls_score_col], dim=0)
            pair_cell_score_list = torch.cat([pair_cell_score_list, cls_score_cell], dim=0)
        data = {
            'pair_row_score_list': pair_row_score_list,
            'pair_row': pair_row,
            'pair_row_pred': torch.argmax(pair_row_score_list, dim=1),
            'pair_row_target': pair_row_target,
            'pair_col': pair_col,
            'pair_col_score_list': pair_col_score_list,
            'pair_col_pred': torch.argmax(pair_col_score_list, dim=1),
            'pair_col_target': pair_col_target,
            'pair_cell': pair_cell,
            'pair_cell_score_list': pair_cell_score_list,
            'pair_cell_pred': torch.argmax(pair_cell_score_list, dim=1),
            'pair_cell_target': pair_cell_target,
        }
        return data

    def get_sampling(self, cell_boxes, targets, graphs, focal_loss_flag, device):
        pair_row_target, pair_row = [], []
        pair_col_target, pair_col = [], []
        pair_cell_target, pair_cell = [], []
        #预测时k最近邻/全连接，两两配对去预测关系
        if targets is None:
            start_index = 0
            for cell_box in cell_boxes:
                box_num = cell_box.shape[0]
                pair_row.extend(list(itertools.combinations(range(0 + start_index, box_num + start_index), r=2)))
                pair_col.extend(list(itertools.combinations(range(0 + start_index, box_num + start_index), r=2)))
                pair_cell.extend(list(itertools.combinations(range(0 + start_index, box_num + start_index), r=2)))
                start_index += box_num
        #训练时正负样本均衡采样
        else:
            start_index = 0
            for target in targets:
                positive_cell_pair, negative_cell_pair = [], []
                positive_row_pair, negative_row_pair = [], []
                positive_col_pair, negative_col_pair = [], []
                for i, loc1 in enumerate(target):
                    for j, loc2 in enumerate(target[i + 1:]):
                        if loc1 == loc2:
                            positive_cell_pair.append([i + start_index, j + i + 1 + start_index])
                        else:
                            negative_cell_pair.append([i + start_index, j + i + 1 + start_index])
                        if loc1[:2] == loc2[:2] or (loc1[0] <= loc2[0]
                                                    and loc1[1] >= loc2[1]) or (loc2[0] <= loc1[0]
                                                                                and loc2[1] >= loc1[1]):
                            positive_row_pair.append([i + start_index, j + i + 1 + start_index])
                        else:
                            negative_row_pair.append([i + start_index, j + i + 1 + start_index])
                        if loc1[2:] == loc2[2:] or (loc1[2] <= loc2[2]
                                                    and loc1[3] >= loc2[3]) or (loc2[2] <= loc1[2]
                                                                                and loc2[3] >= loc1[3]):
                            positive_col_pair.append([i + start_index, j + i + 1 + start_index])
                        else:
                            negative_col_pair.append([i + start_index, j + i + 1 + start_index])

                if focal_loss_flag is False:
                    random.shuffle(negative_cell_pair)
                    random.shuffle(negative_row_pair)
                    random.shuffle(negative_col_pair)
                    pair_cell.extend(positive_cell_pair + negative_cell_pair[:len(positive_cell_pair)])
                    pair_row.extend(positive_row_pair + negative_row_pair[:len(positive_row_pair)])
                    pair_col.extend(positive_col_pair + negative_col_pair[:len(positive_col_pair)])
                    pair_cell_target.extend([1] * len(positive_cell_pair) +
                                            [0] * len(negative_cell_pair[:len(positive_cell_pair)]))
                    pair_row_target.extend([1] * len(positive_row_pair) +
                                           [0] * len(negative_row_pair[:len(positive_row_pair)]))
                    pair_col_target.extend([1] * len(positive_col_pair) +
                                           [0] * len(negative_col_pair[:len(positive_col_pair)]))
                    # print(len(positive_col_pair), len(positive_row_pair), len(positive_cell_pair))
                    # print(len(pair_col), len(pair_col_target), len(pair_row), len(pair_row_target), len(pair_cell), len(pair_cell_target))
                else:
                    pair_cell.extend(positive_cell_pair + negative_cell_pair)
                    pair_row.extend(positive_row_pair + negative_row_pair)
                    pair_col.extend(positive_col_pair + negative_col_pair)
                    pair_cell_target.extend([1] * len(positive_cell_pair) + [0] * len(negative_cell_pair))
                    pair_row_target.extend([1] * len(positive_row_pair) + [0] * len(negative_row_pair))
                    pair_col_target.extend([1] * len(positive_col_pair) + [0] * len(negative_col_pair))
                start_index += len(target)

        return torch.from_numpy(np.array(pair_row_target)).to(device), np.array(pair_row), torch.from_numpy(
            np.array(pair_col_target)).to(device), np.array(pair_col), torch.from_numpy(
                np.array(pair_cell_target)).to(device), np.array(pair_cell)


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
        if self.k_nearest_num:
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
                else:
                    positive_cell_pair = []
                    for i, loc1 in enumerate(target):
                        for j, loc2 in enumerate(target[i + 1:]):
                            if loc1 == loc2:
                                positive_cell_pair.append((i, j + i + 1))
                negative_cell_pair = list(set(pair_cell_list[index]).difference(set(positive_cell_pair)))
                # assert len(set(positive_cell_pair).difference(set(pair_cell_list[index]))) == 0, "!!Lack of pair Set:{}".format(set(positive_cell_pair).difference(set(pair_cell_list[index])))
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
            else:
                # edge_index_list = _get_nearest_pair_knn(cell_boxes[index], self.k_nearest_num)
                cell_boxes_array = cell_boxes[index].cpu().numpy()
                edge_index_list = _get_nearest_pair_custom(cell_boxes_array, self.rope_max_length)
            pair_cell_list.append(edge_index_list)
        return pair_cell_list


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
    # 尝试1
    # edge_index_list = self.graph_transform(graphs[index]).edge_index
    # 尝试2
    # edge_index_list = graph_transform(graphs[index]).edge_index
    # 尝试3
    # pos = torch.as_tensor([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in cell_boxes[index]])
    # 尝试4
    graph_transform = transforms.KNNGraph(k=k_nearest_num)
    # pos = torch.as_tensor([box[0] for box in cell_box])
    pos = torch.as_tensor(cell_box)
    graph = GraphData(pos=pos)
    edge_index_list = graph_transform(graph).edge_index
    # pos = torch.as_tensor([box[1] for box in cell_box])
    # graph = GraphData(pos=pos)
    # edge_index_list = torch.cat((edge_index_list, graph_transform(graph).edge_index), dim=1)
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