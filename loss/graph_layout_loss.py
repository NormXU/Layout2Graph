import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.utils import class_weight

# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
from torch.nn import CrossEntropyLoss

from loss import FocalLoss
from loss.graph_layout_contrastive_loss import LayoutContrastiveLoss


class GraphLayoutLoss(nn.Module):

    def __init__(self, edge_type, gamma=2, alpha=0.25, size_average=True, loss_weight=1, **kwargs):
        super(GraphLayoutLoss, self).__init__()
        if edge_type == 'FocalLoss':
            self.edge_loss = FocalLoss(gamma, alpha, size_average)
        elif 'edge_weight' in kwargs and kwargs['edge_weight'] is not None:
            self.edge_loss = CrossEntropyLoss(weight=torch.tensor(kwargs.get('edge_weight')))
        else:
            self.edge_loss = CrossEntropyLoss()
        if 'node_weight' in kwargs and kwargs['node_weight'] is not None:
            self.node_loss = CrossEntropyLoss(weight=torch.tensor(kwargs.get('node_weight')))
        else:
            self.node_loss = CrossEntropyLoss()
        self.loss_weight = loss_weight
        self.num_classes = kwargs.get('num_classes')
        self.class_weight_flag = kwargs.get('class_weight_flag', False)
        self.node_contrastive_loss_flag = kwargs.get('node_contrastive_loss_flag', False)
        self.edge_contrastive_loss_flag = kwargs.get('edge_contrastive_loss_flag', False)

    def forward(self, outputs, cls_targets):
        losses = {}
        if self.class_weight_flag:
            try:
                w = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=range(2),
                    # classes=np.unique(outputs['pair_cell_target'].cpu().numpy()),
                    y=outputs['pair_cell_target'].cpu().numpy())
                edge_loss = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor(w, dtype=torch.float32).to(outputs['pair_cell_target'].device))
            except:
                edge_loss = self.edge_loss
                # print("ERROR-Pair:", outputs['pair_cell_target'].cpu().numpy())
            try:
                w = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=range(self.num_classes),
                    # classes=np.unique(cls_targets.cpu().numpy()),
                    y=cls_targets.cpu().numpy())
                node_loss = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor(w, dtype=torch.float32).to(outputs['pair_cell_target'].device))
            except:
                node_loss = self.node_loss
                # print("ERROR-Node:", cls_targets.cpu().numpy())
        else:
            node_loss = self.node_loss
            edge_loss = self.edge_loss
        edge_loss.to(outputs['pair_cell_target'].device)
        losses['loss_cell'] = edge_loss(
            outputs['pair_cell_score_list'],
            outputs['pair_cell_target']) if len(outputs['pair_cell_score_list']) > 0 else 0
        node_loss.to(outputs['pair_cell_target'].device)
        losses['loss_cls'] = node_loss(outputs['node_score_list'], cls_targets)
        node_num = outputs['node_score_list'].size()[0]
        if self.edge_contrastive_loss_flag:
            # 对比loss
            mask = torch.zeros(node_num, node_num)
            neg_pair = torch.as_tensor(outputs['pair_cell'])[outputs['pair_cell_target'] == 0].T.cpu().numpy()
            mask[neg_pair] = 1.0
            edge_contrastive_loss = LayoutContrastiveLoss(node_num)
            loss_edge_con = edge_contrastive_loss(outputs['node_feat'], outputs['node_feat'], mask)
            losses['loss_cell'] += float((losses['loss_cell'] / loss_edge_con).cpu()) * loss_edge_con
        if self.node_contrastive_loss_flag:
            # 对比loss
            mask = (~(cls_targets.expand(node_num, node_num) == cls_targets.expand(node_num, node_num).T)).float()
            node_contrastive_loss = LayoutContrastiveLoss(node_num)
            loss_node_con = node_contrastive_loss(outputs['node_feat'], outputs['node_feat'], mask)
            losses['loss_cls'] += float((losses['loss_cls'] / loss_node_con).cpu()) * loss_node_con
        if self.loss_weight == -1:
            loss_weight = float((losses['loss_cell'] / losses['loss_cls']).cpu())
            losses['loss'] = loss_weight * losses['loss_cls'] + losses['loss_cell']
        else:
            losses['loss'] = self.loss_weight * losses['loss_cls'] + losses['loss_cell']
        return losses
