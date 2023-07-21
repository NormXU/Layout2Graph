# -*- coding:utf-8 -*-
# create: 2021/7/16
# region for master
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.nn import CrossEntropyLoss, CTCLoss, MSELoss










# endregion
from .focal_loss import FocalLoss
from .graph_layout_loss import GraphLayoutLoss


def get_criterion(criterion_type, criterion_args):
    return eval(criterion_type)(**criterion_args)