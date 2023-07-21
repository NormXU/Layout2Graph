# -*- coding:utf-8 -*-
# create: 2021/7/1
from .graph_layout_metrics import GraphLayoutMetric


def get_metric(metric_args):
    metric_type = metric_args['type']
    return eval(metric_type)(**metric_args)
