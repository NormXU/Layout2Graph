# -*- coding:utf-8 -*-
# create: 2021/7/15
from .graph_net.graph_layout_net import GraphLayoutNet


def get_network(model_args):
    model_type = model_args.pop("type")
    return eval(model_type)(**model_args)