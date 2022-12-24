# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/7/15
from .graph_net.graph_net import GraphLayoutNet, GraphTableNet


def get_network(model_args):
    model_type = model_args.get("type")
    return eval(model_type)(**model_args)