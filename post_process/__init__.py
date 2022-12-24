#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:weishu
# datetime:2022/5/24 1:37 上午
# software: PyCharm
from .graph_net.default_post_process import DefaultPostProcessor


def get_post_processor(processor_args):
    processor_type = processor_args.get("type")
    return eval(processor_type)(**processor_args)
