# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/6/8
from mydatasets.base_datasets import BaseDataset, BaseImgDataset

from .gragh_net.layout_dataset import GraphLayoutDataset, GraphLayoutEntityDataset

def get_dataset(dataset_args):
    dataset_type = dataset_args.get("type")
    dataset = eval(dataset_type)(**dataset_args)
    return dataset
