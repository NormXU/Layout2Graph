# encoding: utf-8
'''
@email: weishu@datagrand.com
@software: Pycharm
@time: 2021/11/23 2:27 下午
@desc:
'''
from torch.utils.data import Dataset
import os
import torch
import copy
from PIL import Image

from base.common_util import get_file_path_list
from base.driver import logger
import random
import cv2
import json
from torchvision import transforms
import pickle
import numpy as np


class GraphLayoutDataset(Dataset):

    def __init__(self, data_root, label_root, **kwargs):
        super(GraphLayoutDataset, self).__init__()
        self.file_path_list = []
        self.label_path_list = []
        for i, data_path in enumerate(label_root):
            label_path_list = get_file_path_list(data_path, ['json'])
            for label_path in label_path_list:
                self.label_path_list.append(label_path)
                img_path = label_path.replace(data_path,
                                              data_root[i]).replace('/graph_labels/',
                                                                    '/ocr_results_images/').replace('.json', '.jpg')
                if not os.path.exists(img_path):
                    img_path = img_path.replace('/ocr_results_images/', '/images/')
                if not os.path.exists(img_path):
                    img_path = img_path.replace('.jpg', '.png')
                if not os.path.exists(img_path):
                    img_path = img_path.replace('.png', '.jpeg')
                self.file_path_list.append(img_path)
        self.label_list = [None] * len(self.label_path_list)
        self.crop_img_flag = kwargs['crop_img_flag']
        self.encode_text_type = kwargs.get('encode_text_type', None)
        if self.encode_text_type == 'spacy':
            import spacy
            self.text_emb = spacy.load('en_core_web_lg')

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, index):
        image = Image.open(self.file_path_list[index]).convert("RGB")
        if self.label_list[index] is None:
            with open(self.label_path_list[index], 'r') as f:
                try:
                    json_data = json.load(f)
                except:
                    logger.warning('bad json:{}'.format(self.label_path_list[index]))
                    return self[index - 1]
                cell_box, cell_lloc, cell_content, encode_text = [], [], [], []
                for item in json_data['img_data_list']:
                    # TODO 可以过滤一些文本框不去预测
                    if len(item['content']) > 0 and item['text_coor'][2] > item['text_coor'][0] and item['text_coor'][3] > item['text_coor'][1]:
                        cell_box.append(item['text_coor'])
                        cell_lloc.append(item['label'])
                        cell_content.append(item['content'])
                        if self.encode_text_type == 'spacy':
                            textout = self.text_emb(item['content']).vector
                            encode_text.append(textout)
                if len(cell_box) == 0:
                    return self[index - 1]
            if self.crop_img_flag:
                cell_box = np.array(cell_box)
                min_x = cell_box[:, [0, 2]].min()
                min_y = cell_box[:, [1, 3]].min()
                max_x = cell_box[:, [0, 2]].max()
                max_y = cell_box[:, [1, 3]].max()
                image = image.crop([min_x, min_y, max_x, max_y])
                cell_box[:, [0, 2]] -= min_x
                cell_box[:, [1, 3]] -= min_y
                cell_box = cell_box.tolist()
            self.label_list[index] = {
                "cell_box": cell_box,
                'target': cell_lloc,
                'text': cell_content,
                'encode_text': encode_text
            }
        return {
            "image": image,
            'image_name': self.file_path_list[index],
            "cell_box": self.label_list[index]['cell_box'],
            'target': self.label_list[index]['target'],
            'text': self.label_list[index]['text'],
            'encode_text': self.label_list[index]['encode_text']
        }


class GraphLayoutEntityDataset(GraphLayoutDataset):

    def __getitem__(self, index):
        image = Image.open(self.file_path_list[index]).convert("RGB")
        if self.label_list[index] is None:
            with open(self.label_path_list[index], 'r') as f:
                json_data = json.load(f)
                cell_box, cell_lloc, cell_content, linking, encode_text = [], [], [], [], []
                for item in json_data['img_data_list']:
                    cell_box.append(item['text_coor'])
                    cell_lloc.append(item['label'])
                    cell_content.append(item['content'])
                    linking.append(item['linking'])
                    if self.encode_text_type == 'spacy':
                        textout = self.text_emb(item['content']).vector
                        encode_text.append(textout)
                label_index_list = [int(label.split("_")[1]) for label in cell_lloc]
                linking = [(label_index_list.index(link[0]), label_index_list.index(link[1]))
                           for link_list in linking
                           for link in link_list]
                linking = [link if link[0] < link[1] else (link[1], link[0]) for link in linking]
                linking = [link for link in linking if link[0] != link[1]]
                if len(cell_box) == 0:
                    return self[index + 1]
                self.label_list[index] = {
                    "cell_box": cell_box,
                    'target': cell_lloc,
                    'text': cell_content,
                    'linking': linking,
                    'encode_text': encode_text
                }
        return {
            "image": image,
            'image_name': self.file_path_list[index],
            "cell_box": self.label_list[index]['cell_box'],
            'target': self.label_list[index]['target'],
            'text': self.label_list[index]['text'],
            'linking': self.label_list[index]['linking'],
            'encode_text': self.label_list[index]['encode_text']
        }
