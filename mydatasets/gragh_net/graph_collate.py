#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:weishu
# datetime:2023/2/8 2:33 下午
# software: PyCharm
import torch
import copy
from PIL import Image
import random
import cv2
from torchvision import transforms
import numpy as np


def graph_get_pad_transform(image, cell_box, height, width):
    tb_w, tb_h = image.size
    img_transform = []
    im_scale = float(height) / float(tb_h)
    if int(im_scale * tb_w) > width:
        im_scale = float(width) / float(tb_w)
    rew = int(tb_w * im_scale)
    reh = int(tb_h * im_scale)
    pdl, pdt = ((width - rew) // 2, (height - reh) // 2)
    pdr, pdd = (width - rew - pdl, height - reh - pdt)
    img_transform.append(transforms.Resize((reh, rew), Image.BICUBIC))
    img_transform.append(transforms.Pad((pdl, pdt, pdr, pdd), padding_mode='edge'))
    img_transform.append(transforms.ToTensor())
    img_transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    img_transformation = transforms.Compose(img_transform)
    image = img_transformation(image)
    cell_box = np.array(cell_box) * im_scale + np.array([pdl, pdt] * 2)
    return image, cell_box, (pdl, pdt, pdr, pdd)


def graph_get_resize_transform(image, cell_box, height, width):
    tb_w, tb_h = image.size
    img_transform = []
    h_scale, w_scale = height / tb_h, width / tb_w
    img_transform.append(transforms.Resize((height, width), Image.BICUBIC))
    img_transform.append(transforms.ToTensor())
    img_transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    img_transformation = transforms.Compose(img_transform)
    image = img_transformation(image)
    cell_box = (np.array(cell_box).reshape(-1, 2) * np.array([w_scale, h_scale])).reshape(-1, 4)
    return image, cell_box, (0, 0, 0, 0)


def variety_cell(images, cell_boxes, targets, texts, padding_edge, cut_percent, variety_percent, delete_percent,
                 cut_ratio, variety_ratio, delete_ratio):
    img_size = [
        [valid_range[0], valid_range[1],
         int(image.shape[1]) - valid_range[2],
         int(image.shape[2]) - valid_range[3]]  # [l, t, r, d]
        for image, valid_range in zip(images, padding_edge)
    ]
    random.seed(1)
    cell_boxes = [list(cell_box) for cell_box in cell_boxes]
    for i, cell_box in enumerate(cell_boxes):
        choose_index_list = random.sample(list(range(len(cell_box))),
                                          int((cut_percent + variety_percent) *
                                              len(cell_box)))  # 随机选出需要做变换（包括cut和偏移）的框index
        for choose_index in choose_index_list:
            width = cell_box[choose_index][2] - cell_box[choose_index][0]
            height = cell_box[choose_index][3] - cell_box[choose_index][1]
            if random.random() < cut_percent / (cut_percent + variety_percent):  # 做cut
                if width < 50:
                    continue
                cut_strategy = random.random()
                if cut_strategy <= 1.0:  # 只cut x
                    cut_x_ratio = random.random()
                    cut_x_ratio = cut_x_ratio if cut_x_ratio < 0.5 else 1 - cut_x_ratio
                    cut_box1 = copy.deepcopy(cell_box[choose_index])
                    cut_box2 = copy.deepcopy(cell_box[choose_index])
                    cut_box1[2] = cell_box[choose_index][0] + (
                        1 + random.random() * random.choice([-cut_ratio, cut_ratio])) * width * cut_x_ratio
                    cut_box2[0] = cell_box[choose_index][0] + (
                        1 + random.random() * random.choice([-cut_ratio, cut_ratio])) * width * cut_x_ratio
                    text1 = texts[i][choose_index][:int(cut_x_ratio * len(texts[i][choose_index]))]
                    text2 = texts[i][choose_index][int(cut_x_ratio * len(texts[i][choose_index])):]
                elif cut_strategy < 1.0:  # 只cut y
                    cut_y_ratio = random.random()
                    cut_y_ratio = cut_y_ratio if cut_y_ratio < 0.5 else 1 - cut_y_ratio
                    cut_box1 = copy.deepcopy(cell_box[choose_index])
                    cut_box2 = copy.deepcopy(cell_box[choose_index])
                    cut_box1[3] = cell_box[choose_index][1] + (
                        1 + random.random() * random.choice([-cut_ratio, cut_ratio])) * height * cut_y_ratio
                    cut_box2[1] = cell_box[choose_index][1] + (
                        1 + random.random() * random.choice([-cut_ratio, cut_ratio])) * height * cut_y_ratio
                    text1 = texts[i][choose_index]
                    text2 = texts[i][choose_index]
                else:  # x y两者都cut
                    cut_x_ratio = random.random()
                    cut_x_ratio = cut_x_ratio if cut_x_ratio < 0.5 else 1 - cut_x_ratio
                    cut_y_ratio = random.random()
                    cut_y_ratio = cut_y_ratio if cut_y_ratio < 0.5 else 1 - cut_y_ratio
                    cut_box1 = copy.deepcopy(cell_box[choose_index])
                    cut_box2 = copy.deepcopy(cell_box[choose_index])
                    cut_box1[2] = cell_box[choose_index][0] + (1 + random.random() * cut_ratio) * width * cut_x_ratio
                    cut_box2[0] = cell_box[choose_index][0] + (1 + random.random() * cut_ratio) * width * cut_x_ratio
                    cut_box1[3] = cell_box[choose_index][1] + (1 + random.random() * cut_ratio) * height * cut_y_ratio
                    cut_box2[1] = cell_box[choose_index][1] + (1 + random.random() * cut_ratio) * height * cut_y_ratio
                    text1 = texts[i][choose_index][:int(cut_x_ratio * len(texts[i][choose_index]))]
                    text2 = texts[i][choose_index][int(cut_x_ratio * len(texts[i][choose_index])):]
                cut_box1 = refine_box(cut_box1)
                cut_box2 = refine_box(cut_box2)
                cell_boxes[i][choose_index] = cut_box1
                cell_boxes[i].append(cut_box2)
                if targets:
                    targets[i].append(targets[i][choose_index])
                texts[i][choose_index] = text1
                texts[i].append(text2)
            else:  # 做偏移
                variety_strategy = random.random()
                variety_box = cell_box[choose_index]
                if variety_strategy < 0.4:  # 左右偏移
                    variety_x0_ratio = random.random()
                    variety_box[0] = cell_box[choose_index][0] + width * variety_x0_ratio * random.choice(
                        [-variety_ratio, variety_ratio])
                    variety_x1_ratio = random.random()
                    variety_box[2] = cell_box[choose_index][2] + width * variety_x1_ratio * random.choice(
                        [-variety_ratio, variety_ratio])
                elif variety_strategy < 0.8:  # 上下偏移
                    variety_y0_ratio = random.random()
                    variety_box[1] = cell_box[choose_index][1] + height * variety_y0_ratio * random.choice(
                        [-variety_ratio, variety_ratio])
                    variety_y1_ratio = random.random()
                    variety_box[3] = cell_box[choose_index][3] + height * variety_y1_ratio * random.choice(
                        [-variety_ratio, variety_ratio])
                else:  # 缩小/扩大
                    variety_expand_ratio = random.random()
                    expand_flag = random.choice([-1, 1])
                    variety_box[
                        0] = cell_box[choose_index][0] + width * variety_expand_ratio * expand_flag * variety_ratio
                    variety_box[
                        2] = cell_box[choose_index][2] - width * variety_expand_ratio * expand_flag * variety_ratio
                    variety_box[
                        1] = cell_box[choose_index][1] + height * variety_expand_ratio * expand_flag * variety_ratio
                    variety_box[
                        3] = cell_box[choose_index][3] - height * variety_expand_ratio * expand_flag * variety_ratio
                variety_box = refine_box(variety_box)
                if variety_box[0] < img_size[i][0]:
                    variety_box[0] = img_size[i][0]
                if variety_box[1] < img_size[i][1]:
                    variety_box[1] = img_size[i][1]
                if variety_box[2] > img_size[i][2]:
                    variety_box[2] = img_size[i][2]
                if variety_box[3] > img_size[i][3]:
                    variety_box[3] = img_size[i][3]
                cell_boxes[i][choose_index] = variety_box
        if random.random() < delete_percent and len(cell_box) > 5:
            delete_num = int(random.random() * delete_ratio * len(cell_box))  # 需要删除的框数量
            for j in range(delete_num):
                delete_index = int(random.random() * len(cell_box))
                cell_boxes[i].pop(delete_index)
                texts[i].pop(delete_index)
                if targets:
                    targets[i].pop(delete_index)
    return cell_boxes, targets, texts


def refine_box(box_coord):
    if box_coord[1] > box_coord[3]:
        box_coord[1], box_coord[3] = box_coord[3], box_coord[1]
    if box_coord[0] > box_coord[2]:
        box_coord[0], box_coord[2] = box_coord[2], box_coord[0]
    return box_coord


def variety_cell_v1(cell_boxes, targets, cut_percent):
    cell_boxes = [cell_box.tolist() for cell_box in cell_boxes]
    for i, cell_box in enumerate(cell_boxes):
        choose_index_list = random.sample(list(range(len(cell_box))), int(cut_percent * len(cell_box)))
        for choose_index in choose_index_list:
            cut_strategy = random.random()
            if cut_strategy < 0.4:
                cut_x = random.randint(int(cell_box[choose_index][0]), int(cell_box[choose_index][2]))
                cut_box1 = cell_box[choose_index]
                cut_box1[2] = cut_x
                cut_box2 = cell_box[choose_index]
                cut_box2[0] = cut_x
            elif cut_strategy < 0.8:
                cut_y = random.randint(int(cell_box[choose_index][1]), int(cell_box[choose_index][3]))
                cut_box1 = cell_box[choose_index]
                cut_box1[1] = cut_y
                cut_box2 = cell_box[choose_index]
                cut_box2[3] = cut_y
            else:
                cut_x = random.randint(int(cell_box[choose_index][0]), int(cell_box[choose_index][2]))
                cut_box1 = cell_box[choose_index]
                cut_box1[2] = cut_x
                cut_box2 = cell_box[choose_index]
                cut_box2[0] = cut_x
                cut_y = random.randint(int(cell_box[choose_index][1]), int(cell_box[choose_index][3]))
                cut_box1[1] = cut_y
                cut_box2[3] = cut_y
            cell_boxes[i][choose_index] = cut_box1
            cell_boxes[i].append(cut_box2)
            targets[i].append(targets[i][choose_index])
    return cell_boxes, targets
