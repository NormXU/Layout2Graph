#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:weishu
# datetime:2022/11/1 3:27 下午
# software: PyCharm
import copy
import os
from collections import Counter

import numpy as np
import torch
import networkx
import cv2

class_color_list = [[0, 0, 255], [0, 255, 0], [255, 0, 255], [0, 255, 255], [0, 0, 0], [255, 0, 0], [255, 255, 0],
                    [0, 0, 120], [0, 120, 0], [120, 0, 0], [120, 120, 0], [0, 120, 120], [120, 0, 120]]


def visualize_img(img, transform_image, cell_box, transform_cell_box, result_data, save_dir, img_file_name, text=None):
    color_list = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                  for i in range(len(transform_cell_box))]
    file_name = '.'.join(img_file_name.split('.')[:-1])
    origin_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    # for i, box in enumerate(cell_box):
    #     width = int(box[2] - box[0])
    #     height = int(box[3] - box[1])
    #     lu_points = (int(box[0] + width // 3), int(box[1]) + height // 3)
    #     rd_points = (int(box[2]) - width // 3, int(box[3]) - height // 3)
    #     cv2.rectangle(origin_img, lu_points, rd_points, color_list[i], -1)
    # cv2.imwrite(os.path.join(save_dir, file_name + '_origin.jpg'), origin_img)
    #
    # transform_image = (transform_image.detach().cpu().numpy() + 1) * 255 / 2
    # debug_img = cv2.UMat(transform_image.transpose(1, 2, 0).astype(np.uint8))
    # for i, box in enumerate(transform_cell_box):
    #     width = int(box[2] - box[0])
    #     height = int(box[3] - box[1])
    #     lu_points = (int(box[0] + width // 3), int(box[1]) + height // 3)
    #     rd_points = (int(box[2]) - width // 3, int(box[3]) - height // 3)
    #     debug_img = cv2.rectangle(debug_img, lu_points, rd_points, color_list[i], -1)
    # cv2.imwrite(os.path.join(save_dir, file_name + '_debug.jpg'), debug_img)

    if 'node_score_list' in result_data:
        _generate_layout_graph(result_data, cell_box, transform_cell_box, origin_img, transform_image, color_list,
                               save_dir, file_name, text)


def _generate_layout_graph(result_data, cell_box, transform_cell_box, origin_img, transform_image, color_list, save_dir,
                           file_name, text):
    node_pred = result_data['node_pred']

    cls_debug_img = copy.deepcopy(origin_img)

    for i, box in enumerate(cell_box):
        width = int(box[2] - box[0])
        height = int(box[3] - box[1])
        lu_points = (int(box[0] + width // 3), int(box[1]) + height // 3)
        rd_points = (int(box[2]) - width // 3, int(box[3]) - height // 3)
        cv2.rectangle(cls_debug_img, lu_points, rd_points, class_color_list[node_pred[i]], -1)
        # lu_points = (int(box[0]), int(box[1]))
        # rd_points = (int(box[2]) , int(box[3]))
        # cv2.rectangle(cls_debug_img, lu_points, rd_points, class_color_list[node_pred[i]], 1)
    # cv2.imwrite(os.path.join(save_dir, file_name + '_cls.jpg'), cls_debug_img)

    pair_cell = result_data['pair_cell']
    if len(pair_cell) > 0:
        pair_cell_img = copy.deepcopy(cls_debug_img)
        pair_cell_pred = list(map(int, result_data['pair_cell_pred'].cpu().numpy().tolist()))
        all_cell_pairs = [pair for i, pair in enumerate(pair_cell) if pair_cell_pred[i] == 1]
        pair_cell_score_list = torch.nn.functional.softmax(result_data['pair_cell_score_list'], dim=1)
        all_cell_pairs_score = [
            round(float(pair_cell_score_list[i][1].cpu()), 2)
            for i, pair in enumerate(pair_cell)
            if pair_cell_pred[i] == 1
        ]
        for i, pair in enumerate(all_cell_pairs):
            start_center = (int(cell_box[pair[0]][0] + cell_box[pair[0]][2]) // 2,
                            int(cell_box[pair[0]][1] + cell_box[pair[0]][3]) // 2)
            end_center = (int(cell_box[pair[1]][0] + cell_box[pair[1]][2]) // 2,
                          int(cell_box[pair[1]][1] + cell_box[pair[1]][3]) // 2)
            cv2.line(pair_cell_img, start_center, end_center, (0, 0, 0), 3)
            cv2.putText(pair_cell_img, str(all_cell_pairs_score[i]),
                        ((start_center[0] + end_center[0]) // 2, (start_center[1] + end_center[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(save_dir, file_name + '_pair.jpg'), pair_cell_img)

        unpair_cell_img = copy.deepcopy(cls_debug_img)
        all_cell_unpairs = [pair for i, pair in enumerate(pair_cell) if pair_cell_pred[i] == 0]
        for pair in all_cell_unpairs:
            start_center = (int(cell_box[pair[0]][0] + cell_box[pair[0]][2]) // 2,
                            int(cell_box[pair[0]][1] + cell_box[pair[0]][3]) // 2)
            end_center = (int(cell_box[pair[1]][0] + cell_box[pair[1]][2]) // 2,
                          int(cell_box[pair[1]][1] + cell_box[pair[1]][3]) // 2)
            cv2.line(unpair_cell_img, start_center, end_center, (0, 0, 0), 3)
        cv2.imwrite(os.path.join(save_dir, file_name + '_unpair.jpg'), unpair_cell_img)

    pair_graph_img = copy.deepcopy(origin_img)
    for od_label in result_data['od_label_list']:
        color = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
        points = od_label['points']
        label = od_label['label']
        for node in od_label['nodeSet']:
            box = cell_box[node]
            width = int(box[2] - box[0])
            height = int(box[3] - box[1])
            lu_points = (int(box[0] + width // 3), int(box[1]) + height // 3)
            rd_points = (int(box[2]) - width // 3, int(box[3]) - height // 3)
            cv2.rectangle(pair_graph_img, lu_points, rd_points, color, -1)
        cv2.rectangle(pair_graph_img, (int(points[0]), int(points[1])), (int(points[2]), int(points[3])),
                      class_color_list[label], 2)
    cv2.imwrite(os.path.join(save_dir, file_name + '_pair_graph.jpg'), pair_graph_img)