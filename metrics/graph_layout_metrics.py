# -*- coding:utf-8 -*-
# email:weishu@datagrand.com
# create: 2021/12/1
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve


class GraphLayoutMetric(object):

    def __init__(self, num_classes, **kwargs):
        self.pred_labels_cls, self.gt_labels_cls = [], []
        self.pred_labels_cell, self.gt_labels_cell = [], []
        self.num_classes = num_classes

    def add_label(self, pred_label_cell, gt_label_cell, pred_label_cls, gt_label_cls):
        self.pred_labels_cls.extend(list(map(int, pred_label_cls.cpu().numpy().tolist())))
        self.pred_labels_cell.extend(list(map(int, pred_label_cell.cpu().numpy().tolist())))
        self.gt_labels_cls.extend(list(map(int, gt_label_cls.cpu().numpy().tolist())))
        self.gt_labels_cell.extend(list(map(int, gt_label_cell.cpu().numpy().tolist())))

    def get_report(self):
        # evaluation doclaynet model for publaynet
        # update_label = []
        # for label in self.pred_labels_cls:
        #     if label in [5, 6, 8, 10]:
        #         update_label.append(0)
        #     elif label in [7, 9]:
        #         update_label.append(1)
        #     else:
        #         update_label.append(label)
        # self.pred_labels_cls = update_label
        # -----------------------------
        # evaluation doclaynet model for dg_data
        # update_label = []
        # for label in self.pred_labels_cls:
        #     if label in [8, 10]:
        #         update_label.append(0)
        #     elif label in [7, 9]:
        #         update_label.append(1)
        #     else:
        #         update_label.append(label)
        # self.pred_labels_cls = update_label
        # --------------------------------
        # evaluation dg_data model for publaynet
        # update_label = []
        # for label in self.pred_labels_cls:
        #     if label in [5, 6]:
        #         update_label.append(0)
        #     else:
        #         update_label.append(label)
        # self.pred_labels_cls = update_label
        sums = len(self.pred_labels_cls)
        correct = 0
        correct_map = {}
        for class_num in range(self.num_classes):
            class_num_label = np.array(self.pred_labels_cls)[np.where(np.array(self.gt_labels_cls) == class_num)]
            total_num = len(class_num_label)
            if total_num != 0:
                correct_num = len(np.where(class_num_label == class_num)[0])
                correct_map[class_num] = (correct_num, total_num, correct_num / total_num)
                correct += correct_num
        F1_MACRO, F1_MICRO = f1_score(self.gt_labels_cls, self.pred_labels_cls,
                                      average='macro'), f1_score(self.gt_labels_cls,
                                                                 self.pred_labels_cls,
                                                                 average='micro')
        report = {
            "accuracy_cls": sum([item[2] for key, item in correct_map.items()]) / len(correct_map),
            'Node_F1_MACRO': F1_MACRO,
            'Node_F1_MICRO': F1_MICRO,
            "sums_cls": sums,
            "correct_cls": correct,
            'accuracy_map_cls': correct_map
        }

        sums = len(self.pred_labels_cell)
        correct = sums - len(np.where(np.array(self.gt_labels_cell) + np.array(self.pred_labels_cell) == 1)[0])
        correct_map = {}
        for class_num in range(2):
            class_num_label = np.array(self.pred_labels_cell)[np.where(np.array(self.gt_labels_cell) == class_num)]
            total_num = len(class_num_label)
            if total_num != 0:
                correct_num = len(np.where(class_num_label == class_num)[0])
                correct_map[class_num] = (correct_num, total_num, correct_num / total_num)
        F1_MACRO, F1_MICRO = f1_score(self.gt_labels_cell, self.pred_labels_cell,
                                      average='macro'), f1_score(self.gt_labels_cell,
                                                                 self.pred_labels_cell,
                                                                 average='micro')
        report.update({
            "accuracy_cell": sum([item[2] for key, item in correct_map.items()]) / len(correct_map) if len(correct_map)> 0 else 0.0,
            'Pair_F1_MACRO': F1_MACRO,
            'Pair_F1_MICRO': F1_MICRO,
            "sums_cell": sums,
            "correct_cell": correct,
            'accuracy_map_cell': correct_map
        })
        return report
