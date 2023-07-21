# encoding: utf-8
'''
@software: Pycharm
@time: 2020/9/14 11:37 上午
@desc:
'''
import copy
import json
import os
import re
import unittest

import cv2
import numpy as np

import pylab



class TestGenerator(unittest.TestCase):

    def setUp(self):
        self.label_list = ['Text', 'Title', 'Figure', 'Table', 'List', 'Header', 'Footer']
        self.publaynet_label_list = ['text', 'title', 'figure', 'table', 'list']
        self.doclaynet_label_list = [
            'Text', 'Title', 'Picture', 'Table', 'List-item', 'Page-header', 'Page-footer', 'Section-header',
            'Footnote', 'Caption', 'Formula'
        ]
        self.class_color_list = [[0, 0, 255], [0, 255, 0], [255, 0, 255], [0, 255, 255], [0, 0, 0], [255, 0, 0],
                                 [255, 255, 0], [0, 0, 120], [0, 120, 0], [120, 0, 0], [120, 120, 0], [0, 120, 120],
                                 [120, 0, 120]]
        self.annType_id = 1
        self.image_path = ""
        self.gt_path = ""
        self.pred_path = ''

    def test_eval_coco(self):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        pylab.rcParams['figure.figsize'] = (10.0, 8.0)
        annType = ['segm', 'bbox', 'keypoints']
        annType = annType[self.annType_id]  #specify type here
        cocoGt = COCO(self.gt_path)
        annotation = json.load(open(self.pred_path))['annotations']
        cocoDt = cocoGt.loadRes(annotation)
        imgIds = sorted(cocoGt.getImgIds())
        # running evaluation
        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        categories = cocoGt.getCatIds()
        for i, cat in enumerate(categories):
            cocoEval.params.catIds = [i]
            print("\n*****************category:{} result: ************************".format(cat))
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

