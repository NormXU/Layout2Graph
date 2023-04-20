#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:weishu
# datetime:2022/11/1 2:42 下午
# software: PyCharm
# encoding: utf-8
import copy
import json
import os
from PIL import Image, ImageOps
import numpy as np
import time
import torch
from tqdm import tqdm

from base.branch_utils.gragh_net_utils.label_converter import TableGraphLabelConverter
from base.branch_utils.gragh_net_utils.visualize_table import visualize_img
from base.common_util import get_file_path_list
from base.driver import logger
from experiment.base_experiment import BaseExperiment
from metrics.meter import AverageMeter
from mydatasets.gragh_net.graph_collate import graph_get_pad_transform, variety_cell, graph_get_resize_transform
from post_process import get_post_processor


class GraphLayoutExperiment(BaseExperiment):

    def __init__(self, config):
        super(GraphLayoutExperiment, self).__init__(config)
        self.args.trainer.best_eval_result = [-1, -1]

    def predict(self, **kwargs):
        self.model.eval()
        with torch.no_grad():
            self.args.predictor.img_label_paths = [] if self.args.predictor.img_label_paths is None else self.args.predictor.img_label_paths
            if self.args.predictor.img_label_dirs is not None:
                for i, img_label_dir in enumerate(self.args.predictor.img_label_dirs):
                    img_path_list = get_file_path_list(self.args.predictor.img_input_dirs[i], ['png'])
                    self.args.predictor.img_paths.extend(img_path_list)
                    self.args.predictor.img_label_paths.extend([
                        label_path.replace(self.args.predictor.img_input_dirs[i], img_label_dir).split('_')[0]+'.json' for label_path in
                        img_path_list
                    ])
            image_count, instance_count = 0, 0
            if self.args.predictor.save_coco_result:
                coco_path = os.path.join(self.args.predictor.save_dir, 'coco_pred.json')
                if os.path.exists(coco_path):
                    with open(coco_path) as f:
                        coco_data = json.load(f)
                    image_count = len(coco_data['images'])
                    instance_count = len(coco_data['annotations'])
                else:
                    coco_data = {
                        "images": [],
                        "annotations": [],
                        "categories": [{
                            'id': i,
                            'name': label
                        } for i, label in enumerate(self.args.model.class_list)]
                    }
            for i, img_path in enumerate(tqdm(self.args.predictor.img_paths[image_count:])):
                if not os.path.exists(img_path):
                    img_path = img_path.replace('/ocr_results_images/', '/images/')
                if not os.path.exists(img_path):
                    continue
                logger.info(img_path)
                image = Image.open(img_path).convert("RGB")
                with open(self.args.predictor.img_label_paths[i], 'r') as f:
                    json_data = json.load(f)
                    cell_box, text = [], []
                    for item in json_data['cells']:
                        # TODO 可以过滤一些文本框不去预测
                        if len(item['bbox']) > 0 and item['bbox'][2] > 0 and \
                                abs(item['bbox'][3]) > 0:
                            cell_box.append([item['bbox'][0], item['bbox'][1]+item['bbox'][3], item['bbox'][0]+item['bbox'][2], item['bbox'][1]])
                            text.append(item['text'])
                if len(cell_box) == 0:
                    logger.warning('no cell in {}'.format(img_path))
                    continue
                if self.args.predictor.pad_flag:
                    transform_image, transform_cell_box, padding_edge = graph_get_pad_transform(
                        image, copy.deepcopy(cell_box), self.args.predictor.height, self.args.predictor.width)
                else:
                    transform_image, transform_cell_box, padding_edge = graph_get_resize_transform(
                        image, copy.deepcopy(cell_box), self.args.predictor.height, self.args.predictor.width)
                transform_images = torch.stack([transform_image], 0).to(self.args.device.device_id)
                if self.args.predictor.aug_flag:
                    transform_cell_boxes, _, transform_texts = variety_cell(
                        [transform_image],
                        [transform_cell_box],
                        None,
                        [text],
                        [padding_edge],
                        self.args.predictor.cut_percent,
                        self.args.predictor.variety_percent,
                        self.args.predictor.deleted_percent,
                        self.args.predictor.cut_ratio,
                        self.args.predictor.variety_ratio,
                        self.args.predictor.deleted_ratio,
                    )
                    transform_cell_box, text = transform_cell_boxes[0], transform_texts[0]
                transform_cell_boxes = [
                    torch.from_numpy(np.array(transform_cell_box)).to(torch.float32).to(self.args.device.device_id)
                ]
                text_tensor = self.label_converter.encode([text]) if self.label_converter else None
                result_data = self.model(transform_images, transform_cell_boxes, texts=text_tensor)
                predict_post_processor = get_post_processor(self.args.predictor.post_processor)
                result_data = predict_post_processor(result_data, cell_box, transform_cell_box, text)
                if self.args.predictor.do_visualize:
                    visualize_img(image, transform_image, cell_box, transform_cell_box, result_data,
                                  self.args.predictor.save_dir, os.path.basename(img_path), text)
                if self.args.predictor.save_coco_result:
                    coco_data["images"].append({
                        "file_name": img_path.split('/')[-1],
                        "height": image.height,
                        "width": image.width,
                        "id": image_count,
                    })
                    image_count += 1
                    for od_label in result_data['od_label_list']:
                        x1, y1, x2, y2 = od_label['points'][0], od_label['points'][1], od_label['points'][
                            2], od_label['points'][3]
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        segmentation = [x1, y1, x2, y1, x2, y2, x1, y2]
                        coco_data["annotations"].append({
                            "segmentation": [segmentation],
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                            "image_id": i,
                            "bbox": bbox,
                            "category_id": od_label['label'],
                            "id": instance_count,
                            "score": od_label['node_score'],
                        })
                        instance_count += 1
            if self.args.predictor.save_coco_result:
                with open(coco_path, "w", encoding='utf-8') as f:
                    json.dump(coco_data, f, ensure_ascii=False, indent=2)

    def evaluate(self, **kwargs):
        global_eval_step = kwargs.get('global_eval_step', 0)
        eval_model = self.model
        if eval_model.training and self.ema:
            self.ema.update_attr(self.model, include=self.args.trainer.ema_include)
            eval_model = self.ema.ema
        if self.use_torch_amp:
            eval_model.half()
        eval_model.eval()
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        eval_metric = self._init_metric()
        simple_eval_flag = kwargs.get('simple_eval_flag', False)
        if simple_eval_flag:
            logger.info("run evaluation on a small dataset before start training")
        for i, batch in enumerate(self.eval_data_loader):
            if simple_eval_flag and i > self.args.trainer.eval_print_freq:
                break
            start = time.time()
            batch_size = self.args.datasets.train.batch_size
            result = self._step_forward(batch, is_train=False, eval_model=eval_model)
            loss_meter.update(result['loss'].item(), batch_size)
            norm_meter.update(0)
            batch_time.update(time.time() - start)
            eval_metric.add_label(result['outputs']['pair_cell_pred'], result['outputs']['pair_cell_target'],
                                  result['outputs']['node_pred'], result['cls_targets'])
            global_eval_step += 1
            if self.args.device.is_master and self.args.trainer.eval_print_freq > 0 and global_eval_step % self.args.trainer.eval_print_freq == 0:
                self._print_eval_log(global_eval_step, loss_meter, eval_metric)
            self._display_images(batch, result, global_step=global_eval_step, if_train=False)
        acc = self._print_eval_log(global_eval_step, loss_meter, eval_metric)
        eval_model.float()
        return {'acc': acc, 'global_eval_step': global_eval_step}

    def _step_forward(self, batch, is_train=True, eval_model=None, **kwargs):
        model = eval_model if is_train is False and eval_model is not None else self.model
        images = batch["images"]
        targets = batch["targets"]
        cell_boxes = batch["cell_boxes"]
        linkings = batch.get("linkings")
        cell_boxes = [cell_box.to(self.args.device.device_id) for cell_box in cell_boxes]
        images = images.to(self.args.device.device_id)
        encode_texts = batch.get("encode_texts", None)
        if self.label_converter:
            texts = batch.get("texts", None)
            encode_texts = self.label_converter.encode(texts)
        if is_train:
            with self.precision_scope:
                outputs = model(images, cell_boxes, targets, encode_texts, linkings)
        else:
            with self.precision_scope:
                outputs = model(images, cell_boxes, targets, encode_texts, linkings)
                # predictions, references = accelerator.gather_for_metrics((predictions, batch["label"]))
        cls_targets = []
        for target_list in targets:
            for target in target_list:
                label = target.split('_')[0]
                label = self.args.model.class_map.get(label, label)
                label_index = self.args.model.class_list.index(label)
                cls_targets.append(label_index)
        cls_targets = torch.from_numpy(np.array(cls_targets)).to(self.args.device.device_id)
        losses = self.criterion(outputs, cls_targets)
        return {'loss': losses['loss'], 'outputs': outputs, 'cls_targets': cls_targets}

    # config的联动关系可以写在这个函数中
    def _init_config(self, config):
        config['model']['num_classes'] = len(config['model']['class_list'])
        if 'datasets' in config:
            if 'encode_text_type' in config['model']:
                config['datasets']['train']['dataset']['encode_text_type'] = config['model']['encode_text_type']
            config['datasets']['train']['collate_fn']['width'] = config['model']['width']
            config['datasets']['eval']['collate_fn']['width'] = config['model']['width']
            config['datasets']['train']['collate_fn']['height'] = config['model']['height']
            config['datasets']['eval']['collate_fn']['height'] = config['model']['height']
        if 'predictor' in config:
            config['predictor']['width'] = config['model']['width']
            config['predictor']['height'] = config['model']['height']
        if 'trainer' in config:
            config['trainer']['metric']['num_classes'] = config['model']['num_classes']
            if config['trainer']['loss']['class_weight_flag']:
                config['trainer']['loss']['num_classes'] = config['model']['num_classes']
        return super()._init_config(config)

    def init_model(self, config):
        model_args = config["model"]
        vocab_path = model_args.get("vocab_path", None)
        self.label_converter = None
        if vocab_path:
            with open(vocab_path, 'r') as f:
                charsets = f.read().strip('\n')
            self.label_converter = TableGraphLabelConverter(alphabet=charsets)
            config['model']['vocab_size'] = len(self.label_converter.alphabet)
        super().init_model(config)

    def load_model(self, checkpoint_path, strict=True, **kwargs):
        if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            try:
                self.model.load_state_dict(state_dict, strict=strict)
            except:
                origin_num_classes = len(state_dict['cls_node.0.weight'])
                state_dict['cls_node.0.weight'] = state_dict['cls_node.0.weight'][:self.model.num_classes]
                state_dict['cls_node.0.bias'] = state_dict['cls_node.0.bias'][:self.model.num_classes]

                state_dict['linear_cell.0.weight'] = torch.cat([
                    state_dict['linear_cell.0.weight'][:, :-origin_num_classes * 2],
                    state_dict['linear_cell.0.weight'][:,
                    -origin_num_classes * 2:-origin_num_classes * 2 + self.model.num_classes], \
                    state_dict['linear_cell.0.weight'][:,
                    -origin_num_classes:-origin_num_classes + self.model.num_classes]], \
                    dim=1)
                logger.warning('load model: class num dont match:{} vs {}!!!'.format(
                    origin_num_classes, self.model.num_classes))
                self.model.load_state_dict(state_dict, strict=strict)
            logger.info("success load model:{}".format(checkpoint_path))

    def _print_eval_log(self, global_step, loss_meter, eval_metric, **kwargs):
        evaluate_report = eval_metric.get_report()
        Node_F1_MICRO = evaluate_report["Node_F1_MICRO"]
        Pair_F1_MACRO = evaluate_report["Pair_F1_MACRO"]
        message = "experiment:{}; eval,global_step:{}, (step_mean_loss:{},average_loss:{:.5f}, evaluate_report:\n{})".format(
            self.experiment_name, global_step, loss_meter.val, loss_meter.avg, evaluate_report)
        logger.info(message)
        if self.writer is not None:
            self.writer.add_scalar("{}_eval/step_loss".format(self.experiment_name), loss_meter.val, global_step)
            self.writer.add_scalar("{}_eval/average_loss".format(self.experiment_name), loss_meter.avg, global_step)
            self.writer.add_scalar("{}_eval/F1_MICRO_cls".format(self.experiment_name), Node_F1_MICRO, global_step)
            self.writer.add_scalar("{}_eval/F1_MACRO_cell".format(self.experiment_name), Pair_F1_MACRO, global_step)
        return [Node_F1_MICRO, Pair_F1_MACRO]

    def _print_step_log(self, epoch, global_step, global_eval_step, loss_meter, norm_meter, batch_time, ni, **kwargs):
        current_lr = self._get_current_lr(ni, global_step)
        if self.args.device.is_master and self.args.trainer.print_freq > 0 and global_step % self.args.trainer.print_freq == 0:
            message = "experiment:{}; train, (epoch: {}, steps: {}, lr:{:e}, step_mean_loss:{}," \
                      " average_loss:{}), time, (train_step_time: {:.5f}s, train_average_time: {:.5f}s);" \
                      "(grad_norm_mean: {:.5f}, grad_norm_step: {:.5f})". \
                format(self.experiment_name, epoch, global_step, current_lr,
                       loss_meter.val, loss_meter.avg, batch_time.val, batch_time.avg, norm_meter.avg,
                       norm_meter.val)
            logger.info(message)
            if self.writer is not None:
                self.writer.add_scalar("{}_train/lr".format(self.experiment_name), current_lr, global_step)
                self.writer.add_scalar("{}_train/step_loss".format(self.experiment_name), loss_meter.val, global_step)
                self.writer.add_scalar("{}_train/average_loss".format(self.experiment_name), loss_meter.avg,
                                       global_step)
        if global_step > 0 and self.args.trainer.save_step_freq > 0 and global_step % self.args.trainer.save_step_freq == 0:
            message = "experiment:{}; eval, (epoch: {}, steps: {});".format(self.experiment_name, epoch, global_step)
            logger.info(message)
            result = self.evaluate(global_eval_step=global_eval_step)
            global_eval_step = result['global_eval_step']
            Node_F1_MICRO, Pair_F1_MACRO = result['acc']
            if (not self.args.trainer.save_best or (self.args.trainer.save_best
                                                   and Pair_F1_MACRO > self.args.trainer.best_eval_result)) and self.args.device.is_master:
                checkpoint_name = "{}_epoch{}_step{}_lr{:e}_average_loss{:.5f}_NodeF1MICRO{:.5f}_PairF1MACRO{:.5f}.pth".format(
                    self.experiment_name, epoch, global_step, current_lr, loss_meter.avg, Node_F1_MICRO, Pair_F1_MACRO)
                checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
                self.save_model(checkpoint_path, epoch=epoch, global_step=global_step)
                if Node_F1_MICRO > self.args.trainer.best_eval_result[0]:
                    self.args.trainer.best_eval_result[0] = Node_F1_MICRO
                    self.args.trainer.best_model_path = checkpoint_path
                if Pair_F1_MACRO > self.args.trainer.best_eval_result[1]:
                    self.args.trainer.best_eval_result[1] = Pair_F1_MACRO
                message = "best_eval_result:{};".format(self.args.trainer.best_eval_result)
                logger.info(message)
        return global_eval_step

    def _print_epoch_log(self, epoch, global_step, global_eval_step, loss_meter, ni, **kwargs):
        current_lr = self._get_current_lr(ni, global_step)
        if self.args.trainer.save_epoch_freq > 0 and epoch % self.args.trainer.save_epoch_freq == 0:
            message = "experiment:{}; eval, (epoch: {}, steps: {});".format(self.experiment_name, epoch, global_step)
            logger.info(message)
            result = self.evaluate(global_eval_step=global_eval_step)
            global_eval_step = result['global_eval_step']
            Node_F1_MICRO, Pair_F1_MACRO = result['acc']
            if (not self.args.trainer.save_best or (self.args.trainer.save_best
                                                   and Pair_F1_MACRO > self.args.trainer.best_eval_result)) and self.args.device.is_master:
                checkpoint_name = "{}_epoch{}_step{}_lr{:e}_average_loss{:.5f}_NodeF1MICRO{:.5f}_PairF1MACRO{:.5f}.pth".format(
                    self.experiment_name, epoch, global_step, current_lr, loss_meter.avg, Node_F1_MICRO, Pair_F1_MACRO)
                checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
                self.save_model(checkpoint_path, epoch=epoch, global_step=global_step)
                if Node_F1_MICRO > self.args.trainer.best_eval_result[0]:
                    self.args.trainer.best_eval_result[0] = Node_F1_MICRO
                    self.args.trainer.best_model_path = checkpoint_path
                if Pair_F1_MACRO > self.args.trainer.best_eval_result[1]:
                    self.args.trainer.best_eval_result[1] = Pair_F1_MACRO
                message = "best_eval_result:{};".format(self.args.trainer.best_eval_result)
                logger.info(message)
        return global_eval_step
