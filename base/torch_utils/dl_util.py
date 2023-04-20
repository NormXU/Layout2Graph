# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/6/17

import json
import math
import torch
import random
import numpy as np
from torch import nn
import torch.optim as optim
from base.driver import logger
from collections import OrderedDict
from torch.optim import lr_scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from transformers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_linear_schedule_with_warmup
from base.torch_utils.scheduler_util import LinearLRScheduler, get_cosine_schedule_by_epochs, \
    get_stairs_schedule_with_warmup

def seed_all(random_seed):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

def print_network(net, verbose=False, name=""):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        logger.info(net)
    if hasattr(net, 'flops'):
        flops = net.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    logger.info('network:{} Total number of parameters: {}'.format(name, num_params))


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm


def set_params_optimizer(model, keyword=None, keywords=None, weight_decay=0.0, lr=None):
    if keywords is None:
        keywords = []
    param_dict = OrderedDict()
    no_decay_param_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if keyword in name or check_keywords_in_name(name, keywords):
            param_dict[name] = {"weight_decay": weight_decay}
            if lr is not None:
                lr = float(lr)
                param_dict[name].update({"lr": lr})
        else:
            no_decay_param_names.append(name)
    return param_dict, no_decay_param_names


def get_optimizer_yolo(model, optimizer_type="adam", lr=0.001, weight_decay=0.0, momentum=0, **kwargs):
    lr = float(lr)
    weight_decay = float(weight_decay)
    momentum = float(momentum)

    g = [], [], []  # optimizer parameter groups
    # bn = nn.BatchNorm2d, nn.LazyBatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.LazyInstanceNorm2d, nn.LayerNorm
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)
    #yolov7
    # pg = [], [], []  # optimizer parameter groups
    # for k, v in model.named_modules():
    #     if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
    #         pg[1].append(v.bias)  # biases
    #     if isinstance(v, nn.BatchNorm2d):
    #         pg[2].append(v.weight)  # no decay
    #     elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
    #         pg[0].append(v.weight)  # apply decay
    #     if hasattr(v, 'im'):
    #         if hasattr(v.im, 'implicit'):
    #             pg[2].append(v.im.implicit)
    #         else:
    #             for iv in v.im:
    #                 pg[2].append(iv.implicit)
    #     if hasattr(v, 'imc'):
    #         if hasattr(v.imc, 'implicit'):
    #             pg[2].append(v.imc.implicit)
    #         else:
    #             for iv in v.imc:
    #                 pg[2].append(iv.implicit)
    #     if hasattr(v, 'imb'):
    #         if hasattr(v.imb, 'implicit'):
    #             pg[2].append(v.imb.implicit)
    #         else:
    #             for iv in v.imb:
    #                 pg[2].append(iv.implicit)
    #     if hasattr(v, 'imo'):
    #         if hasattr(v.imo, 'implicit'):
    #             pg[2].append(v.imo.implicit)
    #         else:
    #             for iv in v.imo:
    #                 pg[2].append(iv.implicit)
    #     if hasattr(v, 'ia'):
    #         if hasattr(v.ia, 'implicit'):
    #             pg[2].append(v.ia.implicit)
    #         else:
    #             for iv in v.ia:
    #                 pg[2].append(iv.implicit)
    #     if hasattr(v, 'attn'):
    #         if hasattr(v.attn, 'logit_scale'):
    #             pg[2].append(v.attn.logit_scale)
    #         if hasattr(v.attn, 'q_bias'):
    #             pg[2].append(v.attn.q_bias)
    #         if hasattr(v.attn, 'v_bias'):
    #             pg[2].append(v.attn.v_bias)
    #         if hasattr(v.attn, 'relative_position_bias_table'):
    #             pg[2].append(v.attn.relative_position_bias_table)
    #     if hasattr(v, 'rbr_dense'):
    #         if hasattr(v.rbr_dense, 'weight_rbr_origin'):
    #             pg[2].append(v.rbr_dense.weight_rbr_origin)
    #         if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'):
    #             pg[2].append(v.rbr_dense.weight_rbr_avg_conv)
    #         if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):
    #             pg[2].append(v.rbr_dense.weight_rbr_pfir_conv)
    #         if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
    #             pg[2].append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
    #         if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
    #             pg[2].append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
    #         if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):
    #             pg[2].append(v.rbr_dense.weight_rbr_gconv_dw)
    #         if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):
    #             pg[2].append(v.rbr_dense.weight_rbr_gconv_pw)
    #         if hasattr(v.rbr_dense, 'vector'):
    #             pg[2].append(v.rbr_dense.vector)
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {optimizer_type} not implemented.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)
    logger.info(f"{'optimizer:'} {type(optimizer).__name__} with parameter groups "
                f"{len(g[1])} weight (no decay), {len(g[0])} weight, {len(g[2])} bias")
    del g
    return optimizer


def get_optimizer(model,
                  optimizer_type="adam",
                  lr=0.001,
                  beta1=0.9,
                  beta2=0.999,
                  no_decay_keys=None,
                  weight_decay=0.0,
                  layer_decay=None,
                  eps=1e-8,
                  momentum=0,
                  params=None,
                  **kwargs):
    assigner = None
    if layer_decay is not None:
        if layer_decay < 1.0:
            num_layers = kwargs.get('num_layers')
            assigner = LayerDecayValueAssigner(
                list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))

    lr = float(lr)
    beta1, beta2 = float(beta1), float(beta2)
    weight_decay = float(weight_decay)
    momentum = float(momentum)
    eps = float(eps)
    freeze_params = kwargs.get('freeze_params', [])
    for name, param in model.named_parameters():
        freeze_flag = False
        for freeze_param in freeze_params:
            if freeze_param in name:
                freeze_flag = True
                break
        if freeze_flag:
            print("name={} param.requires_grad = False".format(name))
            param.requires_grad = False

    if params is None:
        if weight_decay:
            skip = {}
            if no_decay_keys is not None:
                skip = no_decay_keys
            elif hasattr(model, 'no_weight_decay'):
                skip = model.no_weight_decay()
            param_configs = get_parameter_groups(model, weight_decay, skip, assigner)
            weight_decay = 0.
        else:
            param_configs = model.parameters()
    else:
        param_configs = params
    if optimizer_type == "sgd":
        optimizer = optim.SGD(param_configs, momentum=momentum, nesterov=True, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        optimizer = optim.Adam(param_configs, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    elif optimizer_type == "adadelta":
        optimizer = optim.Adadelta(param_configs, lr=lr, eps=eps, weight_decay=weight_decay)
    elif optimizer_type == "rmsprob":
        optimizer = optim.RMSprop(param_configs, lr=lr, eps=eps, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(param_configs, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', optimizer_type)
    return optimizer


def get_scheduler(optimizer,
                  scheduler_type="linear",
                  num_warmup_steps=0,
                  num_training_steps=10000,
                  last_epoch=-1,
                  step_size=10,
                  gamma=0.1,
                  epochs=20,
                  **kwargs):
    gamma = float(gamma)
    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps,
                                                    last_epoch=last_epoch)
    elif scheduler_type == 'cosine_epoch':
        scheduler = get_cosine_schedule_by_epochs(optimizer, num_epochs=epochs, last_epoch=last_epoch)
    elif scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps,
                                                    last_epoch=last_epoch)
    elif scheduler_type == "stairs":
        logger.info("current use stair scheduler")
        scheduler = get_stairs_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps,
                                                    last_epoch=last_epoch,
                                                    **kwargs)
    elif scheduler_type == "step":
        step_size = int(step_size)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
        """
        def exp_decay(epoch):
           initial_lrate = 0.1
           k = 0.1
           lrate = initial_lrate * exp(-k*t)
           return lrate
        """

    else:
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=num_warmup_steps,
                                                      last_epoch=last_epoch)
    return scheduler


def get_scheduler2(optimizer,
                   scheduler_type="cosine",
                   num_warmup_steps=0,
                   num_training_steps=10000,
                   decay_steps=1000,
                   decay_rate=0.1,
                   lr_min=5e-6,
                   warmup_lr=5e-7):
    lr_min = float(lr_min)
    warmup_lr = float(warmup_lr)
    decay_rate = float(decay_rate)
    if scheduler_type == "cosine":
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=num_training_steps,
                                      t_mul=1,
                                      lr_min=lr_min,
                                      warmup_lr_init=warmup_lr,
                                      cycle_limit=1,
                                      t_in_epochs=False)
    elif scheduler_type == "linear":
        scheduler = LinearLRScheduler(optimizer,
                                      t_initial=num_training_steps,
                                      lr_min_rate=0.01,
                                      warmup_lr_init=warmup_lr,
                                      warmup_t=num_warmup_steps,
                                      t_in_epochs=False)
    else:
        scheduler = StepLRScheduler(optimizer,
                                    decay_t=decay_steps,
                                    decay_rate=decay_rate,
                                    warmup_lr_init=warmup_lr,
                                    warmup_t=num_warmup_steps,
                                    t_in_epochs=False)
    return scheduler


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def get_scheduler_yolo(optimizer, cos_lr=True, lrf=0.1, epochs=20, last_epoch=-1, **kwargs):
    if cos_lr:
        lf = one_cycle(1, lrf, epochs)  # cosine 1->lrf
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf,
                                      last_epoch=last_epoch)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    return scheduler, lf


def get_tensorboard_texts(label_texts):
    new_labels = []
    for label_text in label_texts:
        new_labels.append(label_text.replace("/", "//").replace("<", "/<").replace(">", "/>"))
    return "  \n".join(new_labels)


def get_parameter_groups(model, weight_decay, skip_list=(), assigner=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name.split('.')[-1] in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if assigner is not None:
            layer_id = assigner.get_layer_id(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if assigner is not None:
                scale = assigner.get_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer(var_name, len(self.values))


def get_num_layer(var_name, num_max_layer):
    var_name = var_name.split('.', 1)[-1]
    if var_name.startswith("embeddings"):
        return 0
    elif var_name.startswith("encoder.layer"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1
