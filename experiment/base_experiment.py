# -*- coding:utf-8 -*-
# create: 2021/7/2
import copy
import time
import munch
import logging
import json
import os
import torch
import itertools

import mydatasets
from torch import autocast
from base.driver import logger
from base.torch_utils.torch_util import ModelEMA
from loss import get_criterion
from metrics import get_metric
from mydatasets import get_dataset
from networks import get_network
from accelerate import Accelerator
from contextlib import contextmanager, nullcontext
from torch.utils.data import DataLoader
from base.common_util import save_params

from metrics.meter import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from base.driver import log_formatter, PROJECT_ROOT_PATH
from base.torch_utils.dl_util import get_optimizer, get_scheduler, get_scheduler2, seed_all, get_grad_norm
from base.common_util import get_absolute_file_path, merge_config, get_file_path_list


class BaseExperiment(object):

    def __init__(self, config):
        config = self._init_config(config)
        self.experiment_name = config["name"]
        self.args = munch.munchify(config)
        self.init_device(config)
        self.init_random_seed(config)
        self.init_model(config)
        self.init_dataset(config)
        self.init_trainer_args(config)
        self.init_predictor_args(config)
        self.init_evaluator_args(config)
        self.prepare_accelerator()
        self.init_quantization_model()

    """
        Main Block
    """

    def predict(self, **kwargs):
        # ADD 接入平台的算法，考虑到并发性，预测相关参数优先从request_property中拿，而不是从self参数中拿
        request_property = kwargs.get('request_property')
        pass

    def evaluate(self, **kwargs):
        global_eval_step = kwargs.get('global_eval_step', 0)
        # ADD 增加EMA model
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
        eval_metric = self._init_metric()  # 自己实现
        # ADD 如果只是在训练前评估一下，不需要全跑完，只要跑少量的step（能打印一次log）就行
        simple_eval_flag = kwargs.get('simple_eval_flag', False)
        if simple_eval_flag:
            logger.info("run evaluation on a small dataset before start training")
        for i, batch in enumerate(self.eval_data_loader):
            if simple_eval_flag and i > self.args.trainer.eval_print_freq:
                break
            start = time.time()
            batch_size = self.args.datasets.train.batch_size
            with torch.no_grad():
                result = self._step_forward(batch, is_train=False, eval_model=eval_model)
            loss_meter.update(result['loss'].item(), batch_size)
            norm_meter.update(0)
            batch_time.update(time.time() - start)
            eval_metric.add_label(result['outputs'], result['targets'])
            global_eval_step += 1
            if self.args.device.is_master and self.args.trainer.eval_print_freq > 0 and global_eval_step % self.args.trainer.eval_print_freq == 0:
                self._print_eval_log(global_eval_step, loss_meter, eval_metric)
            self._display_images(batch, result, global_step=global_eval_step, if_train=False)
        acc = self._print_eval_log(global_eval_step, loss_meter, eval_metric)
        eval_model.float()
        return {'acc': acc, 'global_eval_step': global_eval_step}

    def train(self, **kwargs):
        # 大致模板
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        # ADD resume从对应的epoch，step开始训练
        global_step = self.args.trainer.start_epoch * len(self.train_data_loader)
        global_eval_step = 0
        ni = 0
        self.evaluate(simple_eval_flag=True)  # ADD 在开始训练的时候，先eval一下；防止因为eval写错了，导致之前train的时间都白费了
        for epoch in range(self.args.trainer.start_epoch, self.args.trainer.epochs):
            self.model.zero_grad(set_to_none=True)
            for i, batch in enumerate(self.train_data_loader):
                if global_step <= self.args.trainer.start_global_step:
                    global_step += 1
                    continue
                start = time.time()
                self.model.train()
                ni = i + len(self.train_data_loader) * epoch  # number integrated batches (since train start)
                # ADD 训练时with gradient_accumulate_scope
                with self.gradient_accumulate_scope(self.model):
                    result = self._step_forward(batch)
                    self._step_backward(result['loss'])
                    # ADD 梯度累积 grad_norm改为在_step_optimizer获得
                    if self.accelerator is not None or ((i + 1) % self.args.trainer.grad_accumulate
                                                        == 0) or ((i + 1) == len(self.train_data_loader)):
                        grad_norm = self._step_optimizer()
                        norm_meter.update(grad_norm)
                        if not self.args.trainer.scheduler_by_epoch:
                            self._step_scheduler(global_step)
                loss_meter.update(result['loss'].item(), self.args.datasets.train.batch_size)
                batch_time.update(time.time() - start)
                global_step += 1
                global_eval_step = self._print_step_log(epoch, global_step, global_eval_step, loss_meter, norm_meter,
                                                        batch_time, ni)
                self._display_images(batch, result, global_step=global_step, if_train=True)
            if self.args.trainer.scheduler_by_epoch:
                self._step_scheduler(global_step)
            global_eval_step = self._print_epoch_log(epoch, global_step, global_eval_step, loss_meter, ni)
            if self.args.model.quantization_type == 'quantization_aware_training':
                if epoch > 3:
                    # Freeze quantizer parameters
                    self.model.apply(torch.quantization.disable_observer)
                if epoch > 2:
                    # Freeze batch norm mean and variance estimates
                    self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        model_config_path = self._train_post_process()
        if self.args.device.is_master:
            self.writer.close()
        return {
            'acc': self.args.trainer.best_eval_result,
            'best_model_path': self.args.trainer.best_model_path,
            'model_config_path': model_config_path,
        }

    def _step_forward(self, batch, is_train=True, eval_model=None, **kwargs):
        # ADD注意：由于增加了EMA功能，_step_forward就不一定使用self.model了
        model = eval_model if is_train is False and eval_model is not None else self.model
        # ADD Runs the forward pass with autocasting.
        with self.precision_scope:
            output = model(input)
        return output

    def _step_backward(self, loss, **kwargs):
        # ADD grad norm和clip都不在这一步做
        if self.use_torch_amp:
            self.mixed_scaler.scale(loss).backward()
        else:
            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss = loss / self.args.trainer.grad_accumulate
                loss.backward()

    def _get_current_lr(self, ni, global_step=0, **kwargs):
        if self.args.trainer.scheduler_type == "scheduler2":
            current_lr = self.scheduler.get_update_values(global_step)[-1]
        else:
            current_lr = self.scheduler.get_last_lr()[-1]
        return current_lr

    def _step_optimizer(self, **kwargs):
        # ADD lr_scale，计算grad norm,clip在这里
        params_to_clip = (itertools.chain(self.model.parameters()))
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = param_group["lr"] * param_group["lr_scale"]
        if self.use_torch_amp:
            if self.args.trainer.grad_clip is not None:
                self.mixed_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, self.args.trainer.grad_clip)
            else:
                grad_norm = get_grad_norm(params_to_clip)
            self.mixed_scaler.step(self.optimizer)
            self.mixed_scaler.update()
        else:
            if self.args.trainer.grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, self.args.trainer.grad_clip)
            else:
                grad_norm = get_grad_norm(params_to_clip)
            self.optimizer.step()
        self.model.zero_grad(set_to_none=True)
        if self.ema:
            self.ema.update(self.model)
        return grad_norm

    def _step_scheduler(self, global_step, **kwargs):
        if self.args.trainer.scheduler_type == "scheduler2":
            self.scheduler.step_update(global_step)
        else:
            self.scheduler.step()
    """
        Initialization Functions
    """

    # config的联动关系可以写在这个函数中
    def _init_config(self, config):
        if 'trainer' in config and config.get('phase', 'train') == 'train':
            trainer_args = config["trainer"]
            trainer_args['save_dir'] = get_absolute_file_path(trainer_args.get("save_dir"))
            os.makedirs(trainer_args['save_dir'], exist_ok=True)
            # 存储训练的yml，用于复现训练
            save_params(trainer_args['save_dir'], config)
            train_log_path = os.path.join(trainer_args['save_dir'], "{}.log".format(config['name']))
            file_handler = logging.FileHandler(train_log_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(log_formatter)
            logger.addHandler(file_handler)
        return config

    def init_device(self, config):
        # ADD RUN_ON_GPU_IDs=-1是cpu，多张默认走accelerator
        self.args.device = munch.munchify(config.get('device', {}))
        self.accelerator = None
        self.weight_dtype = torch.float32
        self.gradient_accumulate_scope = nullcontext
        self.precision_scope = nullcontext()
        self.use_torch_amp = False
        if os.environ.get("RUN_ON_GPU_IDs", 0) == str(-1):
            # load model with CPU
            self.args.device.device_id = torch.device("cpu")
            self.args.device.device_ids = [-1]
            self.args.device.is_master = True
            self.args.device.is_distributed = False
        else:
            # accelerator configuration
            if len(os.environ.get("RUN_ON_GPU_IDs", 0)) > 1:
                # If you define multiple visible GPU, I suppose you to use accelerator to do ddp training
                self.accelerator = Accelerator(
                    gradient_accumulation_steps=self.args.trainer.grad_accumulate,
                    mixed_precision=self.args.model.mixed_precision)
                self.args.device.device_id = self.accelerator.device
                self.args.device.device_ids = []
                if self.accelerator.mixed_precision == "fp16":
                    self.weight_dtype = torch.float16
                elif self.accelerator.mixed_precision == "bf16":
                    self.weight_dtype = torch.bfloat16
                self.gradient_accumulate_scope = self.accelerator.accumulate
                self.args.device.is_master = self.accelerator.is_main_process
                self.args.device.is_distributed = self.accelerator.num_processes > 1
            else:
                # USE one GPU specified by user w/o using accelerate
                device_id = os.environ.get("RUN_ON_GPU_IDs", 0)
                self.args.device.device_id = torch.device("cuda:{}".format(device_id))
                self.args.device.device_ids = [int(device_id)]
                torch.cuda.set_device(int(device_id))
                self.args.device.is_master = True
                self.args.device.is_distributed = False
                if self.args.model.mixed_precision in ["fp16", "bp16"]:
                    # ADD mixed_precision_flag改为use_torch_amp
                    self.use_torch_amp = True
                    self.weight_dtype = torch.float16
                    self.precision_scope = autocast(device_type="cuda", dtype=self.weight_dtype)
        logger.info("device:{}, is_master:{}, device_ids:{}, is_distributed:{}".format(
            self.args.device.device_id, self.args.device.is_master, self.args.device.device_ids,
            self.args.device.is_distributed))

    def init_model(self, config):
        model_args = config["model"]
        self.model = get_network(model_args)
        if "model_path" in model_args and model_args['model_path'] is not None:
            model_path = get_absolute_file_path(model_args['model_path'])
            self.load_model(model_path)
        self.model.to(self.args.device.device_id)
        total = sum([param.nelement() for param in self.model.parameters()])
        logger.info("Number of parameter: %.2fM" % (total / 1e6))
        # from base.torch_utils.dl_util import print_network
        # print_network(self.model, True)

    def init_dataset(self, config):
        if 'datasets' in config and config.get('phase', 'train') != 'predict':
            dataset_args = config.get("datasets")
            train_data_loader_args = dataset_args.get("train")
            if config.get('phase', 'train') == 'train':
                self.train_dataset = get_dataset(train_data_loader_args['dataset'])
                self.train_data_loader = self._get_data_loader_from_dataset(self.train_dataset,
                                                                            train_data_loader_args,
                                                                            phase='train')
                logger.info("success init train data loader len:{} ".format(len(self.train_data_loader)))
            eval_data_loader_args = dataset_args.get("eval")
            merged_eval_data_loader_args = train_data_loader_args.copy()
            merge_config(eval_data_loader_args, merged_eval_data_loader_args)
            self.eval_dataset = get_dataset(merged_eval_data_loader_args['dataset'])
            self.eval_data_loader = self._get_data_loader_from_dataset(self.eval_dataset,
                                                                       merged_eval_data_loader_args,
                                                                       phase='eval')
            logger.info("success init eval data loader len:{}".format(len(self.eval_data_loader)))

    def init_quantization_model(self):
        # 注意：所有的量化都必须用cpu跑
        self.post_training_dynamic_quantization_preprocess()
        self.post_training_static_quantization_preprocess()
        self.quantization_aware_training_preprocess()

    def init_random_seed(self, config):
        if 'random_seed' in config['trainer']:
            seed_all(config['trainer']['random_seed'])
        else:
            logger.warning("random seed is missing")

    def init_predictor_args(self, config):
        if 'predictor' in config and config.get('phase', 'train') == 'predict':
            predictor_args = config["predictor"]
            self.args.predictor.img_paths = predictor_args.get("img_paths", [])
            if self.args.predictor.img_paths is None:
                self.args.predictor.img_paths = []
            img_dirs = predictor_args.get("img_dirs", [])
            if img_dirs:
                for img_dir in img_dirs:
                    self.args.predictor.img_paths.extend(get_file_path_list(img_dir, ['jpg', 'png', 'jpeg']))
            if predictor_args['save_dir'] is None and 'model_path' in config['model'] and config['model'][
                    'model_path'] is not None:
                predictor_args['save_dir'] = os.path.join(os.path.dirname(config['model']['model_path']),
                                                          'test_results')
            self.args.predictor.save_dir = get_absolute_file_path(predictor_args["save_dir"])
            os.makedirs(self.args.predictor.save_dir, exist_ok=True)

    def init_evaluator_args(self, config):
        if 'evaluator' in config and config.get('phase', 'train') != 'predict':
            evaluator_args = config["evaluator"]
            self.args.evaluator.save_dir = get_absolute_file_path(evaluator_args.get("save_dir"))
            os.makedirs(self.args.evaluator.save_dir, exist_ok=True)

    def init_trainer_args(self, config):
        if 'trainer' in config and config.get('phase', 'train') == 'train':
            trainer_args = config["trainer"]
            self._init_optimizer(trainer_args)
            self._init_scheduler(trainer_args)
            logger.info("current trainer  epochs:{}, train_dataset_len:{}, data_loader_len:{}".format(
                self.args.trainer.epochs, len(self.train_dataset), len(self.train_data_loader)))
            self.mixed_scaler = torch.cuda.amp.GradScaler(enabled=True) if self.use_torch_amp else None
            self.args.trainer.best_eval_result = -1
            self.args.trainer.best_model_path = ''
            self.ema = ModelEMA(self.model) if trainer_args['use_ema'] else None
            self.args.trainer.start_epoch = 0
            self.args.trainer.start_global_step = 0
            if self.args.trainer.resume_flag and 'model_path' in self.args.model and self.args.model.model_path is not None:
                # ADD resume
                resume_path = self.args.model.model_path.replace('.pth', '_resume.pth')
                if os.path.exists(resume_path):
                    resume_checkpoint = torch.load(resume_path)
                    self.optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
                    self.scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
                    self.args.trainer.start_epoch = resume_checkpoint['epoch']
                    self.args.trainer.start_global_step = resume_checkpoint['global_step']
                else:
                    logger.warning("resume path {} doesn't exist: failed to resume!!".format(resume_path))

        if 'trainer' in config and config.get('phase', 'train') != 'predict':
            trainer_args = config["trainer"]
            self._init_criterion(trainer_args)
            # init tensorboard and log
            if "tensorboard_dir" in trainer_args and self.args.device.is_master:
                tensorboard_log_dir = get_absolute_file_path(trainer_args.get("tensorboard_dir"))
                os.makedirs(tensorboard_log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=tensorboard_log_dir, comment=self.experiment_name)
            else:
                self.writer = None

    def _init_optimizer(self, trainer_args, **kwargs):
        optimizer_args = trainer_args.get("optimizer")
        # ADD scale lr
        if optimizer_args["scale_lr"]:
            num_process = 1 if self.accelerator is None else self.accelerator.num_processes
            optimizer_args['lr'] = optimizer_args['lr'] * self.args.trainer.grad_accumulate * \
                                   self.train_data_loader.batch_size * num_process
        self.optimizer = get_optimizer(self.model, **optimizer_args)

    def _init_scheduler(self, trainer_args, **kwargs):
        scheduler_args = trainer_args.get("scheduler")
        self.args.trainer.scheduler_by_epoch = scheduler_args.get("scheduler_by_epoch", False)
        total_epoch_train_steps = len(self.train_data_loader)
        if "warmup_epochs" in scheduler_args:
            warmup_steps = scheduler_args.get("warmup_epochs") * total_epoch_train_steps
        elif "warmup_steps" in scheduler_args:
            warmup_steps = scheduler_args.get("warmup_steps")
        else:
            warmup_steps = 0
        self.args.trainer.scheduler.warmup_steps = warmup_steps
        num_training_steps = total_epoch_train_steps // self.args.trainer.grad_accumulate * self.args.trainer.epochs
        if "scheduler_method" in scheduler_args and scheduler_args["scheduler_method"] == "get_scheduler2":
            self.scheduler = get_scheduler2(self.optimizer,
                                            num_training_steps=num_training_steps,
                                            num_warmup_steps=warmup_steps,
                                            **scheduler_args)
            self.args.trainer.scheduler_type = "scheduler2"
        else:
            self.scheduler = get_scheduler(self.optimizer,
                                           num_training_steps=num_training_steps,
                                           num_warmup_steps=warmup_steps,
                                           epochs=self.args.trainer.epochs,
                                           **scheduler_args)
            self.args.trainer.scheduler_type = "scheduler"

        logger.info(
            "success init optimizer and scheduler, optimizer:{}, scheduler:{}, scheduler_args:{}, warmup_steps:{},"
            "num_training_steps:{}, gradient_accumulator:{}".format(self.optimizer, self.scheduler, scheduler_args,
                                                                    warmup_steps, num_training_steps,
                                                                    self.args.trainer.grad_accumulate))

    def _init_criterion(self, trainer_args):
        if "loss" in trainer_args:
            loss_args = trainer_args.pop("loss")
            criterion_type = loss_args.pop("type")
            self.criterion = get_criterion(criterion_type, loss_args)
        else:
            logger.warning(
                "skip initialize criterion. If you are using XXForTokenClassification or "
                "other models that have initialized citation inside the model, it will be ok to skip this phase;"
                " if not, please double check you are not missing anything inside base.yaml")

    def _init_metric(self, **kwargs):
        metric = get_metric(self.args.trainer.metric)
        return metric

    """
        Tool Functions
    """

    def load_model(self, checkpoint_path, strict=True, **kwargs):
        if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(state_dict, strict=strict)
            logger.info("success load model:{}".format(checkpoint_path))

    def unload(self):
        self.model = None

    def save_model(self, checkpoint_path, **save_kwargs):
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            self.accelerator.save(unwrapped_model.state_dict(), checkpoint_path)
        else:
            if self.args.model.quantization_type == 'quantization_aware_training':
                self.model.eval()
                model_int8 = torch.quantization.convert(self.model)
                torch.save(model_int8.state_dict(), checkpoint_path)
            else:
                torch.save(self.model.state_dict(), checkpoint_path)
                # ADD resume
                if self.args.trainer.resume_flag:
                    save_kwargs.update({
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                    })
                    torch.save(save_kwargs, checkpoint_path.replace('.pth', '_resume.pth'))
        logger.info("model successfully saved to {}".format(checkpoint_path))

    # only for evaluate or predict；只支持量化linear\LSTM\RNN\Transformer,不支持CNN
    def post_training_dynamic_quantization_preprocess(self):
        if self.args.model.quantization_type == 'post_training_dynamic_quantization' and self.args.phase != 'train':
            # 量化前跑一次看看原始结果
            if self.args.phase == 'evaluate':
                self.evaluate()
            else:
                self.predict()
            self.model = torch.quantization.quantize_dynamic(
                self.model,  # the original model
                # {torch.nn.Linear},  # a set of layers to dynamically quantize
                dtype=torch.qint8)  # the target dtype for quantized weights
            os.makedirs(self.args.trainer.save_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.args.trainer.save_dir, 'post_training_dynamic_quantization_model.pth')
            self.save_model(checkpoint_path)

    # only for evaluate or predict；只支持量化linear\CNN,不支持LSTM\RNN
    def post_training_static_quantization_preprocess(self):
        if self.args.model.quantization_type == 'post_training_static_quantization' and self.args.phase != 'train':
            # self.model.qconfig = torch.quantization.default_qconfig # 不同的量化方法，fbgemm适合x86，qnnpack适合arm
            self.model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
            # self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            self.model.fuse_model(
            )  # model记得实现这个方法，可fuse的layer：[Conv, Relu], [Conv, BatchNorm], [Conv, BatchNorm, Relu], [Linear, Relu]
            self.model = torch.quantization.prepare(self.model)
            # 量化前跑一次看看原始结果,static量化必须先forward
            if self.args.phase == 'evaluate':
                self.evaluate()
            else:
                self.predict()
            self.model = torch.quantization.convert(self.model)
            os.makedirs(self.args.trainer.save_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.args.trainer.save_dir, 'post_training_static_quantization_model.pth')
            self.save_model(checkpoint_path)

    # only for train；只支持量化linear\CNN,不支持LSTM\RNN
    def quantization_aware_training_preprocess(self):
        if self.args.model.quantization_type == 'quantization_aware_training' and self.args.phase == 'train':
            # self.model.qconfig = torch.quantization.default_qconfig # 不同的量化方法，fbgemm适合x86，qnnpack适合arm
            self.model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
            # self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            self.model.fuse_model()
            self.model.train()
            torch.quantization.prepare_qat(self.model, inplace=True)

    def _get_data_loader_from_dataset(self, dataset, data_loader_args, phase="train"):
        num_workers = data_loader_args.get("num_workers", 0)
        batch_size = data_loader_args.get("batch_size", 2)
        if phase == "train" and data_loader_args.get('shuffle', True):
            shuffle = data_loader_args.get("shuffle", True)
        else:
            shuffle = data_loader_args.get("shuffle", False)
        pin_memory = data_loader_args.get("shuffle", False)

        collate_fn_args = data_loader_args.get("collate_fn")
        if collate_fn_args.get("type") is None:
            collate_fn = None
        else:
            collate_fn_type = collate_fn_args.get("type")
            collate_fn = getattr(mydatasets, collate_fn_type)(batch_size=batch_size, **collate_fn_args)
        data_loader = DataLoader(dataset,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 collate_fn=collate_fn,
                                 batch_size=batch_size)
        logger.info("use data loader with batch_size:{},num_workers:{}".format(batch_size, num_workers))

        return data_loader

    # 初始化 accelerator
    def prepare_accelerator(self):
        if self.accelerator is not None:
            self.model, self.optimizer, self.train_data_loader, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_data_loader, self.scheduler)

    # 训练结束后存一个yml用来存储最好的model的配置，便于predict
    def _train_post_process(self):
        args = copy.deepcopy(self.args)
        args.model.model_path = args.trainer.best_model_path
        if 'base' in args:
            args.pop('base')
        args.device.pop('device_id')
        args.pop('trainer')
        args.phase = 'predict'
        save_params(self.args.trainer.save_dir, json.loads(json.dumps(args)), 'model_args.yaml')
        return os.path.join(self.args.trainer.save_dir, 'model_args.yaml')

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
            global_eval_step, acc = result['global_eval_step'], result['acc']
            # ADD is_master判断移到这里
            if (not self.args.trainer.save_best or (self.args.trainer.save_best
                                                   and acc > self.args.trainer.best_eval_result)) and self.args.device.is_master:
                checkpoint_name = "{}_epoch{}_step{}_lr{:e}_average_loss{:.5f}_acc{:.5f}.pth".format(
                    self.experiment_name, epoch, global_step, current_lr, loss_meter.avg, acc)
                checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
                # ADD记得传epoch和global_step，resume才能存
                self.save_model(checkpoint_path, epoch=epoch, global_step=global_step)
                if acc > self.args.trainer.best_eval_result:
                    self.args.trainer.best_eval_result = acc
                    self.args.trainer.best_model_path = checkpoint_path
        return global_eval_step

    def _print_epoch_log(self, epoch, global_step, global_eval_step, loss_meter, ni, **kwargs):
        current_lr = self._get_current_lr(ni, global_step)
        if self.args.trainer.save_epoch_freq > 0 and epoch % self.args.trainer.save_epoch_freq == 0:
            message = "experiment:{}; eval, (epoch: {}, steps: {});".format(self.experiment_name, epoch, global_step)
            logger.info(message)
            result = self.evaluate(global_eval_step=global_eval_step)
            global_eval_step, acc = result['global_eval_step'], result['acc']
            # ADD is_master判断移到这里
            if (not self.args.trainer.save_best or (self.args.trainer.save_best
                                                   and acc > self.args.trainer.best_eval_result)) and self.args.device.is_master:
                checkpoint_name = "{}_epoch{}_step{}_lr{:e}_average_loss{:.5f}_acc{:.5f}.pth".format(
                    self.experiment_name, epoch, global_step, current_lr, loss_meter.avg, acc)
                checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
                # ADD记得传epoch和global_step，resume才能存
                self.save_model(checkpoint_path, epoch=epoch, global_step=global_step)
                if acc > self.args.trainer.best_eval_result:
                    self.args.trainer.best_eval_result = acc
                    self.args.trainer.best_model_path = checkpoint_path
        return global_eval_step

    def _print_eval_log(self, global_step, loss_meter, eval_metric, **kwargs):
        evaluate_report = eval_metric.get_report()
        acc = evaluate_report["acc"]
        message = "experiment:{}; eval,global_step:{}, (step_mean_loss:{},average_loss:{:.5f},evaluate_report:{})".format(
            self.experiment_name, global_step, loss_meter.val, loss_meter.avg, evaluate_report)
        logger.info(message)
        if self.writer is not None:
            self.writer.add_scalar("{}_eval/step_loss".format(self.experiment_name), loss_meter.val, global_step)
            self.writer.add_scalar("{}_eval/average_loss".format(self.experiment_name), loss_meter.avg, global_step)
            self.writer.add_scalar("{}_eval/acc".format(self.experiment_name), acc, global_step)
        return acc

    def _display_images(self, batch, result, global_step=0, if_train=False, **kwargs):
        if ((self.args.trainer.display_freq > 0 and global_step % self.args.trainer.display_freq == 0 and if_train) or
            (self.args.trainer.eval_display_freq > 0 and global_step % self.args.trainer.eval_display_freq == 0
             and not if_train)) and self.args.device.is_master and self.writer is not None:
            assert isinstance(self.writer, SummaryWriter)
            pass
