# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/7/2
import torch.distributed as dist
import copy
import time
from torch.nn.parallel import DistributedDataParallel
import munch
from metrics.meter import AverageMeter
from base.torch_utils.dl_util import get_grad_norm
import mydatasets
from torch.utils.data import DataLoader, DistributedSampler
from base.common_util import save_params
from base.torch_utils.dl_util import get_optimizer, get_scheduler, get_scheduler2
import logging
import json
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from loss import get_criterion
from metrics import get_metric
from base.driver import log_formatter, PROJECT_ROOT_PATH
from base.common_util import get_absolute_file_path, merge_config, get_file_path_list
from base.driver import logger
from mydatasets import get_dataset
from networks import get_network
from accelerate import Accelerator


class BaseExperiment(object):

    def __init__(self, config, local_world_size=None, local_rank=None):
        config = self._init_config(config)
        self.experiment_name = config["name"]
        self.args = munch.munchify(config)
        self.init_device(config, local_world_size, local_rank)
        self.init_model(config)
        self.init_dataset(config)
        self.init_trainer_args(config)
        self.init_predictor_args(config)
        self.init_evaluator_args(config)
        self.prepare_accelerator()
        self.init_quantization_model()

    def load_model(self, checkpoint_path, strict=True, **kwargs):
        if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(state_dict, strict=strict)
            logger.info("success load model:{}".format(checkpoint_path))

    def unload(self):
        self.model = None

    def save_model(self, checkpoint_path):
        if isinstance(self.model, torch.nn.DataParallel):
            net = self.model.module
        elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            net = self.model.module
        else:
            net = self.model
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(net)
            self.accelerator.save(unwrapped_model.state_dict(), checkpoint_path)
        elif self.args.device.is_master:
            if self.args.model.quantization_type == 'quantization_aware_training':
                self.model.eval()
                model_int8 = torch.quantization.convert(self.model)
                torch.save(model_int8.state_dict(), checkpoint_path)
            else:
                torch.save(net.state_dict(), checkpoint_path)
        if self.args.device.is_distributed and not self.args.device.is_master:
            dist.barrier()
            self.model.load_state_dict(torch.load(net.state_dict(), map_location=self.args.device.device_id))
            logger.info("model restore for rank: {}".format(self.args.device.local_rank))
        logger.info("model successfully saved to {}".format(checkpoint_path))

    def predict(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        global_eval_step = kwargs.get('global_eval_step', 0)
        if self.args.model.mixed_precision_flag:
            self.model.half()
        self.model.eval()
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        eval_metric = self._init_metric()  # 自己实现
        for i, batch in enumerate(self.eval_data_loader):
            start = time.time()
            batch_size = self.args.datasets.train.batch_size
            with torch.no_grad():
                result = self._step_forward(batch, is_train=False)
            loss_meter.update(result['loss'].item(), batch_size)
            norm_meter.update(0)
            batch_time.update(time.time() - start)
            eval_metric.add_label(result['outputs'], result['targets'])
            global_eval_step += 1
            if self.args.device.is_master and self.args.trainer.eval_print_freq > 0 and global_eval_step % self.args.trainer.eval_print_freq == 0:
                self._print_eval_log(global_eval_step, loss_meter, eval_metric)
            self._display_images(batch, result, global_step=global_eval_step, if_train=False)
        acc = self._print_eval_log(global_eval_step, loss_meter, eval_metric)
        self.model.float()
        return {
            'acc': acc,
            'global_eval_step': global_eval_step
        }

    def train(self, **kwargs):
        # 大致模板
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        global_step = 0
        global_eval_step = 0
        ni = 0
        for epoch in range(self.args.trainer.epochs):
            self.model.zero_grad(set_to_none=True)
            if self.args.device.is_distributed:
                # let all processes sync up before starting with a new epoch of training
                dist.barrier()
            for i, batch in enumerate(self.train_data_loader):
                start = time.time()
                self.model.train()
                ni = i + len(self.train_data_loader) * epoch  # number integrated batches (since train start)
                result = self._step_forward(batch)
                grad_norm = self._step_backward(result['loss'])
                if ((i + 1) % self.args.trainer.grad_accumulate == 0) or ((i + 1) == len(self.train_data_loader)):
                    self._step_scheduler(global_step)
                loss_meter.update(result['loss'].item(), self.args.datasets.train.batch_size)
                norm_meter.update(grad_norm)
                batch_time.update(time.time() - start)
                global_step += 1
                global_eval_step = self._print_step_log(epoch, global_step, global_eval_step, loss_meter, norm_meter,
                                                        batch_time, ni)
                self._display_images(batch, result, global_step=global_step, if_train=True)
            if self.args.trainer.scheduler_by_epoch:
                self.scheduler.step()
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

    def _step_forward(self, batch, is_train=True, **kwargs):
        pass

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

    def init_device(self, config, local_world_size, local_rank):
        self.args.device = munch.munchify(config.get('device', {}))
        self.accelerator = None
        if os.environ.get("RUN_ON_GPU_IDs", 0) == str(-1):
            self.args.device.device_id = torch.device("cpu")
            self.args.device.device_ids = [-1]
            self.args.device.is_master = True
            self.args.device.is_distributed = False
        elif local_world_size is not None and local_rank is not None:
            # Device Configuration for Native Torch DDP
            if os.environ.get("RUN_ON_GPU_IDs", False):
                device_ids = list(os.environ.get("RUN_ON_GPU_IDs").split(","))
                n = len(device_ids)
            else:
                n = torch.cuda.device_count() // local_world_size
                device_ids = list(range(local_rank * n, (local_rank + 1) * n))
            rank = dist.get_rank()
            device_id = device_ids[rank]
            self.args.device.device_id = torch.device("cuda:{}".format(device_id))
            logger.info("pid:{}, rank:{}, world_size:{}, n:{}, device_ids:{}".format(os.getpid(), rank,
                                                                                     dist.get_world_size(), n,
                                                                                     device_id))
            self.args.device.is_master = rank == 0
            self.args.device.is_distributed = True
            self.args.device.device_ids = device_ids
            self.args.device.world_size = dist.get_world_size()
        else:
            if len(os.environ.get("CUDA_VISIBLE_DEVICES", [0])) > 1:
                # If you define multiple visible GPU, it suppose you to use accelerator to do ddp training
                self.accelerator = Accelerator()
                self.args.device.device_id = self.accelerator.device
                self.args.device.device_ids = []
            else:
                # USE one GPU specified by user
                device_id = os.environ.get("RUN_ON_GPU_IDs", 0)
                self.args.device.device_id = torch.device("cuda:{}".format(device_id))
                self.args.device.device_ids = [int(device_id)]
                torch.cuda.set_device(int(device_id))
            self.args.device.is_master = True
            self.args.device.is_distributed = False
        self.args.device.local_world_size = local_world_size
        self.args.device.local_rank = local_rank
        logger.info("device:{}, is_master:{}, device_ids:{}, is_distributed:{}".format(
            self.args.device.device_id, self.args.device.is_master, self.args.device.device_ids,
            self.args.device.is_distributed))

    def init_model(self, config):
        model_args = config["model"]
        self.model = get_network(model_args)
        if "model_path" in model_args and model_args['model_path'] is not None:
            model_path = get_absolute_file_path(model_args['model_path'])
            self.load_model(model_path)
        if self.accelerator is not None:
            self.model.to(self.args.device.device_id)
        elif self.args.device.device_id.type != 'cpu':
            torch.cuda.set_device(self.args.device.device_id.index)
            self.model.to(self.args.device.device_id)
            if self.args.device.is_distributed:
                self.model = DistributedDataParallel(self.model,
                                                     device_ids=[self.args.device.device_id],
                                                     find_unused_parameters=True,
                                                     output_device=self.args.device.device_id)
                logger.info("int ddp model for {}".format(self.model))
        self.args.model.mixed_precision_flag = True if config['model'].get(
            "mixed_precision_flag", False) and self.args.device.device_id.type != 'cpu' else False
        total = sum([param.nelement() for param in self.model.parameters()])
        logger.info("Number of parameter: %.2fM" % (total / 1e6))

    def init_quantization_model(self):
        # 注意：所有的量化都必须用cpu跑
        self.post_training_dynamic_quantization_preprocess()
        self.post_training_static_quantization_preprocess()
        self.quantization_aware_training_preprocess()

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
            self.model.fuse_model()  # model记得实现这个方法，可fuse的layer：[Conv, Relu], [Conv, BatchNorm], [Conv, BatchNorm, Relu], [Linear, Relu]
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

    def _get_data_loader_from_dataset(self, dataset, data_loader_args, phase="train"):
        num_workers = data_loader_args.get("num_workers", 0)
        batch_size = data_loader_args.get("batch_size", 2)
        if phase == "train" and data_loader_args.get('shuffle', True):
            shuffle = data_loader_args.get("shuffle", True)
        else:
            shuffle = data_loader_args.get("shuffle", False)
        pin_memory = data_loader_args.get("shuffle", False)

        collate_fn_args = data_loader_args.get("collate_fn")
        collate_fn_args['device'] = self.args.device.device_id

        # distributed dataset and model
        if self.args.device.is_distributed and phase == "train":
            shuffle = data_loader_args.get("shuffle", False)
            drop_last = data_loader_args.get("drop_last", False)
            self.train_sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
            sampler = self.train_sampler
            batch_size = batch_size // self.args.device.local_world_size
            num_workers = (num_workers + self.args.device.local_world_size - 1) // self.args.device.local_world_size
            # num_workers = 0  # I found creating extra threads in the children processes may be problemistic, for example, lmdb object
            # pin_memory = False # found pin_memory=False avoids many horrible bugs as well
            if collate_fn_args is None:
                collate_fn = None
            else:
                collate_fn_type = collate_fn_args.get("type")
                collate_fn = getattr(mydatasets, collate_fn_type)(batch_size=batch_size, **collate_fn_args)
            data_loader = DataLoader(dataset,
                                     shuffle=False,
                                     # SET This To False, shuffle has been done in sampler initialization
                                     num_workers=num_workers,
                                     pin_memory=pin_memory,
                                     collate_fn=collate_fn,
                                     batch_size=batch_size,
                                     sampler=sampler)
            logger.info("use distributed data sampler with batch_size:{},num_workers:{}".format(
                batch_size, num_workers))
        else:
            if collate_fn_args is None:
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
            if predictor_args['save_dir'] is None and 'model_path' in config['model'] and config['model']['model_path'] is not None:
                predictor_args['save_dir'] = os.path.join(os.path.dirname(config['model']['model_path']), 'test_results')
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
            self.mixed_scaler = torch.cuda.amp.GradScaler(
                enabled=True) if self.args.model.mixed_precision_flag else None
            self.args.trainer.best_eval_result = -1
            self.args.trainer.best_model_path = ''
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
        self.optimizer = get_optimizer(self.model, **optimizer_args)

    def _init_scheduler(self, trainer_args, **kwargs):
        scheduler_args = trainer_args.get("scheduler")
        grad_accumulate = trainer_args.get("grad_accumulate", 1)
        self.args.trainer.scheduler_by_epoch = scheduler_args.get("scheduler_by_epoch", False)
        total_epoch_train_steps = len(self.train_dataset) // self.train_data_loader.batch_size
        if "warmup_epochs" in scheduler_args:
            warmup_steps = scheduler_args.get("warmup_epochs") * total_epoch_train_steps
        elif "warmup_steps" in scheduler_args:
            warmup_steps = scheduler_args.get("warmup_steps")
        else:
            warmup_steps = 0
        num_training_steps = total_epoch_train_steps // grad_accumulate * self.args.trainer.epochs
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
        self.args.trainer.scheduler.nw = self.args.trainer.scheduler.get('warmup_steps_only', 0)
        if self.args.device.is_distributed:
            logger.info(
                "success init optimizer and scheduler, optimizer:{}, scheduler:{}, scheduler_args:{}, warmup_steps:{},"
                "num_training_steps:{}, gradient_accumulator:{}".format(self.optimizer, self.scheduler, scheduler_args,
                                                                        warmup_steps, num_training_steps,
                                                                        grad_accumulate))

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

    def prepare_accelerator(self):
        if self.accelerator is not None:
            self.model, self.optimizer, self.train_data_loader, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_data_loader, self.scheduler)

    def _get_current_lr(self, ni, global_step=0, **kwargs):
        if ni <= self.args.trainer.scheduler.nw and self.args.trainer.scheduler.get('warmup_only', False):
            current_lr = self.optimizer.param_groups[0]['lr']
        else:
            if self.args.trainer.scheduler_type == "scheduler2":
                current_lr = self.scheduler.get_update_values(global_step)[-1]
            else:
                current_lr = self.scheduler.get_last_lr()[-1]
        return current_lr

    def _step_optimizer(self, **kwargs):
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = param_group["lr"] * param_group["lr_scale"]
        if self.args.model.mixed_precision_flag:
            self.mixed_scaler.step(self.optimizer)
            self.mixed_scaler.update()
        else:
            self.optimizer.step()
        self.model.zero_grad(set_to_none=True)

    def _step_scheduler(self, global_step, **kwargs):
        self._step_optimizer(**kwargs)
        if not self.args.trainer.scheduler_by_epoch:
            if self.args.trainer.scheduler_type == "scheduler2":
                self.scheduler.step_update(global_step)
            else:
                self.scheduler.step()

    def _step_backward(self, loss, **kwargs):
        if self.args.model.mixed_precision_flag:
            self.mixed_scaler.scale(loss).backward()
            if self.args.trainer.grad_clip is not None:
                self.mixed_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.trainer.grad_clip)
            else:
                grad_norm = get_grad_norm(self.model.parameters())
        else:
            loss = loss / self.args.trainer.grad_accumulate
            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            if self.args.trainer.grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.trainer.grad_clip)
            else:
                grad_norm = get_grad_norm(self.model.parameters())
        return grad_norm

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
        if global_step > 0 and self.args.trainer.save_step_freq > 0 and self.args.device.is_master and global_step % self.args.trainer.save_step_freq == 0:
            message = "experiment:{}; eval, (epoch: {}, steps: {});".format(self.experiment_name, epoch, global_step)
            logger.info(message)
            result = self.evaluate(global_eval_step=global_eval_step)
            global_eval_step, acc = result['global_eval_step'], result['acc']
            if not self.args.trainer.save_best or (self.args.trainer.save_best
                                                   and acc > self.args.trainer.best_eval_result):
                checkpoint_name = "{}_epoch{}_step{}_lr{:e}_average_loss{:.5f}_acc{:.5f}.pth".format(
                    self.experiment_name, epoch, global_step, current_lr, loss_meter.avg, acc)
                checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
                self.save_model(checkpoint_path)
                if acc > self.args.trainer.best_eval_result:
                    self.args.trainer.best_eval_result = acc
                    self.args.trainer.best_model_path = checkpoint_path
        return global_eval_step

    def _print_epoch_log(self, epoch, global_step, global_eval_step, loss_meter, ni, **kwargs):
        current_lr = self._get_current_lr(ni, global_step)
        if self.args.trainer.save_epoch_freq > 0 and self.args.device.is_master and epoch % self.args.trainer.save_epoch_freq == 0:
            message = "experiment:{}; eval, (epoch: {}, steps: {});".format(self.experiment_name, epoch, global_step)
            logger.info(message)
            result = self.evaluate(global_eval_step=global_eval_step)
            global_eval_step, acc = result['global_eval_step'], result['acc']
            if not self.args.trainer.save_best or (self.args.trainer.save_best
                                                   and acc > self.args.trainer.best_eval_result):
                checkpoint_name = "{}_epoch{}_step{}_lr{:e}_average_loss{:.5f}_acc{:.5f}.pth".format(
                    self.experiment_name, epoch, global_step, current_lr, loss_meter.avg, acc)
                checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
                self.save_model(checkpoint_path)
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
