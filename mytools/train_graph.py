# -*- coding:utf-8 -*-
# create: 2021/6/10
import os, sys
import setproctitle

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT_PATH)
os.environ['RUN_ON_GPU_IDs'] = "0"

import argparse
import experiment

# torch.multiprocessing.set_sharing_strategy('file_system')
from base.common_util import get_absolute_file_path, init_experiment_config
from experiment import get_experiment_name


def init_args():
    parser = argparse.ArgumentParser(description='trainer args')
    parser.add_argument(
        '--config_file',
        # default='config/graph_net/train_layout_graph_doclaynet.yaml',
        # default='config/graph_net/train_layout_graph_publaynet.yaml',
        default='config/graph_net/train_layout_graph_funsd.yaml',
        # default='config/graph_net/train_layout_graph_funsd_entity.yaml',
        type=str,
    )
    parser.add_argument(
        '--experiment_name',
        default='GraphLayout',
        type=str,
    )
    parser.add_argument(
        '--phase',
        default='train',
        # default='predict',
        # default='evaluate',
        type=str,
    )
    args = parser.parse_args()
    os.environ['WORKSPACE'] = args.experiment_name
    return args


def main(args):
    config = init_experiment_config(args.config_file, args.experiment_name)
    config.update({'phase': args.phase})
    experiment_instance = getattr(experiment, get_experiment_name(args.experiment_name))(config)
    if args.phase == 'train':
        experiment_instance.train()
    elif args.phase == 'predict':
        experiment_instance.predict()
    elif args.phase == 'evaluate':
        experiment_instance.evaluate()


if __name__ == '__main__':
    args = init_args()
    setproctitle.setproctitle("{} task for {}".format(args.experiment_name, args.config_file.split('/')[-1]))
    main(args)


