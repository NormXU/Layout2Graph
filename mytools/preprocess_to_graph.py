import argparse
import os
import sys

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT_PATH)

from scripts.preprocess_data import TestGenerator


def get_args_parser():
    parser = argparse.ArgumentParser("Preprocess the Dataset", add_help=False)
    parser.add_argument("--dataset", "-d", type=str, required=True, 
                        choices=['doclaynet', 'funsd', 'publaynet'],
                        help='Specify the dataset name')
    
    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    tg = TestGenerator()
    tg.setUp()
    if args.dataset == 'doclaynet':
        tg.test_convert_DocLayNet2Graph()
    elif args.dataset == 'funsd':
        tg.test_convert_Funsd2Graph()
    elif args.dataset == 'publaynet':
        tg.test_convert_Publaynet2Graph()
