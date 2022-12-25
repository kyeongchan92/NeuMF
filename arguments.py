from typing import Dict
import argparse

parser = argparse.ArgumentParser(description='NeuMF')

parser.add_argument('--ratings_path', required=False, default='data/ml-1m/ratings.dat')
parser.add_argument('--emb_dim', required=False, default=64)
parser.add_argument('--neg_ratio', required=False, default=3)
parser.add_argument('--prep_data_dir', required=False, default='prep_data')
parser.add_argument('--prep_data_name', required=False, default='prep_data.pkl')
parser.add_argument('--test_ratio', required=False, default=0.1)
parser.add_argument('--batch_size', required=False, default=128)
parser.add_argument('--lr', required=False, default=0.01)
parser.add_argument('--n_epoch', required=False, default=30)

def update_dataset_args(args, new_args:Dict):
    for new_arg, new_value in new_args.items():
        print(new_arg, new_value)
        args.new_arg = new_value
        for k, v in vars(args).items():
            print(f"{k:>20} : {v:<}")
    return args

args = parser.parse_args()