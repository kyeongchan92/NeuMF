from typing import Dict
import argparse

parser = argparse.ArgumentParser(description='NeuMF')

parser.add_argument('--ratings_path', required=False, default='data/ml-1m/ratings.dat')
parser.add_argument('--emb_dim', required=False, default=256)
parser.add_argument('--neg_ratio', required=False, default=3)
parser.add_argument('--prep_data_dir', required=False, default='prep_data')
parser.add_argument('--prep_data_name', required=False, default='prep_data.pkl')
parser.add_argument('--test_ratio', required=False, default=0.1)
parser.add_argument('--batch_size', required=False, default=128)
parser.add_argument('--lr', required=False, default=0.0005)
parser.add_argument('--n_epoch', required=False, default=50)
parser.add_argument('--train_test_split_rs', required=False, default=1234)
parser.add_argument('--topk', required=False, default=25)
parser.add_argument('--model_save_path', required=False, default='result/neumf.pth')


args = parser.parse_args()