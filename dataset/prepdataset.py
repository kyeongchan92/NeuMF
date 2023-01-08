import os
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class ML1mDataset:
    def __init__(self, args):
        self.args = args
    
    def preprocess(self):
        if os.path.isfile(self.get_prep_data_path()):
            print(f"Already exist preprocessed data".ljust(60, '='))
            print(f"prep_data_path : {self.get_prep_data_path()}")
            print(f"".ljust(60, '='))
        else: 
            print(f"Preprocess".ljust(60, '='))
            ratings = self.load_ratings_path()

            print(f"Densify".ljust(60, '-'))
            uidx_map = {user:i for i, user in enumerate(set(ratings['user']))}
            iidx_map = {item:i for i, item in enumerate(set(ratings['item']))}

            ratings['uidx'] = ratings['user'].map(uidx_map)
            ratings['iidx'] = ratings['item'].map(iidx_map)
            print(f"".ljust(60, '-'))

            n_users = ratings['uidx'].nunique()
            n_items = ratings['iidx'].nunique()
            self.args.n_users = n_users
            self.args.n_items = n_items
            print(f"".ljust(60, '-'))

            # train test split
            print(f"train test split".ljust(60, '-'))
            train_df, test_df = train_test_split(ratings, test_size=self.args.test_ratio, random_state=self.args.train_test_split_rs)
            print(f"train size : {len(train_df):,}, test_size : {len(test_df):,}")
            print(f"".ljust(60, '-'))

            print(f"Negative sampling for train data".ljust(60, '-'))
            pos_items = train_df.groupby('uidx')['iidx'].agg(lambda x: set(x)).to_dict()
            neg_samples_user, neg_samples_item = [], []
            for u in train_df['uidx'].unique():
                u_pos_items = pos_items[u]  # u_pos_items : set
                neg_items = list(set(range(self.args.n_items)) - u_pos_items)
                neg_samples = np.random.choice(neg_items, min(len(u_pos_items) * self.args.neg_ratio, len(neg_items)), replace=False)
                neg_samples_user.extend([u] * len(neg_samples))
                neg_samples_item.extend(neg_samples)
            print(f"train size after negative sampling: {len(train_df):,} --> {len(train_df)+len(neg_samples_user):,}")
            print(f"".ljust(60, '-'))

            train = {
                'user' : train_df['uidx'].tolist() + neg_samples_user,
                'item': train_df['iidx'].tolist() + neg_samples_item,
                'label' : [1] * len(train_df) + [0] * len(neg_samples_user)
                }
            prep_data = dict(
                train=train,
                test=test_df.groupby('uidx')['iidx'].unique().to_dict(),
                update_args=dict(n_users=n_users, n_items=n_items)
                )
            prep_path = self.get_prep_data_path()
            with open(prep_path, 'wb') as q:
                pickle.dump(prep_data, q)
            print(f"".ljust(60, '='))

    def load_ratings_path(self):
        ratings_path = self.args.ratings_path
        ratings = pd.read_csv(ratings_path, names=['user', 'item', 'rating', 'timestamp'], sep='::', engine='python')
        return ratings

    def get_prep_data_dir(self):
        if not os.path.isdir(self.args.prep_data_dir):
            os.mkdir(self.args.prep_data_dir)
        return self.args.prep_data_dir

    def get_prep_data_path(self):
        get_prep_data_dir = self.get_prep_data_dir()
        prep_path = os.path.join(get_prep_data_dir, self.args.prep_data_name)

        return prep_path

    def load_prep_data(self):
        with open(self.get_prep_data_path(), 'rb') as q:
            prep_data = pickle.load(q)

        return prep_data
        
