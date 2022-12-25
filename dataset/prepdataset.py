import os
import pickle

import pandas as pd
import numpy as np

from arguments import update_dataset_args
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

            print(f"New indexing".ljust(60, '-'))
            uidx_map = {user:i for i, user in enumerate(set(ratings['user']))}
            iidx_map = {item:i for i, item in enumerate(set(ratings['item']))}
            print(f"".ljust(60, '-'))

            print(f"New mapping".ljust(60, '-'))
            ratings['uidx'] = ratings['user'].map(uidx_map)
            ratings['iidx'] = ratings['item'].map(iidx_map)

            n_user = ratings['uidx'].nunique()
            n_item = ratings['iidx'].nunique()
            self.args.n_user = n_user
            self.args.n_item = n_item
            print(f"".ljust(60, '-'))

            print(f"train test split".ljust(60, '-'))

            print(f"".ljust(60, '-'))

            print(f"Negative sampling for training".ljust(60, '-'))
            pos_items = ratings.groupby('uidx')['iidx'].agg(lambda x: set(x)).to_dict()
            neg_user, neg_item = [], []
            for u in range(self.args.n_user):
                u_pos_items = pos_items[u]
                neg_items = list(set(range(self.args.n_item)) - u_pos_items)
                neg_samples = np.random.choice(neg_items, min(len(u_pos_items) * self.args.neg_ratio, len(neg_items)), replace=False)
                neg_user.extend([u] * len(neg_samples))
                neg_item.extend(neg_samples)
            print(f"".ljust(60, '-'))

            train={'user':, 'item':, 'label':}
            test={'user':, 'item':, 'label':}
            prep_data = dict(user=ratings['uidx'].tolist() + neg_user,
            item=ratings['iidx'].tolist() + neg_item,
            label=[1] * len(ratings) + [0] * len(neg_user),
            update_args=dict(n_user=n_user, n_item=n_item)
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
        
