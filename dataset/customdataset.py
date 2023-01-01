from torch.utils.data import Dataset

class CustomTrainDataset(Dataset):
    def __init__(self, args, prepdataset):
        print(f"CustomTrainDataset Init".ljust(60, '='))
        self.args = args
        prep_data = prepdataset.load_prep_data()

        self.user = prep_data['train']['user']
        self.item = prep_data['train']['item']
        self.label = prep_data['train']['label']

        update_args = prep_data['update_args']
        self.args.n_user = update_args['n_user']
        self.args.n_item = update_args['n_item']
        print(f"".ljust(60, '='))

    def __getitem__(self, i):
        return self.user[i], self.item[i], self.label[i]
        
    def __len__(self):
        return len(self.user)

class CustomTestDataset(Dataset):
    def __init__(self, args, prepdataset):
        print(f"CustomTestDataset Init".ljust(60, '='))
        self.args = args
        self.test = prepdataset.load_prep_data()['test']
        '''
        self.test : {
            0 : np.array([1, 3, 4]),
            test user : used item array
        }
        '''
        self.test_users = list(self.test.keys())
        print(f"# of test users : {len(self.test_users)}")
        print(f"".ljust(60, '='))

    def __getitem__(self, index):
        test_user = self.test_users[index]
        return  test_user

    def __len__(self):
        return len(self.test_users)
