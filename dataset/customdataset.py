from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, args, prepdataset):
        print(f"CustomDataset Init".ljust(60, '='))
        self.args = args
        prep_data = prepdataset.load_prep_data()
        self.user = prep_data['user']
        self.item = prep_data['item']
        self.label = prep_data['label']
        self.args.n_user = prep_data['update_args']['n_user']
        self.args.n_item = prep_data['update_args']['n_item']
        print(f"".ljust(60, '='))

    def __getitem__(self, i):
        return self.user[i], self.item[i], self.label[i]
        
    def __len__(self):
        return len(self.user)
