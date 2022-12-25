from torch.utils.data import Dataset, DataLoader, random_split

from arguments import args
from dataset.prepdataset import ML1mDataset
from dataset.customdataset import CustomDataset
from model import NeuMF
from trainer import Trainer


if __name__ == '__main__':
    ml1m_dataset = ML1mDataset(args)
    ml1m_dataset.preprocess()
    dataset = CustomDataset(args, ml1m_dataset)
    for k, v in vars(args).items():
        print(f"{k:>20} : {v:<}")

    n_sample = len(dataset)
    n_test = int(n_sample*(args.test_ratio))
    n_train = n_sample - n_test
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])  # del dataset? memory problem
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = NeuMF(args)
    trainer = Trainer(args, model, train_dataloader, test_dataloader)
    trainer.train()

    print('Done.')