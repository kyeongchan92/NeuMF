from torch.utils.data import Dataset, DataLoader, random_split

from arguments import args
from dataset.prepdataset import ML1mDataset
from dataset.customdataset import CustomTrainDataset, CustomTestDataset
from model import NeuMF
from trainer import Trainer


if __name__ == '__main__':
    # preprocess
    ml1m_dataset = ML1mDataset(args)
    ml1m_dataset.preprocess()

    # dataset
    train_dataset = CustomTrainDataset(args, ml1m_dataset)
    test_data = ml1m_dataset.load_prep_data()['test']

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = NeuMF(args)
    trainer = Trainer(args, model, train_dataloader, test_data)
    trainer.train()

    print('Done.')