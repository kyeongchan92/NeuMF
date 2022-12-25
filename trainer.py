from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, args, model, train_datalaoder, test_dataloader):
        self.args = args
        self.model = model
        self.train_datalaoder = train_datalaoder
        self.test_dataloader = test_dataloader
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()

    def train(self):
        self.model.train()
        for epoch in range(1, self.args.n_epoch+1):
            pbar = self.train_one_epoch(epoch)
            self.test(epoch, pbar)

    def get_criterion(self):
        criterion = nn.BCELoss()
        return criterion
    
    def get_optimizer(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def train_one_epoch(self, epoch):
        pbar = tqdm(self.train_datalaoder, unit='batches')
        for user, item, label in pbar:
            self.optimizer.zero_grad() 
            logits = self.model(user, item)
            label = label.type(torch.FloatTensor)
            loss = self.criterion(logits.flatten(), label)
            loss.backward()
            self.optimizer.step()
            pbar.set_description(f"epoch {epoch:2}")
        return pbar

    def test(self, pbar, epoch):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for user, item, label in tqdm(self.test_dataloader):
                logits = model(user, item)
                label = label.type(torch.FloatTensor)

                correct += torch.sum((logits.flatten()>=0.5) == label).item()
                total += len(label)
        pbar.set_postfix(acc=correct/total)

        # print(f"acc : {correct/total:.4f}")
