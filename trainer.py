from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from metrics import NDCG


class Trainer:
    def __init__(self, args, model, train_dataloader, test_data):
        print(f"Trainer Init".ljust(60, '='))
        self.args = args
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_data = test_data
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        print(self.device)
        print(f"".ljust(60, '='))

    def train(self):
        print(f"Trainer.train()".ljust(60, '-'))
        for k, v in vars(self.args).items():
            print(f"{k:>20} : {v:<}")
        print(f"{self.model}")
        self.model.to(self.device)

        for epoch in range(1, self.args.n_epoch+1):
            self.train_one_epoch(epoch)
            self.test()
        torch.save(self.model.state_dict(), self.args.model_save_path)
        print(f"".ljust(60, '-'))


    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        for user, item, label in self.train_dataloader:
            user = user.to(self.device)
            item = item.to(self.device)
            label = label.type(torch.FloatTensor).to(self.device).view(-1, 1)

            self.optimizer.zero_grad() 
            logits = self.model(user, item)
            loss = self.criterion(logits.view(-1, 1), label)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        print(f"epoch_loss average  : {epoch_loss/len(self.train_dataloader):.4f}")

    def test(self):
        self.model.eval()
        ndcg = NDCG(self.args.topk)
        actual_set_collect, pred_collect = [], []
        with torch.no_grad():
            for user in list(self.test_data.keys()):
                one_test_user = [user] * self.args.n_items
                one_test_user = torch.LongTensor(one_test_user).to(self.device)
                test_items = list(range(self.args.n_items))
                test_items = torch.LongTensor(test_items).to(self.device)
                logits = self.model(one_test_user, test_items)
                pred = logits.flatten().detach().cpu().numpy().argsort()[::-1][:self.args.topk]

                actual_set_collect.append(self.test_data[user])
                pred_collect.append(pred)
        ndcg.cal_ndcg_batch(actual_set_collect, pred_collect)
        print(f"NDCG : {ndcg.avg_ndcg:.4f}")

        # print(f"acc : {correct/total:.4f}")

    def get_criterion(self):
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        return criterion
    
    def get_optimizer(self):
        # SGD면 왜 학습이 안되는거야?
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer