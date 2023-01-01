from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from metrics import NDCG


class Trainer:
    def __init__(self, args, model, train_datalaoder, test_data):
        print(f"Trainer Init".ljust(60, '='))
        self.args = args
        self.model = model
        self.train_datalaoder = train_datalaoder
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
        self.model.to(self.device)
        self.model.train()
        for epoch in range(1, self.args.n_epoch+1):
            pbar, loss = self.train_one_epoch(epoch)
            self.test(epoch, pbar, loss)
        torch.save(self.model.state_dict(), self.args.model_save_path)
        print(f"".ljust(60, '-'))


    def train_one_epoch(self, epoch):
        pbar = tqdm(self.train_datalaoder, unit='batches')
        loss_epoch = 0
        for user, item, label in pbar:
            self.optimizer.zero_grad() 
            user = user.to(self.device)
            item = item.to(self.device)
            logits = self.model(user, item)
            label = label.type(torch.FloatTensor).to(self.device)
            loss = self.criterion(logits.flatten(), label)
            # loss.requires_grad_(True)
            # print(loss)
            loss.backward()
            self.optimizer.step()
            pbar.set_description(f"epoch {epoch:2}")
            loss_epoch += loss.item()
        return pbar, loss_epoch / len(self.train_datalaoder)

    def test(self, epoch, pbar, loss):
        self.model.to(self.device)
        self.model.eval()
        ndcg = NDCG(self.args.topk)
        actual_set_collect, pred_collect = [], []
        with torch.no_grad():
            for user in list(self.test_data.keys()):
                one_test_user = [user] * self.args.n_item
                one_test_user = torch.LongTensor(one_test_user).to(self.device)
                test_items = list(range(self.args.n_item))
                test_items = torch.LongTensor(test_items).to(self.device)
                logits = self.model(one_test_user, test_items)
                pred = logits.flatten().detach().cpu().numpy().argsort()[::-1][:self.args.topk]

                actual_set_collect.append(self.test_data[user])
                pred_collect.append(pred)
        ndcg.cal_ndcg_batch(actual_set_collect, pred_collect)
        pbar.set_postfix(dict(loss=f"{loss:.4f}", ndcg=f"{ndcg.avg_ndcg:.4f}"))
        print(f"loss : {loss:.4f}, ndcg : {ndcg.avg_ndcg:.4f}")

        # print(f"acc : {correct/total:.4f}")

    def get_criterion(self):
        criterion = nn.BCELoss()
        return criterion
    
    def get_optimizer(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)
        return optimizer