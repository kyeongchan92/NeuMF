import numpy as np


class NDCG:
    def __init__(self, topk):
        self.topk = topk
        self.ndcg = []
        self.n_sample = 0
        
    def cal_ndcg_batch(self, actual_set, pred):
        assert len(actual_set) == len(pred)

        for i in range(len(actual_set)):
            ndcg = self.cal_ndcg_one_sample(actual_set[i], pred[i])
            self.ndcg.append(ndcg)
            self.n_sample += 1

        self.avg_ndcg = sum(self.ndcg) / self.n_sample

    def cal_ndcg_one_sample(self, actual_set, pred):
        # idcg
        cal_num = min(self.topk, len(actual_set))
        idcg = sum([1.0 / np.log2(i + 1) for i in range(1, cal_num+1)])

        # dcg
        dcg = 0
        for i, item in enumerate(pred, start=1):
            if item in actual_set:
                dcg += 1.0 / np.log2(i + 1)
        # ndcg
        ndcg = dcg / idcg
        return ndcg