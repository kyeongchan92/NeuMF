import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, args):
        super().__init__()
        # hyper parameters
        n_user = args.n_user
        n_item = args.n_item
        emb_dim = args.emb_dim

        # GMF
        self.GMF_user = nn.Embedding(n_user, emb_dim)
        self.GMF_item = nn.Embedding(n_item, emb_dim)

        # MLP
        self.MLP_user = nn.Embedding(n_user, emb_dim)
        self.MLP_item = nn.Embedding(n_item, emb_dim)
        self.MLP_linear = nn.Sequential(
            nn.Linear(emb_dim*2, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, emb_dim//(2*2)),
            nn.ReLU(),
            nn.Linear(emb_dim//(2*2), emb_dim//(2*2*2)),
        )

        # output layer
        self.output_layer = nn.Linear(emb_dim + (emb_dim//(2*2*2)), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        # GMF
        phi_gmf = torch.mul(self.GMF_user(user), self.GMF_item(item))

        # MLP
        concat = torch.cat([self.MLP_user(user), self.MLP_item(item)], dim=1)
        phi_mlp = self.MLP_linear(concat)
        
        # fusion & output layer
        output = self.output_layer(torch.cat([phi_gmf, phi_mlp], dim=1))
        logit = self.sigmoid(output)

        return logit
