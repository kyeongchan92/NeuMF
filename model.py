import torch
import torch.nn as nn
from torch.nn.init import normal_

class NeuMF(nn.Module):
    def __init__(self, args):
        super().__init__()
        # hyper parameters
        self.n_users = args.n_users
        self.n_items = args.n_items
        self.emb_dim = args.emb_dim

        # GMF
        self.GMF_user = nn.Embedding(self.n_users, self.emb_dim)
        self.GMF_item = nn.Embedding(self.n_items, self.emb_dim)

        # MLP
        self.MLP_user = nn.Embedding(self.n_users, self.emb_dim)
        self.MLP_item = nn.Embedding(self.n_items, self.emb_dim)
        self.MLP_linear = nn.Sequential(
            nn.Linear(self.emb_dim*2, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim//2),
            nn.ReLU(),
            nn.Linear(self.emb_dim//2, self.emb_dim//(2*2)),
        )

        # output layer
        self.output_layer = nn.Linear(self.emb_dim + (self.emb_dim//(2*2)), 1)
        # self.sigmoid = nn.Sigmoid()
        
        # initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, user, item):
        # GMF
        phi_gmf = torch.mul(self.GMF_user(user), self.GMF_item(item))

        # MLP
        concat = torch.cat([self.MLP_user(user), self.MLP_item(item)], dim=1)
        phi_mlp = self.MLP_linear(concat)
        
        # fusion & output layer
        output = torch.cat([phi_gmf, phi_mlp], dim=1)
        output = self.output_layer(output)
        # logit = self.sigmoid(output)

        return output

# class NeuMF(nn.Module):
#     """Neural Matrix Factorization Model
#         참고 문헌 : https://arxiv.org/abs/1708.05031

#     예시 :
#         model = NeuMF(cfg) 
#         output = model.forward(user_ids, item_ids, [feat0, feat1]) 
#     """
#     def __init__(self, cfg):
#         """ 
#         Args:
#             cfg : config 파일로 네트워크 생성에 필요한 정보들을 담고 있음 
#         """
#         super(NeuMF, self).__init__()
#         self.n_userss = cfg.n_users
#         self.n_itemss = cfg.n_items
#         self.emb_dim = cfg.emb_dim
#         self.layer_dim = cfg.layer_dim
#         # self.n_continuous_feats = 1
#         # self.n_genres = cfg.n_genres
#         self.dropout = 0.05
#         self.build_graph()

#     def build_graph(self):
#         """Neural Matrix Factorization Model 생성
#             구현된 모습은 위의 그림을 참고 
#         """
#         self.user_embedding_mf = nn.Embedding(num_embeddings=self.n_userss, embedding_dim=self.emb_dim)
#         self.item_embedding_mf = nn.Embedding(num_embeddings=self.n_itemss, embedding_dim=self.emb_dim)
        
#         self.user_embedding_mlp = nn.Embedding(num_embeddings=self.n_userss, embedding_dim=self.emb_dim)
#         self.item_embedding_mlp = nn.Embedding(num_embeddings=self.n_itemss, embedding_dim=self.emb_dim)
                
#         # self.genre_embeddig = nn.Embedding(num_embeddings=self.n_genres, embedding_dim=self.n_genres//2)
        
#         self.mlp_layers = nn.Sequential(
#             nn.Linear(2*self.emb_dim, self.layer_dim), 
#             nn.ReLU(), 
#             nn.Dropout(p=self.dropout), 
#             nn.Linear(self.layer_dim, self.layer_dim//2), 
#             nn.ReLU(), 
#             nn.Dropout(p=self.dropout)
#         )
#         self.affine_output = nn.Linear(self.layer_dim//2 + self.emb_dim, 1)
#         self.apply(self._init_weights)
        

#     def _init_weights(self, module):
#         if isinstance(module, nn.Embedding):
#             normal_(module.weight.data, mean=0.0, std=0.01)
#         elif isinstance(module, nn.Linear):
#             normal_(module.weight.data, 0, 0.01)
#             if module.bias is not None:
#                 module.bias.data.fill_(0.0)
    
#     def forward(self, user_indices, item_indices):
#         """ 
#         Args:
#             user_indices : 유저의 인덱스 정보 
#                 ex) tensor([ 3100,  3100,  ..., 14195, 14195])
#             item_indices : 아이템의 인덱스 정보
#                 ex) tensor([   50,    65,   ..., 14960, 11527])
#             feats : 특징 정보 
#         Returns: 
#             output : 유저-아이템 쌍에 대한 추천 결과 
#                 ex) tensor([  9.4966,  22.0261, ..., -19.3535, -23.0212])
#         """
#         user_embedding_mf = self.user_embedding_mf(user_indices)
#         item_embedding_mf = self.item_embedding_mf(item_indices)
#         mf_output = torch.mul(user_embedding_mf, item_embedding_mf)
        
#         user_embedding_mlp = self.user_embedding_mlp(user_indices)
#         item_embedding_mlp = self.item_embedding_mlp(item_indices)
#         # genre_embedding_mlp = self.genre_embeddig(feats[1])
#         input_feature = torch.cat((user_embedding_mlp, item_embedding_mlp), -1)
#         mlp_output = self.mlp_layers(input_feature)
        
#         output = torch.cat([mlp_output, mf_output], dim=-1)
#         output = self.affine_output(output).squeeze(-1)
#         return output