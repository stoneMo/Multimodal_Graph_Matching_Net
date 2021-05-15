import torch
import math
import torch.nn as nn
import torchsparse.nn as spnn

from torchsparse import SparseTensor
from torch_geometric.nn import MessagePassing, knn_graph, knn

# Hack per-node softmax in a stupid way, can use torch_scatter op to replace...
class CenterOp(MessagePassing):
    def __init__(self, aggr='add'):
        super(CenterOp,self).__init__(aggr=aggr)
        self.aggr = aggr

    def forward(self, x, edge_index):
        return self.propagate(edge_index,x=x)
    
    def message(self, x_i, x_j):
        if self.aggr == 'max':
            return x_j
        elif self.aggr == 'add':
            return torch.exp(x_j-x_i)
            

class CenterConv(MessagePassing):
    def __init__(self, F_out):
        super(CenterConv,self).__init__(aggr='add')
        self.rel_encoder = nn.Sequential(
            nn.Linear(3+3+3+1,64),
            nn.ReLU(),
            nn.Linear(64,F_out)
        )
        self.h_dim = F_out
    def forward(self, x, feat, atten, mask, attenmax, attensum, langfeat, edge_index):
        return self.propagate(edge_index, x=x, feat=feat, atten=atten, mask=mask, attenmax=attenmax, attensum=attensum, langfeat=langfeat)
    
    def message(self, x_i, x_j, feat_i, feat_j, atten_j, mask_i, attenmax_i, attensum_i, langfeat_j):
        # langfeat: E*length*num_feat
        atten_j = torch.exp(atten_j-attenmax_i)
        attention = (atten_j / attensum_i)*mask_i
        attention = attention / (attention.sum(-1,keepdim=True)+1e-7)
        langfeat_j = langfeat_j.reshape([mask_i.shape[0],mask_i.shape[1],-1])
        attention = attention.unsqueeze(-1)*langfeat_j
        attention = attention.sum(1)
        edge_weights = torch.cat([x_i,x_j,x_i-x_j,torch.norm(x_i-x_j,p=2,dim=-1,keepdim=True)],-1)
        edge_weights = self.rel_encoder(edge_weights)
        return feat_j*attention*edge_weights

class TARelationConv(nn.Module):
    def __init__(self, Fv_in, Fl_in, F_out, args, mode='full'):
        super().__init__()
        self.mode = mode
        self.args = args
        self.k = args.k
        self.feat_encoder = nn.Sequential(nn.Linear(Fv_in, F_out),
                                        nn.LayerNorm(F_out),
                                        nn.ReLU(),
                                        nn.Linear(F_out, F_out),
                                        )

        self.lang_encoder = nn.Sequential(nn.Linear(Fl_in, F_out),
                                         nn.BatchNorm1d(F_out),
                                         nn.ReLU(),
                                         nn.Linear(F_out, F_out),
                                         )
        # for numerical stability, will be failed otherwise
        self.cm = CenterOp('max')
        self.cs = CenterOp('add')
        self.csm = CenterConv(F_out)

    def forward(self, support_xyz, batch_index, filtered_index, features, lang_features, mask_flattened):
        if self.mode == 'part':
            assert filtered_index is not None
            # knn
            query_xyz = torch.index_select(support_xyz, 0, filtered_index)
            query_batch_index = torch.index_select(batch_index, 0, filtered_index)
            row, col = knn(support_xyz, query_xyz, self.k, batch_index, query_batch_index)
        elif self.mode == 'full':
            row, col = knn(support_xyz, support_xyz, self.k, batch_index, batch_index)
        else:
            raise ValueError("mode should be either part or full")
        edge_index = torch.stack([col, row], dim=0)
        
        features = self.feat_encoder(features)
        batch_size, max_len, _ = lang_features.shape
        lang_features = self.lang_encoder(lang_features.reshape([batch_size*max_len,-1])).reshape([batch_size,max_len,-1])
        lang_features_flattened = lang_features[batch_index]

        atten = torch.bmm(features.unsqueeze(1),lang_features_flattened.permute(0,2,1)).squeeze(1)

        if self.mode == 'part':
            attenmax = self.cm((atten,torch.index_select(atten, 0, filtered_index)),edge_index)
        else:
            attenmax = self.cm((atten,None),edge_index)
        attensum = self.cs((atten,attenmax),edge_index)

        lang_features_flattened = lang_features_flattened.reshape([batch_index.shape[0],-1])
        if self.mode == 'part':
            query_features = torch.index_select(features, 0, filtered_index)
            mask_flattened = torch.index_select(mask_flattened, 0, filtered_index)
            rel_residual = self.csm((support_xyz,query_xyz),(features,query_features),(atten,None),(None,mask_flattened),(None,attenmax),(None,attensum),(lang_features_flattened,None),edge_index)
            return query_features+rel_residual
        else:
            rel_residual = self.csm(support_xyz,features,(atten,None),(None,mask_flattened),(None,attenmax),(None,attensum), (lang_features_flattened,None), edge_index)
            return features+rel_residual