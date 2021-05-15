import torch
import math
import torch.nn as nn
import torchsparse.nn as spnn

from torchsparse import SparseTensor
from torch_geometric.nn import MessagePassing, knn_graph, knn
from models.MGMN.util import auction_lap

class VisualEdgeConv(MessagePassing):
    def __init__(self, F_in, F_out, args, mode='full'):
        super(VisualEdgeConv,self).__init__(aggr='max')
        self.args = args
        self.rel_encoder = nn.Sequential(
            nn.Linear(3+3+3+1+self.args.num_classes*2, 64),
            nn.ReLU(),
            nn.Linear(64, F_in)
        )
        self.vis_encoder = nn.Sequential(
            nn.Linear(3 * F_in, F_out),
            nn.ReLU(),
            nn.Linear(F_out, F_out)
        )
    def forward(self, support_xyz, batch_index, filtered_index, features):
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
        if self.mode == 'part':
            query_features = torch.index_select(features, 0, filtered_index)
            return self.propagate(edge_index, x=(support_xyz, query_xyz), feat=(features, query_features))
        elif self.mode == 'full':
            return self.propagate(edge_index, x=support_xyz, feat=features)
    
    def message(self, x_i, x_j, feat_i, feat_j):
        # langfeat: E*length*num_feat
        edge_weights = torch.cat([x_i,x_j,x_i-x_j,torch.norm(x_i-x_j,p=2,dim=-1,keepdim=True),feat_i[:, -self.args.num_classes:],feat_j[:, -self.args.num_classes:]],-1)
        edge_weights = self.rel_encoder(edge_weights)
        return self.vis_encoder(torch.cat([feat_i,edge_weights,feat_j]))

class LangConv(nn.Module):
    def __init__(self, args, embed_type='glove', aggr='max'):
        super().__init__()
        self.embed_type = embed_type
        self.aggr = aggr
        if embed_type == 'glove':
            # copied from lang_module
            self.word_projection = nn.Sequential(nn.Linear(args.embedding_size, args.word_output_dim),
                                                nn.ReLU(),
                                                nn.Dropout(args.word_dropout),
                                                nn.Linear(args.word_output_dim, args.word_output_dim),
                                                nn.ReLU())
            self.gru = nn.GRU(
                input_size=args.word_output_dim,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                batch_first=True,
                bidirectional=args.use_bidir,
            )
        elif embed_type == 'gru':
            self.word_projection = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(args.hidden_size, args.hidden_size),
                                                nn.ReLU())
        else:
            raise ValueError("Embed type either glove or gru")
        if aggr == 'attention':
            hidden_size = args.hidden_size if not self.use_bidir else args.hidden_size * 2
            self.fc_attribute = nn.Linear(hidden_size, 1)
        elif aggr != 'max':
            raise ValueError("Aggregation either attention or max-pooling")

    def _process_attr_embed(self, x, length=None):
        b,l,e = x.shape
        mask = (x.abs().sum(-1,keepdim=True)==0).float()
        x = self.word_projection(x.reshape([b*l,e])).reshape([b,l,e])
        if self.embed_type == 'glove':
            if length:
                raise NotImplementedError("Not support currently")
            else:
                x = self.gru(x)*mask
        if self.aggr == 'max':
            x = x.max(1)[0]
        elif self.aggr == 'attention':
            raise NotImplementedError("Not support currently")
        return x

    def forward(self, data_dict):
        data_dict['parse_center_node_attr_embeddings'] = data_dict['parse_center_node_attr_embeddings'].squeeze(1)
        data_dict['parse_center_node_attr_embeddings'] = self._process_attr_embed(data_dict['parse_center_node_attr_embeddings'])
        data_dict['parse_leaf_node_attr_embeddings'] = self._process_attr_embed(data_dict['parse_leaf_node_attr_embeddings'])
        data_dict['parse_edge_embeddings'] = self.word_projection(data_dict['parse_edge_embeddings'])
        data_dict['parse_leaf_node_embeddings'] = self.word_projection(data_dict['parse_leaf_node_embeddings'])
        data_dict['parse_leaf_all_embeddings'] = torch.cat([data_dict['parse_edge_embeddings'],data_dict['parse_leaf_node_embeddings'],data_dict['parse_leaf_node_attr_embeddings']],dim=-1)
        return data_dict

class LangEdgeConv(MessagePassing):
    def __init__(self, F_in, F_out):
        super(LangEdgeConv,self).__init__(aggr='max')
        self.encoder = nn.Sequential(
            nn.Linear(F_in, F_out),
            nn.ReLU(),
            nn.Linear(F_out, F_out)
        )
    def forward(self, x_i, x_j, edge_index):
        return x_i+self.propagate(edge_index, x=(x_j,x_i))
    def message(self, x_i, x_j):
        return self.encoder(x_j)

class GraphConvDist(nn.Module):
    def __init__(self, h_dim):
        super().__init__()    
        self.gcn = LangEdgeConv(h_dim*3, h_dim)

    def forward(self,center_node_attr, leaf_node_all, node_idx, gcnfeats):
        x_i = center_node_attr
        x_j = leaf_node_all
        edge_index = torch.stack([torch.arange(len(node_idx)).to(node_idx.device),node_idx],0)
        lang_gcnfeats = self.gcn(edge_index,x_i,x_j)
        return nn.functional.cosine_similarity(gcnfeats, lang_gcnfeats, dim=1)

class GraphMatchDist(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass