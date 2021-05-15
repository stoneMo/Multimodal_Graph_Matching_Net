import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchsparse.nn as spnn

from models.IR.basic_blocks import SparseEncoder
from torchsparse import SparseTensor
from torchsparse.utils import sparse_quantize, sparse_collate, sparse_collate_tensors
from models.MGMN.basic_blocks import VisualEdgeConv, LangConv, GraphConvDist, GraphMatchDist

class GraphMatchModule(nn.Module):
    def __init__(self, input_feature_dim, args):
        super().__init__()

        self.args = args
        self.input_feature_dim = input_feature_dim

        v_dim = args.visual_dim
        h_dim = args.hidden_dim

        self.voxel_size = np.array([args.voxel_size_ap, args.voxel_size_ap, args.voxel_size_ap])

        self.net = SparseEncoder(self.input_feature_dim, v_dim)
        self.pooling = spnn.GlobalMaxPooling()
        self.vis_emb_fc = nn.Sequential(nn.Linear(v_dim, h_dim),
                                        nn.LayerNorm(h_dim),
                                        nn.ReLU(),
                                        nn.Linear(h_dim, h_dim),
                                        )
        self.lang_embed = LangConv(args, args.attr_embed_type, args.attr_aggr)

        self.gcn1 = VisualEdgeConv(h_dim + 3 + args.num_classes, h_dim, args=self.args, mode='full')
        self.gcn2 = VisualEdgeConv(h_dim + 3 + args.num_classes, h_dim, args=self.args, mode='full')
        if args.graph_match_op == 'gcd':
            self.gcn3 = VisualEdgeConv(h_dim + 3 + args.num_classes, h_dim, args=self.args, mode='part')
            self.gm = GraphConvDist(h_dim)
        elif args.graph_match_op == 'gmd':
            raise NotImplementedError()
        else:
            raise ValueError("Graph Matching operation should be either gcd or gmd")

        self.one_hot_array = np.eye(args.num_classes)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def filter_candidates(self, data_dict, lang_cls_pred):
        instance_points = data_dict['instance_points']
        pred_obb_batch = data_dict['pred_obb_batch']
        instance_obbs = data_dict['instance_obbs']
        batch_size = len(instance_points)

        batch_index = []
        pred_obbs = []
        feats = []
        filtered_index = []
        pts = []
        node_center_pad = []
        node_all_pad = []
        node_idx = []

        for i in range(batch_size):
            num_filtered_obj = len(pred_obb_batch[i])
            if num_filtered_obj < 2:
                continue

            instance_point = instance_points[i]
            instance_obb = instance_obbs[i]
            instance_class = data_dict['instance_class'][i]
            num_obj = len(instance_point)
            pred_obbs += list(instance_obb)

            node_center_batch = data_dict['parse_center_node_attr_embeddings'][i]
            node_tmp_idx = torch.where(data_dict['parse_batch_index']==i)
            node_all_batch = data_dict['parse_leaf_all_embeddings'][node_tmp_idx]

            # filter by class
            for j in range(num_obj):
                point_cloud = instance_point[j]
                onhot_semantic = self.one_hot_array[instance_class[j]]
                pc_feat = np.concatenate([instance_obb[j][:3], onhot_semantic], -1)
                # rep_cls = np.tile(onhot_semantic, [point_cloud.shape[0], 1])
                # point_cloud = np.concatenate([point_cloud, rep_cls], -1)

                c, f = sparse_quantize(
                    point_cloud[:, :3],
                    point_cloud,
                    quantization_size=self.voxel_size
                )
                pt = SparseTensor(f, c)
                pts.append(pt)
                
                feats.append(pc_feat)
                if instance_class[j] == lang_cls_pred[i]:
                    filtered_index.append(len(batch_index))
                    node_all_pad.append(node_all_batch)
                    node_idx.append(torch.ones(len(node_all_batch))*len(node_center_pad))
                    node_center_pad.append(node_center_batch)
                batch_index.append(i)
        # feats: mean of pc; batch_index: [0,0,1,1,1,...]; filter_index: [0,3,...] (ins_cls=lang_cls_pred)
        return feats, pts, batch_index, filtered_index, pred_obbs, node_center_pad, node_all_pad, node_idx

    def forward(self, data_dict):

        # lang encoding
        # lang_feats = data_dict['lang_rel_feats']  # (B, l_dim)
        # lang_feats = self.lang_emb_fc(lang_feats).unsqueeze(1)  # (B, 1, h_dim)

        # lang_len = data_dict["lang_len"]
        # lang_feats = data_dict["lang_feat"]
        # batch_size, max_len, _ = lang_feats.shape
        # lang_feats = self.lang_emb_fc(lang_feats.reshape([batch_size*max_len,-1])).reshape([batch_size,max_len,-1])
        data_dict = self.lang_embed(data_dict)
        if not self.args.use_gt_lang:
            lang_scores = data_dict["lang_scores"]
            lang_cls_pred = torch.argmax(lang_scores, dim=1)
        else:
            lang_cls_pred = data_dict['object_cat']

        feats, pts, batch_index, filtered_index, pred_obbs, center_node_attr, leaf_node_all, node_idx = \
            self.filter_candidates(data_dict, lang_cls_pred)

        if len(pred_obbs) == 0:
            data_dict['relation_scores'] = torch.zeros(0).cuda()
            return data_dict
            
        # prepare data feeding
        feats = torch.Tensor(feats).cuda()
        batch_index = torch.LongTensor(batch_index).cuda()
        filtered_index = torch.LongTensor(filtered_index).cuda()
        support_xyz = torch.Tensor(pred_obbs)[:, :3].cuda()
        center_node_attr = torch.stack(center_node_attr, dim=0).cuda()
        if self.args.graph_match_op == 'gcd':
            leaf_node_all = torch.cat(leaf_node_all, dim=0).cuda()
        node_idx = torch.cat(node_idx,dim=0).long().cuda()

        # concat sparse conv feat
        pts = sparse_collate_tensors(pts).cuda()
        pts = self.net(pts)
        pts = self.pooling(pts)  # (num_filtered_obj, dim)
        pts = self.vis_emb_fc(pts)
        gcnfeats = torch.cat([pts,feats],dim=1)
        # GCN
        gcnfeats = self.gcn1(support_xyz, batch_index, None, gcnfeats)
        gcnfeats = F.relu(gcnfeats)
        gcnfeats = self.gcn2(support_xyz, batch_index, None, torch.cat([gcnfeats,feats],dim=1))
        if self.args.graph_match_op == 'gcd':
            gcnfeats = F.relu(gcnfeats)
            gcnfeats = self.gcn3(support_xyz, batch_index, filtered_index, torch.cat([gcnfeats,feats],dim=1))
            scores = self.gm(center_node_attr, leaf_node_all, node_idx, gcnfeats)
        else:
            raise NotImplementedError()
        # feats = self.vis_emb_fc(feats)
        # feats = nn.functional.normalize(feats, p=2, dim=1)
        # scores = nn.functional.cosine_similarity(feats, lang_feats_flatten, dim=1)

        data_dict['relation_scores'] = scores
        # data_dict['relation_dist'] = torch.norm(feats-lang_feats_flatten, p=2, dim=1)
        return data_dict
