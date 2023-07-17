import torch
from torch import nn
from DPFM.dpfm.utils import FrobeniusLoss, WeightedBCELoss
import gin
import numpy as np
import torch.nn.functional as F


class NCESoftmaxLoss(nn.Module):
    def __init__(self, nce_t, nce_num_pairs):
        super().__init__()
        self.nce_t = nce_t
        self.nce_num_pairs = nce_num_pairs
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, features_1, features_2, map21):
        features_1, features_2 = features_1.squeeze(0), features_2.squeeze(0)

        if map21.shape[0] > self.nce_num_pairs:
            selected = np.random.choice(map21.shape[0], self.nce_num_pairs, replace=False)
        else:
            selected = torch.arange(map21.shape[0])

        features_1, features_2 = F.normalize(features_1, p=2, dim=-1), F.normalize(features_2, p=2, dim=-1)

        #query = features_1[map21[selected]]
        #keys = features_2[selected]
        query = features_1[map21[selected][:,0]]
        keys = features_2[map21[selected][:,1]]
        logits = - torch.cdist(query, keys)
        logits = torch.div(logits, self.nce_t)
        labels = torch.arange(selected.shape[0]).long().to(features_1.device)
        loss = self.cross_entropy(logits, labels)
        return loss
    
@gin.configurable
class DPFMLoss(nn.Module):
    def __init__(self, w_fmap=1, w_acc=1, w_nce=0.1, nce_t=0.07, nce_num_pairs=4096):
        super().__init__()

        self.w_fmap = w_fmap
        self.w_acc = w_acc
        self.w_nce = w_nce

        self.frob_loss = FrobeniusLoss()
        self.binary_loss = WeightedBCELoss()
        self.nce_softmax_loss = NCESoftmaxLoss(nce_t, nce_num_pairs)

    def forward(self, C12, C_gt, map21, feat1, feat2, overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21):
        loss = 0

        # fmap loss
        fmap_loss = self.frob_loss(C12, C_gt) * self.w_fmap
        loss += fmap_loss

        # overlap loss
        acc_loss = self.binary_loss(overlap_score12, gt_partiality_mask12.float()) * self.w_acc
        acc_loss += self.binary_loss(overlap_score21, gt_partiality_mask21.float()) * self.w_acc
        loss += acc_loss

        # nce loss
        if feat1.dim() ==3:
            nce_loss = 0
            m = feat1.shape[0]
            for feat11, feat22, map212 in zip(feat1, feat2, map21):
                nce_loss += self.nce_softmax_loss(feat11, feat22, map212) * self.w_nce * 1/m
        else:
            nce_loss = self.nce_softmax_loss(feat1, feat2, map21) * self.w_nce
        loss += nce_loss

        return loss
