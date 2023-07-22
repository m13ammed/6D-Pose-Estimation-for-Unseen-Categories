import torch
from torch import nn
from DPFM.dpfm.utils import FrobeniusLoss, WeightedBCELoss
import gin
import numpy as np
import torch.nn.functional as F

class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum((a - b) ** 2, axis=(1, 2))
        loss = torch.clamp(loss, min=-1, max=1000)
        return torch.mean(loss)

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
        #C12 = F.normalize(C12, p=2, dim=(1,2))
        #C_gt = F.normalize(C_gt, p=2, dim=(1,2))

        fmap_loss = self.frob_loss(C12, C_gt) * self.w_fmap
        loss += fmap_loss

        # overlap loss

        # nce loss
        if feat1.dim() ==3:
            nce_loss = 0
            acc_loss = 0
            #fmap_loss = 0
            if overlap_score12.dim() ==1:
                overlap_score12 = overlap_score12.unsqueeze(0)
                overlap_score21 = overlap_score21.unsqueeze(0)
            m = feat1.shape[0]
            for feat11, feat22, map212, o1, gtp1, o2, gtp2 in zip(feat1, feat2, map21,overlap_score12, gt_partiality_mask12.float(), overlap_score21, gt_partiality_mask21.float()):
                nce_loss += self.nce_softmax_loss(feat11, feat22, torch.Tensor(map212).type(torch.int)) * self.w_nce * 1/m

                acc_loss += self.binary_loss(o1, gtp1) * self.w_acc  * 1/m
                acc_loss += self.binary_loss(o2, gtp2) * self.w_acc  * 1/m
            loss += acc_loss
            loss += nce_loss

        else:
            nce_loss = self.nce_softmax_loss(feat1, feat2, map21) * self.w_nce

            acc_loss = self.binary_loss(overlap_score12, gt_partiality_mask12.float()) * self.w_acc
            acc_loss += self.binary_loss(overlap_score21, gt_partiality_mask21.float()) * self.w_acc
            loss += acc_loss

            loss += nce_loss

        return loss, {"nce_loss":nce_loss.item(),
                      "acc_loss":acc_loss.item(),
                      "fmap_loss":fmap_loss.item(),
                      "loss":loss.item()
                        }
