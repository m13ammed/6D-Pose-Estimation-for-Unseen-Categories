import sys 
import os
if '__file__' in vars():
    print("We are running the script non interactively")
    path = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path.append(path)    
else:
    print('We are running the script interactively')
    sys.path.append("..")
from modeling.dpfm import *
from DPFM.dpfm.diffusion_net.layers import DiffusionNet 
import gin 

@gin.configurable
class DPFMNet(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, cfg):
        super().__init__()

        # feature extractor
        self.feature_extractor = DiffusionNet(
            C_in=cfg["fmap"]["C_in"],
            C_out=cfg["fmap"]["n_feat"],
            C_width=64,
            N_block=2,
            dropout=False,
            with_gradient_features=False,
            with_gradient_rotations=True,
        )
        # cross attention refinement
        
        self.feat_refiner = CrossAttentionRefinementNet(n_in=cfg["fmap"]["n_feat"], num_head=cfg["attention"]["num_head"], gnn_dim=cfg["attention"]["gnn_dim"],
                                                        overlap_feat_dim=cfg["overlap"]["overlap_feat_dim"],
                                                        n_layers=cfg["attention"]["ref_n_layers"],
                                                        cross_sampling_ratio=cfg["attention"]["cross_sampling_ratio"],
                                                        attention_type=cfg["attention"]["attention_type"])

        # regularized fmap
        self.fmreg_net = RegularizedFMNet(lambda_=cfg["fmap"]["lambda_"], resolvant_gamma=cfg["fmap"]["resolvant_gamma"])
        self.n_fmap = cfg["fmap"]["n_fmap"]
        self.robust = cfg["fmap"]["robust"]

    def forward(self, batch):
        verts1, faces1, mass1, L1, evals1, evecs1, gradX1, gradY1 = (batch["shape1"]["xyz"], batch["shape1"]["faces"], batch["shape1"]["mass"],
                                                                     batch["shape1"]["L"], batch["shape1"]["evals"], batch["shape1"]["evecs"],
                                                                     batch["shape1"]["gradX"], batch["shape1"]["gradY"])
        verts2, mass2, L2, evals2, evecs2, gradX2, gradY2 = (batch["shape2"]["xyz"], batch["shape2"]["mass"],
                                                                     batch["shape2"]["L"], batch["shape2"]["evals"], batch["shape2"]["evecs"],
                                                                     batch["shape2"]["gradX"], batch["shape2"]["gradY"])
        L2, gradX2, gradY2 = None, None, None
        # set features to vertices
        features1, features2 = (verts1-110)/50, (verts2-110)/50 #orch.nn.functional.normalize(verts2, p=2, dim = -1)

        feat1 = self.feature_extractor(features1, mass1, L=L1, evals=evals1, evecs=evecs1,
                                       gradX=gradX1, gradY=gradY1, faces=faces1)#.unsqueeze(0)
        feat2 = self.feature_extractor(features2, mass2, L=L2, evals=evals2, evecs=evecs2,
                                       gradX=gradX2, gradY=gradY2)#.unsqueeze(0)

        # refine features
        ref_feat1, ref_feat2, overlap_score12, overlap_score21 = self.feat_refiner(verts1, verts2, feat1, feat2, batch)
        use_feat1, use_feat2 = (ref_feat1, ref_feat2) if self.robust else (feat1, feat2)
        # predict fmap
        torch.cuda.empty_cache()
        #evecs_trans1, evecs_trans2 = evecs1.t()[:self.n_fmap].cpu() @ torch.diag(mass1).cpu(), evecs2.t()[:self.n_fmap].cpu() @ torch.diag(mass2).cpu()
        if evecs1.dim() ==3:
            evecs_trans1, evecs_trans2 = [], []
            for e1,m1,e2,m2 in zip(evecs1, mass1, evecs2, mass2):
                evecs_trans1.append(torch.einsum('ij,i->ji', e1[:, :self.n_fmap], m1))
                evecs_trans2.append(torch.einsum('ij,i->ji', e2[:, :self.n_fmap], m2))
            evecs_trans1 = torch.stack(evecs_trans1)
            evecs_trans2 = torch.stack(evecs_trans2)
        else:
            evecs_trans1 = torch.einsum('ij,i->ji', evecs1[:, :self.n_fmap], mass1)
            evecs_trans2 = torch.einsum('ij,i->ji', evecs2[:, :self.n_fmap], mass2)
        if evals1.dim ==1:
            evals1, evals2 = evals1[:self.n_fmap], evals2[:self.n_fmap]
        else:
            evals1, evals2 = evals1[:,:self.n_fmap], evals2[:,:self.n_fmap]
        C_pred = self.fmreg_net(use_feat1, use_feat2, evals1, evals2, evecs_trans1, evecs_trans2)

        return C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2, ref_feat1, ref_feat2
