#import gin
import sys 
import os
import torch
from torch.utils.tensorboard import SummaryWriter
if '__file__' in vars():
    print("We are running the script non interactively")
    path = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path.append(path)    
else:
    print('We are running the script interactively')
    sys.path.append("..")

#from utils import utils 

#from modeling.dpfm import DPFMNet
from models.dpfm import DPFMNet
from dataset import base_object_dataset
import utils
import gin.torch

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from pathlib import Path
def compute_inlier_ratio(pred_corr, CAD, PC_aligned, threshold):
    ''' given predicted correspondences, calculate inlier ratio. 
        input corr should be list of index pairs

general definition (def from geo tranformer):
        Inlier Ratio (IR) is the fraction of inlier matches among all putative point matches. 
        A match is considered as an inlier if the distance between the two points is smaller than thredhold = 10cm under the ground-truth transformation T

special case for us:
        if threshold is set to be the same for both gt corr and inlier ratio, then this function holds, otherwise should use a more general formulatoin.

    '''
    
    total_corr = len(pred_corr)
    if total_corr == 0: return 0

    CAD = CAD[pred_corr[:,0]]
    PC_aligned = PC_aligned[pred_corr[:,1]]

    sq_dist = torch.square(CAD - PC_aligned).sum(-1) ** 0.5

    inliers = (sq_dist<(threshold)).sum()
    inlier_ratio = inliers / total_corr

    return inlier_ratio

def fmap2pointmap(C12, evecs_x, evecs_y):
    """
    Convert functional map to point-to-point map

    Args:
        C12: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    if C12.dim() == 3:
        C12 = C12.squeeze(0)

    pp = nn_query(torch.matmul(evecs_x[:], C12.t()), evecs_y[:], -2)

    return pp
def nn_query(feat_x, feat_y, dim=-2):
    """
    Find correspondences via nearest neighbor query
    Args:
        feat_x: feature vector of shape x. [V1, C].
        feat_y: feature vector of shape y. [V2, C].
        dim: number of dimension
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2].
    """
    dist = torch.cdist(feat_x[:], feat_y[:]) #/ overlap_score12[:ov[0]].unsqueeze(1) # [V1, V2]
    K = 5
    _, idx = dist.sort(dim=-2)
    idx = idx.t()[:,:K]
    idx_p = torch.linspace(0,idx.shape[0]-1, idx.shape[0],device=idx.device).type(torch.int16).unsqueeze(1).repeat(1,K)

    p2p = torch.stack([idx, idx_p], 0).reshape(2,-1)
    
    return p2p

def euclidean_distance(tensor):
    # Calculate the squared Euclidean distances
    squared_distances = torch.sum((tensor[:, None] - tensor) ** 2, dim=2)
    
    # Take the square root to get the actual Euclidean distances
    euclidean_distances = torch.sqrt(squared_distances)
    
    return euclidean_distances

def spacial_filtering(CAD, PC, p_pred):
    B = euclidean_distance(PC[p_pred[1]])
    A = euclidean_distance(CAD[p_pred[0]])
    p_pred = p_pred[:,torch.absolute(A-B).mean(0)<0.3*Obj['diam_cad']]

    B = euclidean_distance(PC[p_pred[1]])
    A = euclidean_distance(CAD[p_pred[0]])
    p_pred = p_pred[:,torch.absolute(A-B).mean(0)<0.2*Obj['diam_cad']]

    B = euclidean_distance(PC[p_pred[1]])
    A = euclidean_distance(CAD[p_pred[0]])
    p_pred = p_pred[:,torch.absolute(A-B).mean(0)<0.08*Obj['diam_cad']]
    return p_pred

def shape_to_device(dict_shape, device):
    names_to_device = ["xyz", "faces", "mass", "evals", "evecs", "gradX", "gradY"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                if name in v.keys() and v[name] is not None:
                    v[name] = v[name].to(device)
            dict_shape[k] = v
        else:
            if type(v) is list:
                for ii,vv in enumerate(v):
                    dict_shape[k][ii] = vv.to(device)

            else:
                dict_shape[k] = v.to(device)

    return dict_shape
def collate(data):
    import numpy as np
    CAD, PC, Obj = {},{},{}

    for key in data[0][0].keys():
        if isinstance(data[0][0][key], np.ndarray):
            CAD.update({key : [torch.Tensor(d[0][key]) for d in data]})
            CAD[key] = torch.nn.utils.rnn.pad_sequence(CAD[key], batch_first=True)
        else:
            CAD[key] = None #torch.nn.utils.rnn.pad_sequence(CAD[key], batch_first=True)
            #CAD.update({key : [ torch.sparse_coo_tensor(idx, val, shape)   d[0][key] for d in data]})
        

    for key in data[0][2].keys():
        if isinstance(data[0][2][key], np.ndarray) and data[0][2][key].size>1:
            Obj.update({key : [torch.Tensor(d[2][key]) for d in data]})
            if key != 'P':
                Obj[key] = torch.nn.utils.rnn.pad_sequence(Obj[key], batch_first=True)
        else:
            Obj.update({key : [d[2][key] for d in data]})
            #Obj[key] = torch.Tensor(Obj[key])

    for key in data[0][1].keys():
        if isinstance(data[0][1][key], np.ndarray):
            PC.update({key : [torch.Tensor(d[1][key]) for d in data]})
            PC[key] = torch.nn.utils.rnn.pad_sequence(PC[key], batch_first=True)
        else:
            PC[key] = None #torch.nn.utils.rnn.pad_sequence(CAD[key], batch_first=True)

    return CAD, PC, Obj
def collate_noprocess(data):
    for b in data:
        to_del= ["L", "gradX", "gradY"]
        #print(b)
        for v in to_del:
            b[0].pop(v)
            b[1].pop(v)

    return data


from datetime import datetime
class TensorboardLogger:
    def __init__(self, log_dir,comment):
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir = log_dir[:-1]if log_dir[-1] == '/' else log_dir

        self.log_dir = log_dir + "/" + current_time + "_"+comment
        self.writer = None
        self.global_step = 0
        self.epoch = 1
    def log_summaries(self, summaries):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)
        for tag, value in summaries.items():
            self.writer.add_scalar(tag, value, self.global_step)
            self.global_step += 1
    def log_summaries_epoch(self, summaries, epoch):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)
        
        for tag in summaries[0].keys():
            value = torch.Tensor([v[tag] for v in summaries])
            value = torch.mean(value)
            self.writer.add_scalar(tag+"_epoch", value, epoch)
from tqdm import tqdm

score_I = 0
#if loss > 1e5:
#    print(Obj["obj_id"])
ppreds = []
IR = []
IR_PO = {}
path = Path('/home/unseen_object/data/tmp/results/')#yiz_obj_6
#path = Path('/home/unseen_object/data/tmp/results/initial_synth_4')
output = Path('/home/unseen_object/data/tmp/results/new_res_pt_3')
output.mkdir(exist_ok=True)
scores = 0
for i, pt in tqdm(enumerate(path.rglob('yiz_obj_*/*.pt'))):

    [CAD, PC, Obj] = torch.load(pt)
    PC['xyz'] = torch.Tensor(PC['xyz']).cuda()
    CAD['xyz'] = torch.Tensor(CAD['xyz']).cuda()
    CAD["evecs"] = torch.Tensor(CAD["evecs"]).cuda()
    PC["evecs"] = torch.Tensor(PC["evecs"]).cuda()
    Obj["C_pred"] = torch.Tensor(Obj["C_pred"]).cuda()
    Obj["align_pc"] = torch.Tensor(Obj["align_pc"]).cuda()

    p_pred = fmap2pointmap(Obj["C_pred"].squeeze(0), CAD["evecs"][:,:30], PC["evecs"][:,:30])


    p_pred =spacial_filtering(CAD['xyz'], PC['xyz'], p_pred) #p_pred[:,torch.absolute(A-B).mean(0)<0.08*Obj['diam_cad']]

    #ppreds.append(p_pred)
    score_I = compute_inlier_ratio(p_pred.t(), CAD['xyz'], Obj["align_pc"].cuda(), 0.1*Obj['diam_cad'])
    IR.append(score_I)
    try:
        score_I = score_I.cpu()   
    except:
        score_I = score_I
    if int(Obj["obj_id"]) in IR_PO.keys():
        
        IR_PO[int(Obj["obj_id"])].append(score_I)
    else:
        IR_PO.update({int(Obj["obj_id"]):[score_I]})
        Obj["p_pred"] = p_pred.cpu()
        Obj["ir"] = score_I.cpu()

    PC['xyz'] = torch.Tensor(PC['xyz']).cpu()
    CAD['xyz'] = torch.Tensor(CAD['xyz']).cpu()
    CAD["evecs"] = torch.Tensor(CAD["evecs"]).cpu()
    PC["evecs"] = torch.Tensor(PC["evecs"]).cpu()
    Obj["C_pred"] = torch.Tensor(Obj["C_pred"]).cpu()
    Obj["align_pc"] = torch.Tensor(Obj["align_pc"]).cpu()

    b = [CAD, PC, Obj]
    out_file = output / f'{i}_obj_{Obj["obj_id"]}.pt'
    torch.save(b, out_file)
#score_I = score_I/C_pred.shape[0]
    scores += score_I
    print(scores/(i+1))

import numpy as np
for key in IR_PO.keys():
    r = np.mean(IR_PO[key])
    print(f"obj {key}: {r}\n")

