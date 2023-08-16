import torch
from torch.utils.tensorboard import SummaryWriter
#import gin
import sys 
import os
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
from dataset import *
import utils
import gin.torch
from fmap2pointmap_solvers import *

import os

@gin.configurable()
def train_net(model, criterion, optimizer, decay_iter, decay_factor, epochs, logging_dir, comment=""):
    if torch.cuda.is_available() and os.environ["cuda"].lower() == 'true':
        device = torch.device(f'cuda:{os.environ["device"]}')
    else:
        device = torch.device("cpu")
    idx = 0

    IR_PO = {}
    # create dataset
    #dataset = base_object_dataset(render_data_name = "lmo",obj_take=[5,6,8,12,11], mode = 'test') #)[1,2,3,4,14,10,6,12,9,11]
    #dataset = base_object_dataset(render_data_name = "lm1k",obj_take=[5,6,8,12,11]) #)[1,2,3,4,14,10,6,12,9,11]
    with gin.config_scope('eval'):
        dataset = base_object_dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(os.environ["BS"]), shuffle=False, collate_fn = collate, drop_last = False, num_workers = int(os.environ["num_workers"]))
    loader_ = iter(torch.utils.data.DataLoader(dataset, batch_size=int(os.environ["BS"]), shuffle=False, collate_fn = collate_noprocess, drop_last = False, num_workers = int(os.environ["num_workers"]))) 

    # define model
    if "pretrained_model" in os.environ.keys():
        if os.environ["pretrained_model"].lower() != "none":
            state_dict = torch.load(os.environ["pretrained_model"]) 
            model.load_state_dict(state_dict)
    else:
        raise Exception("no models found") 
    model = model.to(device)

    criterion = criterion.to(device)
    fmap2pointmap_solver = choose_fmap2pointmap_solver()
    # Training loop
    print("start eval")
    iterations = 0
    losses = []
    model.eval()
    scores = 0
    for j, (CAD, PC, Obj) in enumerate(loader):

        batch = {
            "shape1":CAD,
            "shape2":PC
        } 
        batch = shape_to_device(batch, device)


        # prepare iteration data
        map21 = Obj["P"]
        gt_partiality_mask12, gt_partiality_mask21 = Obj["overlap_12"].to(device), Obj["overlap_21"].to(device)
        
        # do iteration
        C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2, evecs_trans1, evecs_trans2 = model(batch)
        C_gt = torch.stack([utils.C_from_sparse_P(p.type(torch.int), cad[:,:30], pc[:,:30]) for p,cad,pc in zip(Obj["P"], CAD["evecs"], PC["evecs"])])

        loss, log_loss = criterion(C_gt, C_pred, map21, use_feat1, use_feat2,
                        overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)

        score_I = 0
        ppreds = []
        IR = []
        batches = next(loader_)
        for i in range(C_pred.shape[0]):
            
            non_zero_indices_CAD = torch.any(CAD["xyz"][i] != 0, dim=1)
            non_zero_indices_PC = torch.any(PC["xyz"][i] != 0, dim=1)
            p_pred = fmap2pointmap_solver(C_pred[i], CAD["evecs"][i][non_zero_indices_CAD,:30], PC["evecs"][i][non_zero_indices_PC,:30], CAD =  CAD['xyz'][i], PC = PC['xyz'][i], diam_cad = Obj['diam_cad'][i]) #spacial_filtering_fmap2pointmap(C_pred[i], CAD["evecs"][i][:,:30], PC["evecs"][i][:,:30]) #fmap2pointmap(C_pred[i], CAD["evecs"][i][:,:30], PC["evecs"][i][:,:30])
            ppreds.append(p_pred)
            score_Ii = utils.compute_inlier_ratio(p_pred.t(), CAD['xyz'][i], Obj["align_pc"][i].to(device), 0.1*Obj['diam_cad'][i])
            score_I += score_Ii
            IR.append(score_Ii)
            try:
                score_Ii = score_Ii.cpu()   
            except:
                score_Ii = score_Ii
            if int(Obj["obj_id"][i]) in IR_PO.keys():
                
                IR_PO[int(Obj["obj_id"][i])].append(score_Ii)
            else:
                IR_PO.update({int(Obj["obj_id"][i]):[score_Ii]})
        scores += score_I

        # log
        iterations += 1
        if iterations % int(os.environ["log_interval"]) == 0:
            print(f"##batch:{i + 1}, #iteration:{iterations}, loss:{loss}, IR: {score_I/ (i + 1)}")
            print(f"IR average: {scores /  (i*(j+1)) }")

        
        for  (b, ppp,ccc, ir) in zip(batches, ppreds,C_pred, IR):
            #print(ppp)
            b[-1].update({"p_pred": ppp,
                           "C_pred": ccc,
                           'ir':ir})

            if "save_results" in os.environ.keys():
                if os.environ["save_results"].lower() != "none":
                    os.makedirs(os.environ["save_results"], exist_ok=True) 
                    torch.save(b, os.path.join(os.environ["save_results"], f"{idx}_obj_{b[-1]['obj_id']}.pt"))
            idx +=1
        #break
    print(f"IR average: {scores / len(dataset)}")
    import numpy as np
    for key in IR_PO.keys():
        r = np.mean(IR_PO[key])
        print(f"obj {key}: {r}\n")


if __name__ == '__main__':
    gin.external_configurable(torch.optim.RMSprop) 
    gin.external_configurable(torch.optim.Adam) 
    try:
        gin.parse_config_file('./config/dpfm_orig.gin')
    except:
        gin.parse_config_file('../config/dpfm_orig.gin')
    utils.set_env_variables()
    
    train_net()

