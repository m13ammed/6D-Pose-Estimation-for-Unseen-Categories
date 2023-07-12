#import gin
import sys 
import os
import torch
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

def C_from_sparse_P(P, Basis_1, Basis_2_inv):
    P_ = torch.zeros((Basis_2_inv.shape[-1], Basis_1.shape[0])).to(Basis_2_inv.device)
    P_[P[:,1],P[:,0]] = 1.
    C = Basis_2_inv @ P_ @ Basis_1
    #C = Basis_2_inv[:, P[:,0]]@Basis_1[P[:,1],:] #wrong Imlmenetation 
    return C
def shape_to_device(dict_shape, device):
    names_to_device = ["xyz", "faces", "mass", "evals", "evecs", "gradX", "gradY"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                if name in v.keys():
                    v[name] = v[name].to(device)
            dict_shape[k] = v
        else:
            if type(v) is list:
                for ii,vv in enumerate(v):
                    dict_shape[k][ii] = vv.to(device)

            else:
                dict_shape[k] = v.to(device)

    return dict_shape

@gin.configurable()
def train_net(model, criterion, optimizer, decay_iter, decay_factor, epochs):
    if torch.cuda.is_available() and os.environ["cuda"].lower() == 'true':
        device = torch.device(f'cuda:{os.environ["device"]}')
    else:
        device = torch.device("cpu")


    #save_dir_name = f'saved_models_{cfg["dataset"]["subset"]}'
    #model_save_path = os.path.join(base_path, f"data/{save_dir_name}/ep" + "_{}.pth")
    #if not os.path.exists(os.path.join(base_path, f"data/{save_dir_name}/")):
    #    os.makedirs(os.path.join(base_path, f"data/{save_dir_name}/"))

    # create dataset
    train_dataset = base_object_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=False)

    # define model
    model = model.to(device)
    optimizer = optimizer(model.parameters())
    criterion = criterion.to(device)

    # Training loop
    print("start training")
    iterations = 0
    for epoch in range(1, epochs + 1):
        if epoch % decay_iter == 0:
            #lr *= decay_factor
            #print(f"Decaying learning rate, new one: {lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']*decay_factor

        model.train()
        for i, (CAD, PC, Obj) in enumerate(train_loader):
            #if i != 1: continue
            print("start training")

            batch = {
                "shape1":CAD,
                "shape2":PC
            } 
            batch = shape_to_device(batch, device)
            #for k in Obj.keys():
            #    try:
            #        Obj[k] = Obj[k].to(device)
            #    except:
            #        continue
            #data = shape_to_device(data, device)

            # data augmentation
            #data = augment_batch(data, rot_x=30, rot_y=30, rot_z=60, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)

            # prepare iteration data
            map21 = Obj["P"]#[] #should be 21 confirm
            gt_partiality_mask12, gt_partiality_mask21 = Obj["overlap_12"].to(device), Obj["overlap_21"].to(device)
            
            # do iteration
            C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2, evecs_trans1, evecs_trans2 = model(batch)
            #C_gt = C_from_sparse_P(Obj["P"], PC["evecs"][:,:30], torch.linalg.pinv(CAD["evecs"][:,:30])) #data["C_gt"].unsqueeze(0)
            C_gt = C_from_sparse_P(Obj["P"], CAD["evecs"][:,:30], evecs_trans2).unsqueeze(0) #data["C_gt"].unsqueeze(0)

            loss = criterion(C_gt, C_pred, map21, use_feat1, use_feat2,
                             overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log
            iterations += 1
            if iterations % int(os.environ["log_interval"]) == 0:
                print(f"#epoch:{epoch}, #batch:{i + 1}, #iteration:{iterations}, loss:{loss}")

        # save model
        #if (epoch + 1) % os.environ["checkpoint_interval"] == 0:
        #    torch.save(dpfm_net.state_dict(), model_save_path.format(epoch))

if __name__ == '__main__':
    #gin.external_configurable(DPFMLoss)
    gin.external_configurable(torch.optim.Adam)
    gin.parse_config_file('/home/morashed/repo/6D-Pose-Estimation-for-Unseen-Categories/config/dpfm_orig.gin')
    utils.set_env_variables()
    
    train_net()
    #model = DPFMNet()

