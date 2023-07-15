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

#def C_from_sparse_P(P, Basis_1, Basis_2_inv):
#    P_ = torch.zeros((Basis_2_inv.shape[-1], Basis_1.shape[0])).to(Basis_2_inv.device)
#    P_[P[:,1],P[:,0]] = 1.
#    C = Basis_2_inv @ P_ @ Basis_1
#    #C = Basis_2_inv[:, P[:,0]]@Basis_1[P[:,1],:] #wrong Imlmenetation 
#    return C
def C_from_sparse_P(P, evecs1, evecs2):
    '''vts1, vts2 is a list of correspondeing point pair indexes
    this should be the general form of solving for functional mappings
    return the optimal functional mapping between two shapes
    intuition(my word): given the specified landmark correspondences, solve for the best fitted FM under those constraints
    
    xieyizheng
    '''
    #align the two basis
    evec_1_a, evec_2_a = evecs1[P[:,0]], evecs2[P[:,1]]
    #solve the linear system: AX = B, phi_2_a @ C = phi_1_a, aligned_number_of_pointsx50, 50x50 = aligned_number_of_pointsx50
    C_gt = torch.linalg.lstsq(evec_2_a, evec_1_a)[0][:evec_1_a.size(-1)]
    #this is the best functional mapping given the provided p2p corredpondence information
    return C_gt
def compute_inlier_ratio(pred_corr, gt_corr, gt_T=None, threshold=0.02):
    ''' given predicted correspondences, calculate inlier ratio. 
        input corr should be list of index pairs

general definition (def from geo tranformer):
        Inlier Ratio (IR) is the fraction of inlier matches among all putative point matches. 
        A match is considered as an inlier if the distance between the two points is smaller than thredhold = 10cm under the ground-truth transformation T

special case for us:
        if threshold is set to be the same for both gt corr and inlier ratio, then this function holds, otherwise should use a more general formulatoin.

    '''
    inliers = 0
    total_corr = len(pred_corr)
    if total_corr == 0: return 0
    for p, q in pred_corr:
        # general definition of inlier ratio
        
        #transformed_p = ground_truth_T(p)
        #distance = np.linalg.norm(transformed_p - q)
        #if distance < tau_1:
        #    inliers += 1

        # special case for us
        if torch.Tensor([p,q]) in gt_corr:
            inliers += 1
    inlier_ratio = inliers / total_corr

    return inlier_ratio
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

    pp = nn_query(torch.matmul(evecs_x, C12.t()), evecs_y)
    return torch.stack([pp, torch.linspace(0,pp.shape[0]-1, pp.shape[0],device=pp.device).type(torch.int16)], 0)

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
    dist = torch.cdist(feat_x, feat_y)  # [V1, V2]
    p2p = dist.argmin(dim=dim)
    return p2p
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
            if Obj["obj_id"] >= 11 : continue
            print("start training")

            batch = {
                "shape1":CAD,
                "shape2":PC
            } 
            batch = shape_to_device(batch, device)

            # data augmentation
            #data = augment_batch(data, rot_x=30, rot_y=30, rot_z=60, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1)

            # prepare iteration data
            map21 = Obj["P"]#[] #should be 21 confirm
            gt_partiality_mask12, gt_partiality_mask21 = Obj["overlap_12"].to(device), Obj["overlap_21"].to(device)
            
            # do iteration
            C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2, evecs_trans1, evecs_trans2 = model(batch)
            #C_gt = C_from_sparse_P(Obj["P"], PC["evecs"][:,:30], evecs_trans2) #data["C_gt"].unsqueeze(0)
            C_gt = C_from_sparse_P(Obj["P"], CAD["evecs"][:,:30], PC["evecs"][:,:30]).unsqueeze(0) #data["C_gt"].unsqueeze(0)

            loss = criterion(C_gt, C_pred, map21, use_feat1, use_feat2,
                             overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)
            p_pred = fmap2pointmap(C_pred, CAD["evecs"][:,:30], PC["evecs"][:,:30])
            score_I = compute_inlier_ratio(p_pred.t(), CAD['xyz'], Obj["align_pc"].to(device), 0.1*Obj['diam_cad'])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #   compute_inlier_ratio()
            # log
            iterations += 1
            if iterations % int(os.environ["log_interval"]) == 0:
                print(f"#epoch:{epoch}, #batch:{i + 1}, #iteration:{iterations}, loss:{loss}, IR: {score_I}")
                model.eval()
        for i, (CAD, PC, Obj) in enumerate(train_loader):
            if Obj["obj_id"] < 10 : continue

            print("start training")

            batch = {
                "shape1":CAD,
                "shape2":PC
            } 
            batch = shape_to_device(batch, device)
            # prepare iteration data
            map21 = Obj["P"]#[] #should be 21 confirm
            gt_partiality_mask12, gt_partiality_mask21 = Obj["overlap_12"].to(device), Obj["overlap_21"].to(device)
            
            # do iteration
            C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2, evecs_trans1, evecs_trans2 = model(batch)
            #C_gt = C_from_sparse_P(Obj["P"], PC["evecs"][:,:30], torch.linalg.pinv(CAD["evecs"][:,:30])) #data["C_gt"].unsqueeze(0)
            #C_gt = C_from_sparse_P(Obj["P"], CAD["evecs"][:,:30], evecs_trans2).unsqueeze(0) #data["C_gt"].unsqueeze(0)
            C_gt = C_from_sparse_P(Obj["P"], CAD["evecs"][:,:30], PC["evecs"][:,:30]).unsqueeze(0) #data["C_gt"].unsqueeze(0)

            loss = criterion(C_gt, C_pred, map21, use_feat1, use_feat2,
                            overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)

            #loss.backward()
            #optimizer.step()
            #optimizer.zero_grad()
            p_pred = fmap2pointmap(C_pred, CAD["evecs"][:,:30], PC["evecs"][:,:30])
            score_I = compute_inlier_ratio(p_pred.t(), CAD['xyz'], Obj["align_pc"].to(device), 0.1*Obj['diam_cad'])

            # log
            iterations += 1
            if iterations % int(os.environ["log_interval"]) == 0:
                print(f"#epoch:{epoch}, #batch:{i + 1}, #iteration:{iterations}, val loss:{loss}, IR: {score_I}")

        # save model
        #if (epoch + 1) % os.environ["checkpoint_interval"] == 0:
        #    torch.save(dpfm_net.state_dict(), model_save_path.format(epoch))

if __name__ == '__main__':
    #gin.external_configurable(DPFMLoss)
    gin.external_configurable(torch.optim.Adam)
    gin.parse_config_file('/home/morashed/repo/6D-Pose-Estimation-for-Unseen-Categories/config/dpfm_orig.gin')
    utils.set_env_variables()
    
    train_net()

