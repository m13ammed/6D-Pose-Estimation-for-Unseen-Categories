import torch
import gin

@gin.configurable()
def spacial_filtering_fmap2pointmap(C12, evecs_x, evecs_y, CAD, PC, diam_cad):
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
    pp = spacial_filtering(CAD, PC, pp, diam_cad)
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

def spacial_filtering(CAD, PC, p_pred, diam_cad):
    B = euclidean_distance(PC[p_pred[1]])
    A = euclidean_distance(CAD[p_pred[0]])
    p_pred = p_pred[:,torch.absolute(A-B).mean(0)<0.3*diam_cad]

    B = euclidean_distance(PC[p_pred[1]])
    A = euclidean_distance(CAD[p_pred[0]])
    p_pred = p_pred[:,torch.absolute(A-B).mean(0)<0.15*diam_cad]

    #B = euclidean_distance(PC[p_pred[1]])
    #A = euclidean_distance(CAD[p_pred[0]])
    #p_pred = p_pred[:,torch.absolute(A-B).mean(0)<0.08*diam_cad]

    B = euclidean_distance(PC[p_pred[1]])
    A = euclidean_distance(CAD[p_pred[0]])


    if (torch.absolute(A-B).mean(0)<0.055*diam_cad).sum() == 0:
        p_pred = p_pred[:,torch.absolute(A-B).mean(0)<0.065*diam_cad]
        if (torch.absolute(A-B).mean(0)<0.065*diam_cad).sum() ==0:
            print("skipped")
    else:
        p_pred = p_pred[:,torch.absolute(A-B).mean(0)<0.055*diam_cad]

    return p_pred
