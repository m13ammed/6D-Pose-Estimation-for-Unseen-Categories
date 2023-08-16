
import torch
import gin

@gin.configurable()
def naive_fmap2pointmap(C12, evecs_x, evecs_y, **kwargs):
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
