import os 
import gin
import numpy as np
import yaml 
import torch 

@gin.configurable
def set_env_variables(**kwargs):
    for key, value in kwargs.items():
        os.environ[key] = str(value)


def quaternion_rotation_matrix(Q):
    """
    source: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

@gin.configurable
def yaml_read(path):
    return yaml.safe_load(open(path, "r"))

@gin.configurable()
def prepare_train_datasets(datasets_list):
    if len(datasets_list) == 1 :
        return datasets_list[0]()
    else:
        new_list = [dataset() for dataset in datasets_list]
        return torch.utils.data.ConcatDataset(new_list)


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
