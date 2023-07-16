import os
import numpy as np
import open3d as o3d
import gin
import sys 
from dataset.object import base_object_dataset
from utils import utils

import copy
import time
import numpy as np
import open3d as o3d
import teaserpp_python

NOISE_BOUND = 0.05
N_OUTLIERS = 1700
OUTLIER_TRANSLATION_LB = 5
OUTLIER_TRANSLATION_UB = 10

def get_angular_error(R_exp, R_est):
    """
    Calculate angular error
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)));

def replace_idx_by_coo(idx, coo):
    '''
    This function replaces the indices with the coordinates

    Inputs:
    idx: np.array containing indices [N,] (i.e. column of P)
    coo: np.array containing coordinates (i.e. PC/CAD vertices) [M,3]

    Outputs:
    result: np.array containing corresponding coordinates [N,3]
    '''
    num_pts = idx.shape[0] # num of points
    num_coo = coo.shape[0] # num of coordinates
    
    # Create a new array to store the updated values
    result = np.empty([num_pts, 3]) #(N,3)
    
    # Iterate over each entry in P
    for i in range(num_pts): # go through all rows in P
        row_index = idx[i]  # Get the row index from the first column
        if row_index >= num_coo:
            raise ValueError("Index out of bounds")
        result[i] = coo[row_index]  # Replace entry with corresponding cad coordinates

    return result

def inject_incorrect_correspondences(P, M):
    N = P.shape[0]  # Number of existing correspondences
    num_vertices = np.max(P[:, 0]) + 1  # Number of vertices in CAD model
    num_points = np.max(P[:, 1]) + 1  # Number of points in point cloud
    
    # Generate M random incorrect correspondences
    incorrect_correspondences = np.random.randint(0, num_vertices, size=(M, 2))
    incorrect_correspondences[:, 1] = np.random.randint(0, num_points, size=M)
    
    # Concatenate the incorrect correspondences with the existing P matrix
    P_incorrect = np.concatenate((P, incorrect_correspondences), axis=0)
    
    return P_incorrect

def R_t_2_pose(R, t):
    '''
    This function constructs a 4x4 pose from R [3,3] and t [3,] matrices 
    '''
    T = np.empty((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, :] = [0, 0, 0, 1]
    return T

def transform(pcd, pose):
    '''
    This function transforms the pcd using the pose
    '''
    pcd_ = pcd @ pose[:3,:3].T
    pcd_ = pcd_ + pose[:3:,-1]
    return pcd_

def add(T_est, T_gt, pcd):
    '''
    This function computes the add score
    '''
    pts_est = transform(pcd, T_est)
    pts_gt = transform(pcd, T_gt)

    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e

if __name__ == "__main__":
    print("==================================================")
    print("        TEASER++ Python registration example      ")
    print("==================================================")
    
    # set config file
    gin.parse_config_file('config/example_dataset.gin')
    utils.set_env_variables()

    # load data from dataloader
    data = base_object_dataset()
    pcd = data[0]
    P = data[0][-1]['P']
    R_m2c = data[0][-1]['R_m2c']
    t_m2c = data[0][-1]['t_m2c']
    o = data[0][-1]['obj_id']

    '''
    # sparsify the P matrix (N,2) -> (S,2)
    np.random.shuffle(P)
    S = 1000 # num of correspondences
    P = P[:S,:]
    
    # inject M random incorrect correspondences into P
    M = 10
    P_injected = inject_incorrect_correspondences(P, M)
    np.random.shuffle(P_injected) # randomly shuffle the indices
    '''

    # load example P_pred
    P_pred = np.transpose(np.load('sample-data/sample_P_pred/p_i0.npy')) # (576, 2)

    # split P matrix into CAD and PC indices
    cad_idx = P_pred[:,0] # CAD indices [N,2]
    pc_idx = P_pred[:,1] # pc indices [M,2]
    #cad_idx = P_injected[:,0] # CAD indices [N,2]
    #pc_idx = P_injected[:,1] # pc indices [M,2]
    #cad_idx = P[:,0] # CAD indices [N,2]
    #pc_idx = P[:,1] # pc indices [M,2]

    # load CAD and PC coordinates
    cad_coo = data[0][0]['xyz'] # CAD coordinates [P,3]
    pc_coo = data[0][-1]['pcd_depth'] # pc coordinates [P,3]
    
    # replace indices by coordinates for Teaser++
    src = replace_idx_by_coo(cad_idx, cad_coo)
    dst = replace_idx_by_coo(pc_idx, pc_coo)
    
    # Transpose np array
    src = np.transpose(src)
    dst = np.transpose(dst)
        
    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    start = time.time()
    solver.solve(src, dst)
    end = time.time()

    solution = solver.getSolution()

    print("=====================================")
    print("          TEASER++ Results           ")
    print("=====================================")

    print("Expected rotation: ")
    print(R_m2c)
    print("Estimated rotation: ")
    print(solution.rotation)
    print("Error (rad): ")
    print(get_angular_error(R_m2c, solution.rotation))

    print("Expected translation: ")
    print(t_m2c)
    print("Estimated translation: ")
    print(solution.translation)
    print("Error (m): ")
    print(np.linalg.norm(t_m2c - solution.translation))

    print("Number of correspondences: ", src.shape[1])
    print("Number of outliers: ", N_OUTLIERS)
    print("Time taken (s): ", end - start)
    
    # Compute pred and gt pose (4x4) from R and t
    T_est = R_t_2_pose(solution.rotation, solution.translation)
    T_gt = R_t_2_pose(R_m2c, t_m2c)
  
    # Compute add score
    add_score = add(T_est, T_gt, cad_coo)
    print("add score:", add_score)
     
    # Visualization: Write CAD, CAD_transformed and pc to ply files
    
    #check if directory exists
    isExist = os.path.exists("poses")
    if not isExist:
        os.makedirs("poses")

    # CAD coordinates
    cad_03d = o3d.geometry.PointCloud()
    cad_03d.points = o3d.utility.Vector3dVector(cad_coo)
    o3d.io.write_point_cloud("poses/cad.ply", cad_03d)
    
    # CAD coordinates transformed (ESTIMATED)
    cad_transformed_est = transform(cad_coo, T_est)
    cad_o3d_transformed_est = o3d.geometry.PointCloud()
    cad_o3d_transformed_est.points = o3d.utility.Vector3dVector(cad_transformed_est)
    o3d.io.write_point_cloud("poses/cad_pose_est.ply", cad_o3d_transformed_est)
    #cad_transformed_est = cad_03d.transform(T_est)

    # CAD coordinates transformed (GROUND TRUTH)
    cad_transformed_gt = transform(cad_coo, T_gt)
    cad_o3d_transformed_gt = o3d.geometry.PointCloud()
    cad_o3d_transformed_gt.points = o3d.utility.Vector3dVector(cad_transformed_gt)
    o3d.io.write_point_cloud("poses/cad_pose_gt.ply", cad_o3d_transformed_gt)
    #cad_transformed_gt = cad_03d.transform(T_gt)

    # PC coordinates
    pc_03d = o3d.geometry.PointCloud()
    pc_03d.points = o3d.utility.Vector3dVector(pc_coo)
    o3d.io.write_point_cloud("poses/pc.ply", pc_03d)
