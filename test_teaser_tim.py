'''
                        -------------- TEASER++ registration Evaluation Script ---------------    
                                                
                                                Author: Tim Strohmeyer
                                                Date: 25. July 2023 
                                                Project: 6D Pose Estimation of Unseen Categories   


This script performs 3D registration using TEASER++ to estimate the poses based on the predicted point2point Correspondence 
Matrix, as well as evaluates the pose restuls in terms of ADD, ADDs and percentage of correct poses. 
The scripts:
1. loads the resulting data files (.pt) from the DPFM pipeline 
2. loads the predicted Correspondence Matrix (P_red)
3. restructures the correspondences into TEASER++ readible format (src, dst) by replacing the indices with corresponding 
   coordinates.
4. runs TEASER++ registration (pose estimation)
5. Evaluates pose based on ADD, ADDs, 
6. runs ICP to improve estimated pose further
7. Reevaluates pose based on ADD, ADDs
8. Computes the average scores per object ID
9. Computes transformations on CAD and Point CLouds and writes them to ply files
10. Writes results into "result_poses_TEASER" folder of structure:
   result_poses_TEASER
     -> ply
        -> obj_[obj_id]_result_[index]  
          -> cad_[index]_pose_est.ply: CAD vertices trasnformed by TEASER+ICP estimated Pose
          -> cad_[index]_pose_gt.ply:  CAD vertices trasnformed by TEASER+ICP ground truth Pose
          -> cad_[index].ply:          CAD vertices
          -> pc_[index].ply:           partial Point Cloud
     -> results
       -> obj_[obj_id]_result_[index].txt: txt files containing evaluated scores 
     -> avg_results.txt: txt files containing average evaluated scores per object class   
'''

import os
import numpy as np
import open3d as o3d
import gin
import sys 
import copy
import time
import numpy as np
import open3d as o3d
import teaserpp_python
import torch
import json
import statistics

from numpy import linalg as LA
from dataset.object import base_object_dataset
from utils import utils
from sklearn.neighbors import KDTree
from scipy.linalg import logm
from tqdm import tqdm

def create_named_lists(n, score):
    lists_container = {}

    for obj_id in range(1, n + 1):
        list_name = "obj_{}_{}".format(obj_id, score)
        new_list = []  # Create an empty list
        lists_container[list_name] = new_list

    return lists_container

def calculate_average(lst):
    if not lst:  # Check if the list is empty
        return 0
    avg = float(sum(lst) / len(lst))

    return avg

def get_angular_error(R_exp, R_est):
    """
    Calculate angular error
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)));

def rad_2_deg(rad):
    deg = rad * (180/np.pi)
    return deg

def replace_idx_by_coo(idx, coo):
    '''
    This function replaces the indices with the coordinates

    Inputs:
    idx: np.array containing indices [N,] (i.e. column of P)
    coo: np.array containing coordinates (i.e. PC/CAD vertices) [M,3]

    Outputs:
    result: np.array containing corresponding coordinates [N,3]
    '''
    num_corr = idx.shape[0] # num of correspondences
    num_coo = coo.shape[0] # num of coordinates
    
    # Create a new array to store the updated values
    #result = np.zeros([num_corr, 3]) # (1000,3)
    result_list = []
    
    # Iterate over each entry in P
    for i in range(num_corr): # go through all rows in P
        row_index = idx[i]  # Get the row index of P
        if row_index >= num_coo:
            continue
            #raise ValueError("Index out of bounds")
        #result[i] = coo[row_index]  # Replace entry with corresponding cad coordinates
        result_list.append(coo[row_index])
    print(idx)
    
    if not result_list:
        return 0
    else:
        result = np.stack(result_list, axis=0)
    
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

def pose_2_R_t(T):
    '''
    This function constructs R [3,3] and t [3,] matrices from a 4x4 pose 
    '''
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t

    return R, t

def transform(pcd, pose):
    '''
    This function transforms the pcd using the pose
    '''
    pcd_ = pcd @ pose[:3,:3].T
    pcd_ = pcd_ + pose[:3:,-1]
    return pcd_

def add(T_est, T_gt, pcd, diameter, percentage=0.1):
    '''
    This function computes the add score
    '''
    pts_est = transform(pcd, T_est)
    pts_gt = transform(pcd, T_gt)

    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    threshold = diameter * percentage
    score = (e < threshold)
    score = int(score)
    return e, score

def add_score(T_est, T_gt, pcd):
    '''
    This function computes the add score
    '''
    pts_est = transform(pcd, T_est)
    pts_gt = transform(pcd, T_gt)

    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()

    return e
    
def compute_add_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
    R_gt, t_gt = pose_2_R_t(pose_gt)
    R_pred, t_pred = pose_2_R_t(pose_pred)
    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        pts_xformed_gt = R_gt[i].reshape((1, 3)).dot(pts3d.transpose()) + t_gt[i]
        #pts_xformed_gt = R_gt[i].reshape((1, 3)) * pts3d.transpose() + t_gt[i] # (1,3)*(3,5000)
        pts_xformed_pred = R_pred[i].reshape((1, 3)).dot(pts3d.transpose()) + t_pred[i]    
        #pts_xformed_pred = R_pred[i].reshape((1, 3)) * pts3d.transpose() + t_pred[i]
        distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
        mean_distances[i] = np.mean(distance)            

    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score

def compute_adds_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
    R_gt, t_gt = pose_2_R_t(pose_gt)
    R_pred, t_pred = pose_2_R_t(pose_pred)

    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        if np.isnan(np.sum(t_pred[i])):
            mean_distances[i] = np.inf
            continue
        pts_xformed_gt = R_gt[i].reshape((1, 3)).dot(pts3d.transpose()) + t_gt[i]
        #pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
        pts_xformed_pred = R_pred[i].reshape((1, 3)).dot(pts3d.transpose()) + t_pred[i] 
        #pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        kdt = KDTree(pts_xformed_gt.transpose(), metric='euclidean')
        distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
        mean_distances[i] = np.mean(distance)
    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score

def compute_pose_error(diameter, pose_gt, pose_pred):
    R_gt, t_gt = pose_2_R_t(pose_gt)
    R_pred, t_pred = pose_2_R_t(pose_pred)

    count = R_gt.shape[0]
    R_err = np.zeros(count)
    t_err = np.zeros(count)
    for i in range(count):
        if np.isnan(np.sum(t_pred[i])):
            continue
        #r_err = logm(np.dot(R_pred[i].reshape((1, 3)).transpose(), R_gt[i].reshape((1, 3)))) / 2
        r_err = logm(np.dot(R_pred[i].reshape((1, 3)), R_gt[i].reshape((1, 3)).transpose())) / 2
        R_err[i] = LA.norm(r_err, 'fro')
        t_err[i] = LA.norm(t_pred[i] - t_gt[i])
    return np.median(R_err) * 180 / np.pi, np.median(t_err) / diameter

def np_2_o3d(np_pc):
    # Convert numpy pc to o3d pc
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(np_pc)
    return o3d_pc

def print_Teaser_results(R_m2c, solution, t_m2c, src, N_OUTLIERS, end, start):
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

def write_results_to_txt(file_path, obj_id, ir, correspondence, add_score, add_score_thres,add_score_thres_xyz, add_s_score, add_score_icp, add_score_icp_thres, add_score_icp_thres_xyz, add_s_score_icp,
                          T_gt, T_pred, T_pred_icp, error_m, error_rad
                         #inlier_ratio, 
                         #accuracy, 
                         #recall
                         ):
    with open(file_path, 'w') as txt_file:
        txt_file.write("Object ID: {}\n".format(obj_id))
        txt_file.write("Inlier ration of P_pred: {}\n".format(ir))
        txt_file.write("Num. of correspondences: {}\n".format(correspondences))
        txt_file.write("Avg. Euclidean Distance (ADD) [cm]: {}\n".format(add_score))
        txt_file.write("Add Score thres: {}\n".format(add_score_thres))
        txt_file.write("Add Score thres (xyz direction): {}\n".format(add_score_thres_xyz))
        txt_file.write("Add-S Score: {}\n".format(add_s_score))
        txt_file.write("Avg. Euclidean Distance (ADD) ICP: {}\n".format(add_score_icp))
        txt_file.write("Add Score ICP thres: {}\n".format(add_score_icp_thres))
        txt_file.write("Add Score ICP thres (xyz direction): {}\n".format(add_score_icp_thres_xyz))
        txt_file.write("Add-S Score ICP: {}\n".format(add_s_score_icp))
        txt_file.write("Error [cm]: {}\n".format(error_m))
        txt_file.write("Error [deg]: {}\n".format(error_rad))
        txt_file.write("T_gt (Ground Truth Transformation):\n")
        txt_file.write("{}\n".format(T_gt))
        txt_file.write("T_pred (Predicted Transformation):\n")
        txt_file.write("{}\n".format(T_pred))
        txt_file.write("T_pred_ICP (Predicted Transformation from ICP):\n")
        txt_file.write("{}\n".format(T_pred_icp))
        #txt_file.write("Inlier Ratio: {}\n".format(inlier_ratio))
        #txt_file.write("Accuracy: {}\n".format(accuracy))
        #txt_file.write("Recall: {}\n".format(recall))
        
def calculate_and_write_averages_to_txt(result_lists_add, output_file):
    # Calculate averages for each list in the dictionary
    averages = {}
    for key, values in result_lists_add.items():
        if len(values) > 0:
            averages[key] = sum(values) / len(values)
        else:
            averages[key] = 0  # or you can set a default value here

    # Write averages to the output text file
    with open(output_file, 'w') as txt_file:
        for key, avg in averages.items():
            txt_file.write(f"{key}: {avg}\n")

# Teaser Parameters
NOISE_BOUND = 0.05
N_OUTLIERS = 1700
OUTLIER_TRANSLATION_LB = 5
OUTLIER_TRANSLATION_UB = 10

if __name__ == "__main__":
    print("==================================================")
    print("              TEASER++ registration               ")
    print("==================================================")
    
    # load data
    # set directory path of .pt files for evaluation 
    # --> download result data (.pt files) from server using:
    # TERMINAL: scp -r unseen_object@131.159.10.67:/data/unseen_object_data/tmp/results/currrent_best_real_new/ /local/directory/
    directory_path = 'currrent_best_real_new'
    loaded_models = [] # 16 items

    # Sort loaded data
    path = sorted(os.listdir(directory_path))
    #path.sort()
    for filename in path:
        if filename.endswith(".pt"):
            file_path = os.path.join(directory_path, filename)
            loaded_model = torch.load(file_path)
            loaded_models.append(loaded_model)

    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    
    #check if results_poses_TEASER directory exists
    isExist = os.path.exists("results_poses_TEASER")
    if not isExist:
        os.makedirs("results_poses_TEASER")
    
    #check if results_poses_TEASER/results directory exists
    isExist = os.path.exists("results_poses_TEASER/results")
    if not isExist:
        os.makedirs("results_poses_TEASER/results")

    #check if results_poses_TEASER/ply directory exists
    isExist = os.path.exists("results_poses_TEASER/ply")
    if not isExist:
        os.makedirs("results_poses_TEASER/ply")

    num_of_obj = 15
    # Create n different named lists
    result_lists_add_score = create_named_lists(num_of_obj, 'add_score')
    result_lists_add = create_named_lists(num_of_obj, 'add')
    result_lists_add_score_xyz = create_named_lists(num_of_obj, 'add_score_xyz')
    result_lists_adds_score = create_named_lists(num_of_obj, 'adds_score')
    
    # Access each loaded object in list
    iterator = range(len(loaded_models))

    for i in tqdm(iterator):
        result = loaded_models[i]

        # load P_gt, P_pred, CAD_ver, PC_ver
        P_gt = result[-1]['P'] # (5176, 2) (cad_indices, pc_indices)
        P_pred = result[-1]['p_pred'].T.cpu().detach().numpy() # (500, 2) (cad_indices, pc_indices)
        CAD_ver = result[0]['xyz'] # (5002, 3)
        PC_ver = result[-1]['pcd_depth'] # (1369, 3)
        R_m2c = result[-1]['R_m2c']
        t_m2c = result[-1]['t_m2c']
        CAD_diam = result[-1]['diam_cad'] # units: cm
        ir = result[-1]['ir']
        obj_id = result[-1]['obj_id']
        #print(obj_id)
        correspondences = P_pred.shape[0]

        # split P matrix into CAD and PC indices
        cad_idx = P_pred[:,0] # CAD indices [N,] (500,)
        pc_idx = P_pred[:,1] # pc indices [N,] (500,)
    
        # replace indices by coordinates for Teaser++
        
        src_T = replace_idx_by_coo(cad_idx, CAD_ver)
        dst_T = replace_idx_by_coo(pc_idx, PC_ver)

        # Check if src_T is = 0 (empty)
        if isinstance(src_T, int):
            continue

        src = np.transpose(src_T)
        dst = np.transpose(dst_T)

        # TEASER++ pose estimation
        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        start = time.time()
        solver.solve(src, dst)
        end = time.time()

        solution = solver.getSolution()

        print_Teaser_results(R_m2c, solution, t_m2c, src, N_OUTLIERS, end, start)
    
        # Compute pred and gt pose (4x4) from R and t
        T_est = R_t_2_pose(solution.rotation, solution.translation)
        T_gt = R_t_2_pose(R_m2c, t_m2c)
  
        # Compute add score
        add_score, score = add(T_est, T_gt, CAD_ver, CAD_diam)
        print("add score (before ICP):", add_score)

        # Compute add score (HybridPose)
        HybridPose_add_score = compute_add_score(CAD_ver, CAD_diam, T_gt, T_est, percentage=0.1)
        print("HybridPose add score (before ICP):", HybridPose_add_score)

        # Compute add-s score (HybridPose)
        adds_score = compute_adds_score(CAD_ver, CAD_diam, T_gt, T_est, percentage=0.1)
        print("add-s score (before ICP):", adds_score)

        # Compute Pose error:
        pose_error = compute_pose_error(CAD_diam, T_gt, T_est)
        print("pose error (before ICP):", pose_error)
     
        # --------------------------- ICP ---------------------------

        # Convert CAD_ver to o3d point cloud:
        cad_03d = np_2_o3d(CAD_ver)
        
        # transform CAD_ver by TEASER ESTIMATED pose --> convert to o3d point cloud:
        cad_RANSAC_transformed_est = transform(CAD_ver, T_est)
        cad_o3d_RANSAC_transformed_est = np_2_o3d(cad_RANSAC_transformed_est)

        # transform CAD_ver by GROUND TRUTH pose --> convert to o3d point cloud:
        cad_transformed_gt = transform(CAD_ver, T_gt)
        cad_o3d_transformed_gt = np_2_o3d(cad_transformed_gt)
        
        # Inputs for ICP:
        source = cad_03d # CAD 
        target = cad_o3d_transformed_gt # gt transformed gt
        threshold = 0.2 # Maximum correspondence points-pair distance
        trans_init = T_est # initial trasnformation (coming from RANSAC, Kabsch-RANSAC, TEASER etc.)

        print("Apply point-to-point ICP")
    
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        #draw_registration_result(source, target, reg_p2p.transformation)

        # Compute add score (After ICP)
        #add_score = add(T_est.transformation, T_gt, cad_coo)
        add_score_ICP, add_score_ICP_score = add(reg_p2p.transformation, T_gt, CAD_ver, CAD_diam)
        print("add score (after ICP):", add_score_ICP)
        #obj_id.append(add_score_ICP_score)
        # Append elements to a specific list (e.g., "obj_3")
        
        result_lists_add_score["obj_{}_add_score".format(obj_id)].append(add_score_ICP_score)
        result_lists_add["obj_{}_add".format(obj_id)].append(add_score_ICP)
        
        # Compute add score (HybridPose) (After ICP)
        HybridPose_add_score_ICP = compute_add_score(CAD_ver, CAD_diam, T_gt, reg_p2p.transformation, percentage=0.1)
        print("HybridPose add score (after ICP):", HybridPose_add_score_ICP)
        result_lists_add_score_xyz["obj_{}_add_score_xyz".format(obj_id)].append(HybridPose_add_score_ICP)
        
        # Compute add-s score (HybridPose) (After ICP)
        adds_score_ICP = compute_adds_score(CAD_ver, CAD_diam, T_gt, reg_p2p.transformation, percentage=0.1)
        print("add-s score (after ICP):", adds_score_ICP)
        result_lists_adds_score["obj_{}_adds_score".format(obj_id)].append(adds_score_ICP)

        # Compute Pose error: (After ICP)
        pose_error_ICP = compute_pose_error(CAD_diam, T_gt, reg_p2p.transformation)
        print("pose error (after ICP):", pose_error_ICP)

        R_tmp, t_tmp = pose_2_R_t(reg_p2p.transformation)
        error_cm = np.linalg.norm(t_m2c - t_tmp)
        error_rad = get_angular_error(R_m2c, R_tmp)
        error_deg = rad_2_deg(error_rad)

        write_results_to_txt("results_poses_TEASER/results/obj_{}_result_{}.txt".format(obj_id, i), obj_id,  ir, correspondences, add_score, score, HybridPose_add_score, adds_score, add_score_ICP, add_score_ICP, HybridPose_add_score_ICP, adds_score_ICP,
                      T_gt, T_est, reg_p2p.transformation, error_cm, error_deg)
        
        # Write ply files

        #check if results_poses_TEASER/ply/i directory exists
        isExist = os.path.exists("results_poses_TEASER/ply/obj_{}_result_{}".format(obj_id, i))
        if not isExist:
            os.makedirs("results_poses_TEASER/ply/obj_{}_result_{}".format(obj_id, i))

        # CAD coordinates
        cad_03d = np_2_o3d(CAD_ver)
        o3d.io.write_point_cloud("results_poses_TEASER/ply/obj_{}_result_{}/cad_{}.ply".format(obj_id, i, i), cad_03d)
    
        # CAD coordinates transformed (ESTIMATED)
        cad_transformed_est = transform(CAD_ver, reg_p2p.transformation)
        cad_o3d_transformed_est = np_2_o3d(cad_transformed_est)
        o3d.io.write_point_cloud("results_poses_TEASER/ply/obj_{}_result_{}/cad_{}_pose_est.ply".format(obj_id, i, i), cad_o3d_transformed_est)

        # CAD coordinates transformed (GROUND TRUTH)
        cad_transformed_gt = transform(CAD_ver, T_gt)
        cad_o3d_transformed_gt = np_2_o3d(cad_transformed_gt)
        o3d.io.write_point_cloud("results_poses_TEASER/ply/obj_{}_result_{}/cad_{}_pose_gt.ply".format(obj_id, i, i), cad_o3d_transformed_gt)

        # PC coordinates
        pc_03d = np_2_o3d(PC_ver)
        o3d.io.write_point_cloud("results_poses_TEASER/ply/obj_{}_result_{}/pc_{}.ply".format(obj_id, i, i), pc_03d)

# Append results to metric specific lists and write average results to txt file
with open('results_poses_TEASER/avg_results.txt', 'w') as txt_file:

        # txt_file.write("Object ID: {}\n".format(obj_id))

        for list_name, lst in result_lists_add_score.items():
            avg = calculate_average(lst)
            print(f"Average for {list_name}: {lst}")
            txt_file.write(f"Average for {list_name}: {avg}\n")
    
        for list_name, lst in result_lists_add.items():
            avg = calculate_average(lst)
            print(f"Average for {list_name}: {lst}")
            txt_file.write(f"Average for {list_name}: {avg}\n")
        
        for list_name, lst in result_lists_add_score_xyz.items():
            avg = calculate_average(lst)
            print(f"Average for {list_name}: {lst}")
            txt_file.write(f"Average for {list_name}: {avg}\n")
        
        for list_name, lst in result_lists_adds_score.items():
            avg = calculate_average(lst)
            print(f"Average for {list_name}: {lst}")
            txt_file.write(f"Average for {list_name}: {avg}\n")
