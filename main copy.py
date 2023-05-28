from dataloader import ModelNet40
import numpy as np 
from pointnet import PointNet
import torch
import torch.nn.functional as F
from resnet import *
def voxelize_point_cloud(points, voxel_size=0.005):
    if points.shape[0] == 3 :
        points = points.T
    # Determine voxel grid size based on the desired resolution
    min_bound = np.min(points, axis=0)#-16*voxel_size
    max_bound = np.max(points, axis=0)#+16*voxel_size
    voxel_grid_size = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

    # Calculate voxel indices for all points
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)

    # Create an empty voxel grid
    voxel_grid = np.zeros(voxel_grid_size, dtype=int)

    # Count the number of points in each voxel
    voxel_grid[tuple(voxel_indices.T)] += 1

    return voxel_grid, voxel_indices  

def get_grid_around_voxels(voxel_indices, N, voxel_grid):
    # Calculate the size of the voxel grid
    voxel_grid_size = np.max(voxel_indices, axis=0) + 1

    # Pad the grid to handle boundaries
    padded_grid = np.pad(voxel_grid, N // 2, mode='constant')

    # Generate an array of sliding window indices
    window_indices = np.indices((N,N,N )).reshape(3,-1)#- N // 2 #.reshape(3, -1).T - N // 2

    # Add voxel indices to window indices and reshape for broadcasting
    sliding_indices = voxel_indices[..., np.newaxis] + window_indices
    N_grids = voxel_indices.shape[0]
    sliding_indices = sliding_indices.transpose(1,0,2).reshape(3,-1)

    # Extract the grid values around each voxel using advanced indexing
    grid_around_voxels = padded_grid[sliding_indices[0], sliding_indices[1], sliding_indices[2]]
    grid_around_voxels = grid_around_voxels.reshape(N_grids, N,N,N)
    return grid_around_voxels

def find_points_within_radius(points, r, num_of_points):
    points = points.T
    # Calculate pairwise Euclidean distances between all points
    distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

    # Create a mask of points within the specified radius
    mask = distances <= r

    # Generate random indices for each group of points within the radius
    random_indices = np.random.rand(*mask.shape) 
    random_indices *= mask  # Mask out indices outside the radius
    
    top_indices = np.argsort(random_indices, axis=1)[:, ::-1]  # Sort indices in descending order

    # Replace top indices associated with False in the mask with the corresponding row index
    row_indices = np.arange(mask.shape[0])[:,np.newaxis]
    row_indices = np.repeat(row_indices, mask.shape[1], 1)
    new_mask = np.take(mask, top_indices)
    top_indices = np.where(new_mask, top_indices, row_indices)

    # Select up to N points for each group
    sampled_indices = top_indices[:, :num_of_points]
    sampled_points = np.take(points, sampled_indices, axis=0)

    return sampled_points

import open3d as o3d
def find_positives (pc1, pc2, r=0.02, num_of_points=3, V = 1000000):
    pc1, pc2 = pc1.T, pc2.T
    # Compute pairwise Euclidean distances between all pairs of points
    distances = np.linalg.norm(pc1[:, np.newaxis] - pc2, axis=2)

    # Create a mask of points within the specified radius
    mask = distances <= r

    # Find the indices of points within the radius for each point in pc1
    indices = np.asarray(np.where(mask)).T

    # Shuffle the indices along the first axis
    np.random.shuffle(indices)

    # Select up to num_points_within_radius positive points for each point in pc1
    selected_indices = np.full((pc1.shape[0], num_of_points), V)

    for row in indices:
        empty_spots = np.where(selected_indices[row[0]] == V)[0]
        num_empty_spots = len(empty_spots)
        if num_empty_spots > 0:
            selected_indices[row[0], empty_spots[0]] = row[1]
    return indices
        

def find_negatives(pc1, pc2, r=0.05, num_of_points=64):
    pc1, pc2 = pc1.T, pc2.T
    # Compute pairwise Euclidean distances between all pairs of points
    distances = np.linalg.norm(pc1[:, np.newaxis] - pc2, axis=2)

    # Create a mask of points within the specified radius
    mask = distances >= r
    # Generate random indices for each group of points within the radius
    random_indices = np.random.rand(*mask.shape) 
    random_indices *= mask  # Mask out indices outside the radius
    
    top_indices = np.argsort(random_indices, axis=1)[:, ::-1]  # Sort indices in descending order
    sampled_indices = top_indices[:, :num_of_points]

    return sampled_indices


def visualize_pcd(pcd_list):
    import open3d as o3d 
    new_list = []
    for i,pcd in enumerate(pcd_list):
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(pcd.T)
        colors = [[100,0,0], [0,100,0]]
        pcd_temp.paint_uniform_color(colors[i])
        new_list.append(pcd_temp)
    o3d.visualization.draw_geometries(new_list)


class OnlineTripletLoss(torch.nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, pp_1, pp_2, np_):


        ap_distances = (pp_1.unsqueeze(-1)- pp_2).pow(2).sum((1,2))  # .pow(.5)
        an_distances = (pp_1.unsqueeze(-1) - np_).pow(2).sum((1,2))  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

modelnet = ModelNet40(768,num_subsampled_points=768, gaussian_noise=True, unseen=True, rot_factor = 1)
PointNet_model = PointNet(512).cuda()
#resnet_model = generate_model(10, n_input_channels=1, n_classes=400)
resnet = ResNet(BasicBlock, [1, 1, 1, 1], [16, 32, 64, 64],  n_input_channels=1, n_classes=512).cuda()

opt = torch.optim.Adam(PointNet_model.parameters())
loss_function = OnlineTripletLoss(300)
for epoch in range(5000):
    for (pc1,pc2, Rab,  Tab, Rba, Tba, _, _) in modelnet:

        voxel_1, voxel_idx = voxelize_point_cloud(pc1)

        pacthes_voxel = get_grid_around_voxels(voxel_grid=voxel_1 , N = 32,  voxel_indices=voxel_idx)
        patches_pc1 = find_points_within_radius(pc1,0.16, 64)
        patches_pc2 = find_points_within_radius(pc2,0.16, 64)
        #visualize_pcd([pc1, Rba@pc2 + Tba[:,np.newaxis]])
        positive = find_positives(pc1, Rba@pc2 + Tba[:,np.newaxis])
        negatives = find_negatives(pc1, Rba@pc2 + Tba[:,np.newaxis])

        positive_1 = find_positives(Rba@pc2 + Tba[:,np.newaxis], pc1)
        negatives_1= find_negatives(Rba@pc2 + Tba[:,np.newaxis], pc1)

        max_size = 500
        num_splits = int(np.ceil(patches_pc1.shape[0] / max_size))

        #patches_pc1_splits = torch.split(torch.Tensor(patches_pc1).permute(0,2,1), max_size, dim=0)
        #patches_pc2_splits = torch.split(torch.Tensor(patches_pc2).permute(0,2,1), max_size, dim=0)
        patches_pc2 = torch.Tensor(patches_pc2).permute(0,2,1)
        #patches_pc1 = torch.Tensor(patches_pc1).permute(0,2,1)
        positive_splits = torch.split(torch.Tensor(positive).type(torch.int), max_size, dim=0)
        v_sp = torch.Tensor(pacthes_voxel[:,np.newaxis])
        for positive in zip( positive_splits):
            positive = positive[0]
            pp_1 = resnet(v_sp[positive[:,0]].cuda())
            pp_2 = PointNet_model(patches_pc2[positive[:,1]].cuda()) 


            np_1 = negatives[positive[:,1],0]
            np_1 = PointNet_model(patches_pc2[np_1].cuda())  

            loss = loss_function(pp_1, pp_2, np_1)
            print(loss, pp_1.shape[0])  
            opt.zero_grad()
            loss.backward()
            opt.step()

        #voxel1, idx1 = voxelize_point_cloud(pc1)

        #voxel2, idx2 = voxelize_point_cloud(pc2)  




