from DPFM.dpfm.diffusion_net import geometry
import open3d as o3d
import torch
from pathlib import Path
from torch.utils.data import Dataset
from .scene import base_scene_dataset
import numpy as np
import os
import gin
import copy 
from sklearn.neighbors import BallTree
from tqdm import tqdm # TIM STROHMEYER
from DPFM.dpfm.utils import farthest_point_sample, square_distance
import json
@gin.configurable()
class base_object_dataset(Dataset):
    def __init__(self, min_vis=0.25, cache_dir = '/home/morashed/repo', LBO_pc = True, **kwargs):
        self.scenes = base_scene_dataset(**kwargs)
        self.min_vis = min_vis
        self.cache_dir = cache_dir
        self.LBO_pc = LBO_pc
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.cache_dir = self.cache_dir / self.scenes.render_data_name
            self.cache_dir.mkdir(exist_ok=True)
            self.cache_dir = self.cache_dir / self.scenes.mode
            self.cache_dir.mkdir(exist_ok=True)
        self.collect_obj_data()

    def remove_outliers(self, pcd): # TIM STROHMEYER
        '''
        nb_neighbors:   num of neighbors to calculate the average distance for a given point.
        std_ratio:      threshold of average distances across the point cloud. 
                        lower --> stronger outlier removal
        '''
        # Pass xyz to Open3D.o3d.geometry.PointCloud
        pcd_03d = o3d.geometry.PointCloud()
        pcd_03d.points = o3d.utility.Vector3dVector(pcd)

        # Statisticial Outlier removal
        cl, ind = pcd_03d.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.2)
        inlier_cloud = pcd_03d.select_by_index(ind)

        # Convert Open3D.o3d.geometry.PointCloud to numpy array
        pcd_clean = np.asarray(inlier_cloud.points)

        return pcd_clean

    def dpt_2_pcld(self, dpt, cam_scale, K, mask):
        idx = np.indices(dpt.shape[:2])
        xmap = idx[0]
        ymap = idx[1]
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        dpt = dpt[mask]
        #msk = (dpt > 1e-8).astype(np.float32)
        row = (ymap[mask] - K[0,2]) * dpt / K[0,0]
        col = (xmap[mask]- K[1,2]) * dpt / K[1,1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=1
        )
        return dpt_3d

    def collect_obj_data(self):
        #mapping list cache
        if self.cache_dir is not None:
            cache = True
            mapping_list_filename = self.cache_dir / 'mapping_list.npz'
        if mapping_list_filename.exists() and cache: 
            self.mapping_list = np.load(mapping_list_filename, allow_pickle=True)['mapping_list']
        else:
            #normal collecting
            CAD_mesh = None
            self.mapping_list = []
            print(f"collecting obj data")
            for i, scene in tqdm(enumerate(self.scenes)):

                for j, obj in enumerate(scene["scene_info"]):

                    if obj['visib_fract'] < self.min_vis:
                        continue
                    else:
                        self.mapping_list.append((i,j))
            #saving mapping list cache
            if cache: np.savez(mapping_list_filename, mapping_list=self.mapping_list)
    
    def __getitem__(self, index):
        i,j = self.mapping_list[index]
        CAD_mesh = None
        sparse_keys = ["L", "gradX", "gradY"]
        if self.cache_dir is not None:
            cache = True
            base_name = f'{i}_{j}_'
            CAD_filename = self.cache_dir / (base_name+'CAD_LBO.npz')
            pc_filename = self.cache_dir / (base_name+'pc_LBO.npz')
            obj_filename = self.cache_dir / (base_name+'obj.npz')

        if obj_filename.exists() and cache: 
            obj_dict = dict(np.load(obj_filename, allow_pickle=True))
            cad_path = obj_dict['cad_path']

        else:
            scene = self.scenes[i]
            depth = scene['depth']

            seg = scene["seg"]
            seg_mask = seg[j] == 255

            depth = scene["depth"].T
            K = np.array(scene["camera"]["cam_K"]).reshape(3,3)
            cam_scale = scene["camera"]["depth_scale"]

            pcd = self.dpt_2_pcld(depth.T, cam_scale*1000, K, seg_mask)
            pcd = self.remove_outliers(pcd) # TIM STROHMEYER
            if pcd.shape[0]>2000:
                ratio = 2000/pcd.shape[0]
                idx0 = farthest_point_sample(torch.Tensor(pcd).t(), ratio=ratio)
                pcd = pcd[idx0]
            gt_info = self.scenes[i]['scene_gt'][j]

            model_folder = "models" if self.scenes.mode.lower() != 'test' else "models_eval"
            cad_path = Path(os.environ["data_root"]) / self.scenes.render_data_name/ model_folder/  f"obj_{gt_info['obj_id']:06d}.ply"
            CAD_diam = json.load((Path(os.environ["data_root"]) / self.scenes.render_data_name/ model_folder/'models_info.json').open())[str(gt_info['obj_id'])]["diameter"]*0.1
            #load cad here or specifiy transform o3d.io.read_triangle_mesh
            obj_dict = {
                'visib_fract':self.scenes[i]['scene_info'][j]['visib_fract'],
                'R_m2c': np.array(gt_info['cam_R_m2c']).reshape(3,3),     #list to np
                't_m2c': np.asarray(gt_info['cam_t_m2c'])*0.1, #list to np
                'obj_id': gt_info['obj_id'],
                'pcd_depth': pcd,
                'cad_path': cad_path, 
                'scale_cad': 0.1,
                'diam_cad': CAD_diam
            } 
            if CAD_filename.exists() and cache: 
                CAD_LBO_dict = dict(np.load(CAD_filename, allow_pickle=True))
                #CAD_LBO_dict = self.dict_to_tensor(CAD_LBO_dict, sparse_keys)
                CAD_ver = CAD_LBO_dict['xyz']
            else:

                CAD_mesh = o3d.io.read_triangle_mesh(str(cad_path))
                CAD_mesh = CAD_mesh.simplify_quadric_decimation(10000)
                CAD_ver =   np.asarray(CAD_mesh.vertices)*obj_dict['scale_cad'] #0.1
            align_pc = self.transform(pcd, obj_dict['R_m2c'], obj_dict['t_m2c'], inv=True) 
            #P = self.find_positives(CAD_ver, align_pc, r = 0.2)
            #p_new = np.argwhere(P)
            p_new = self.find_positives(CAD_ver, align_pc, r = obj_dict['diam_cad']*0.05)
            l2 = pcd.shape[0]
            l1 = CAD_ver.shape[0]
            overlap_12, overlap_21 = self.get_overlap(l1,l2,p_new)
            obj_dict.update({
                'align_pc': align_pc,
                'P': p_new,
                'overlap_12':overlap_12,
                'overlap_21': overlap_21,
            })

            if cache: np.savez(obj_filename, **obj_dict)
        if cache  :
            
            #base_name = f'min_vis_{self.min_vis}_'
            CAD_filename = self.cache_dir / (f'CAD_LBO_{obj_dict["obj_id"]}.npz')

        if CAD_filename.exists() and cache: 
            CAD_LBO_dict = dict(np.load(CAD_filename, allow_pickle=True))
            CAD_LBO_dict = self.dict_to_tensor(CAD_LBO_dict, sparse_keys)

        else:
            
            CAD_mesh = o3d.io.read_triangle_mesh(str(cad_path)) if CAD_mesh is None else CAD_mesh
            CAD_mesh = CAD_mesh.simplify_quadric_decimation(10000)
            CAD_ver =   torch.Tensor(np.asarray(CAD_mesh.vertices)*obj_dict['scale_cad'])
            CAD_faces = torch.Tensor(np.asarray(CAD_mesh.triangles)).to(int)
            CAD_norm =  torch.Tensor(np.asarray(CAD_mesh.vertex_normals)*(1/obj_dict['scale_cad']))

            

            #idx0 = farthest_point_sample(CAD_ver.t(), ratio=0.25)
            #dists, idx1 = square_distance(CAD_ver.unsqueeze(0), CAD_ver[idx0].unsqueeze(0)).sort(dim=-1)
            #dists, idx1 = square_distance(CAD_ver, CAD_ver[idx0])#.sort(dim=-1)

            #dists, idx1 = dists[:, :, :130].clone(), idx1[:, :, :130].clone()

            CAD_frames, CAD_mass, CAD_L, CAD_evals, CAD_evecs, CAD_gradX, CAD_gradY = geometry.get_operators(verts=CAD_ver, faces=CAD_faces, normals=CAD_norm) #for future utilize cahcing add caching to reading of gt json 

            CAD_LBO_dict = {
                "frames" :  CAD_frames ,
                "mass" :    CAD_mass , 
                "evals" :   CAD_evals , 
                "evecs" :   CAD_evecs , 
                "xyz": CAD_ver,
                "faces": CAD_faces,
                "norm": CAD_norm,
                "gradX" :   CAD_gradX , 
                "gradY" :   CAD_gradY ,
                "L" : CAD_L,
                #"sample_idx": [idx0, idx1, dists]

            } 
            
            
            if cache: 
                CAD_LBO_dict_save = self.save_sparse_tensor(copy.deepcopy(CAD_LBO_dict), sparse_keys)
                np.savez(CAD_filename, **CAD_LBO_dict_save)

        if self.LBO_pc:
            if pc_filename.exists() and cache: 
                pcd_LBO_dict = dict(np.load(pc_filename, allow_pickle=True))
                pcd_LBO_dict = self.dict_to_tensor(pcd_LBO_dict, sparse_keys)
            else:
                pcd_frames, pcd_mass, pcd_L, pcd_evals, pcd_evecs, pcd_gradX, pcd_gradY = geometry.get_operators(verts=torch.Tensor(obj_dict['pcd_depth']), faces=torch.Tensor([])) #for future utilize cahcing add caching to reading of gt json 

                pcd_LBO_dict = {
                    "frames" :  pcd_frames ,
                    "mass" :    pcd_mass , 
                    "L" :       pcd_L, 
                    "evals" :   pcd_evals , 
                    "evecs" :   pcd_evecs , 
                    "gradX" :   pcd_gradX , 
                    "gradY" :   pcd_gradY,
                    "xyz" : obj_dict['pcd_depth'].astype(np.float32),

                }
                if cache: 
                    pcd_LBO_dict_save = self.save_sparse_tensor(copy.deepcopy(pcd_LBO_dict), sparse_keys)
                    np.savez(pc_filename, **pcd_LBO_dict_save)
        else:
            pcd_LBO_dict = None


        return CAD_LBO_dict, pcd_LBO_dict, obj_dict
    def __len__(self):

        return len(self.mapping_list)

    def find_positives(self, pc1, pc2, r=0.2):
        # Compute pairwise Euclidean distances between all pairs of points
        #distances = np.linalg.norm(pc1[:, np.newaxis] - pc2, axis=2)

        # Create a mask of points within the specified radius
        #mask = distances <= r

        #return mask.astype(bool)
        
        #---------memory and speed optimized-----------------
        
        # Fit a BallTree model to pc2
        tree = BallTree(pc2)

        # Find all points in pc1 that have a neighbor in pc2 within distance r
        indices = tree.query_radius(pc1, r)

        # Create a numpy array of pairs (i, j) where point i in pc1 is within distance r of point j in pc2
        pairs = np.array([(i, j) for i, js in enumerate(indices) for j in js])

        return pairs
        
    def transform(self, pc, R, t, inv = False):
        if inv:
            t = -1.0 *t.reshape(1,3) @ R
            return pc @ R + t
        else:
            return pc @ R.T + t

    def get_overlap (self,l_1, l_2, p):
        overlap_12 = np.zeros((l_1), dtype=np.byte)
        overlap_21 = np.zeros((l_2), dtype=np.byte)
        overlap_12[p[:,0]] = 1
        overlap_21[p[:,1]] = 1

        return overlap_12, overlap_21
    def save_sparse_tensor(self, dict, keys):
        for key in keys:
            tensor_ = dict.pop(key)
            dict.update({
                key+'_idx': tensor_.indices(),
                key+'_val': tensor_.values(),
            })
        return dict
    def dict_to_tensor(self, dict, keys):
        shape = dict['evecs'].shape[0]
        shape = (shape,shape)
        for key in keys:
            idx = dict.pop(key+'_idx')
            val = dict.pop(key+'_val')

            tensor_ = torch.sparse_coo_tensor(idx, val, shape)

            dict.update({
                key: tensor_
            })
        return dict
