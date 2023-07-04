from DPFM.dpfm.diffusion_net import geometry
import open3d as o3d
import torch
from pathlib import Path
from torch.utils.data import Dataset
from .scene import base_scene_dataset
import numpy as np
import os
import gin

@gin.configurable()
class base_object_dataset(Dataset):
    def __init__(self, min_vis=0.25, cache_dir = '/home/morashed/repo', LBO_pc = True, **kwargs):
        self.scenes = base_scene_dataset(**kwargs)
        self.min_vis = min_vis
        self.collect_obj_data()
        self.cache_dir = cache_dir
        self.LBO_pc = LBO_pc
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.cache_dir = self.cache_dir / self.scenes.render_data_name
            self.cache_dir.mkdir(exist_ok=True)
            self.cache_dir = self.cache_dir / self.scenes.mode
            self.cache_dir.mkdir(exist_ok=True)
        
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
        CAD_mesh = None
        self.mapping_list = []
        for i, scene in enumerate(self.scenes):

            for j, obj in enumerate(scene["scene_info"]):

                if obj['visib_fract'] < self.min_vis:
                    continue
                else:
                    self.mapping_list.append((i,j))
    
    def __getitem__(self, index):
        i,j = self.mapping_list[index]
        CAD_mesh = None
        if self.cache_dir is not None:
            cache = True
            base_name = f'min_vis_{self.min_vis}_{i}_{j}_'
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

            gt_info = self.scenes[i]['scene_gt'][j]

            cad_path = Path(os.environ["data_root"]) / self.scenes.render_data_name/"models" /  f"obj_{gt_info['obj_id']:06d}.ply"

            #load cad here or specifiy transform o3d.io.read_triangle_mesh
            obj_dict = {
                'visib_fract':self.scenes[i]['scene_info'][j]['visib_fract'],
                'R_m2c': np.array(gt_info['cam_R_m2c']).reshape(3,3),     #list to np
                't_m2c': np.asarray(gt_info['cam_t_m2c'])*0.1, #list to np
                'obj_id': gt_info['obj_id'],
                'pcd_depth': pcd,
                'cad_path': cad_path, 
                'scale_cad': 0.1
            } 
            CAD_mesh = o3d.io.read_triangle_mesh(str(cad_path))
            CAD_ver =   np.asarray(CAD_mesh.vertices)*obj_dict['scale_cad'] #0.1
            align_pc = self.transform(pcd, obj_dict['R_m2c'], obj_dict['t_m2c'], inv=True) 
            P = self.find_positives(CAD_ver, align_pc, r = 0.2)
            p_new = np.argwhere(P)
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
            
            base_name = f'min_vis_{self.min_vis}_'
            CAD_filename = self.cache_dir / (base_name+ f'CAD_LBO_{obj_dict["obj_id"]}.npz')

        if CAD_filename.exists() and cache: 
            CAD_LBO_dict = dict(np.load(CAD_filename, allow_pickle=True))
        else:
            
            CAD_mesh = o3d.io.read_triangle_mesh(str(cad_path)) if CAD_mesh is None else CAD_mesh
            CAD_ver =   torch.Tensor(np.asarray(CAD_mesh.vertices)*obj_dict['scale_cad'])
            CAD_faces = torch.Tensor(np.asarray(CAD_mesh.triangles)).to(int)
            CAD_norm =  torch.Tensor(np.asarray(CAD_mesh.vertex_normals)*(1/obj_dict['scale_cad']))

            


            CAD_frames, CAD_mass, CAD_L, CAD_evals, CAD_evecs, CAD_gradX, CAD_gradY = geometry.get_operators(verts=CAD_ver, faces=CAD_faces, normals=CAD_norm) #for future utilize cahcing add caching to reading of gt json 

            CAD_LBO_dict = {
                "frames" :  CAD_frames ,
                "mass" :    CAD_mass , 
                #"L" :       CAD_L , 
                "evals" :   CAD_evals , 
                "evecs" :   CAD_evecs , 
                "CAD_ver": CAD_ver,
                "CAD_faces": CAD_faces,
                "CAD_norm": CAD_norm,
                #"gradX" :   CAD_gradX , 
                #"gradY" :   CAD_gradY 
            }
            if cache: np.savez(CAD_filename, **CAD_LBO_dict)
        if self.LBO_pc:
            if pc_filename.exists() and cache: 
                pcd_LBO_dict = dict(np.load(pc_filename, allow_pickle=True))
            else:
                pcd_frames, pcd_mass, pcd_L, pcd_evals, pcd_evecs, pcd_gradX, pcd_gradY = geometry.get_operators(verts=torch.Tensor(pcd), faces=torch.Tensor([])) #for future utilize cahcing add caching to reading of gt json 

                pcd_LBO_dict = {
                    "frames" :  pcd_frames ,
                    "mass" :    pcd_mass , 
                    #"L" :       pcd_L, 
                    "evals" :   pcd_evals , 
                    "evecs" :   pcd_evecs , 
                    #"gradX" :   pcd_gradX , 
                    #"gradY" :   pcd_gradY 
                }
                if cache: np.savez(pc_filename, **pcd_LBO_dict)
        else:
            pcd_LBO_dict = None


        return CAD_LBO_dict, pcd_LBO_dict, obj_dict
    def __len__(self):

        return len(self.mapping_list)

    def find_positives(self, pc1, pc2, r=0.2):
        # Compute pairwise Euclidean distances between all pairs of points
        distances = np.linalg.norm(pc1[:, np.newaxis] - pc2, axis=2)

        # Create a mask of points within the specified radius
        mask = distances <= r

        return mask.astype(bool)
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