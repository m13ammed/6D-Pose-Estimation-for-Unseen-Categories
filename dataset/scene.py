from pathlib import Path
from PIL import Image
import json
import numpy as np
from torch.utils.data import Dataset
import os
from tqdm import tqdm

class base_scene_dataset(Dataset):
    def __init__(self, cache_dir = '/home/morashed/repo', mode = "train_pbr", split_txt = None, num_samples = -1, color = False):
        mode = mode.lower()
        if mode =='validation': mode = 'val'
        assert mode in ["train", "val", "test", "train_pbr"], "invalid mode, select train, val, or test"

        self.mode = mode
        self.data_root = Path(os.environ["data_root"])
        self.render_data_name = Path(os.environ["render_data_name"])
        self.render_data_path = self.data_root / self.render_data_name
        self.color = color
        #cache part from obj.py
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.cache_dir = self.cache_dir / self.render_data_name
            self.cache_dir.mkdir(exist_ok=True)
            self.cache_dir = self.cache_dir / self.mode
            self.cache_dir.mkdir(exist_ok=True)
        #
        if split_txt is None:
            self.collect_scenes()
        else:
            NotImplementedError

    def replace_directory(self, path_name ,replace, drop_file=False, drop_suffix = False):
        if drop_file:
            path_name = path_name.parents[1] / replace
        elif drop_suffix:
            path_name = path_name.parents[1] / replace / path_name.parts[-1]
            path_name = path_name.with_suffix('')
        else:
            path_name = path_name.parents[1] / replace / path_name.parts[-1]
        return path_name
    
    def check_exists (self, paths_list):
        exists = True
        for path in paths_list:
            try:
                if not path.exists():
                    exists = False
                    print(f"Warning {path} does not exist and the scene will be dropped")
                    break
            except:
                if not self.check_exists(path):
                    exists = False
                    print(f"Warning {path} does not exist and the scene will be dropped")
                    break

        return exists

    def collect_scenes(self):
        #scene list cache
        if self.cache_dir is not None:
            cache = True
            scene_list_filename = self.cache_dir / 'scene_list.npz'
        if scene_list_filename.exists() and cache: 
            print('loading scene cache')
            scene_list = np.load(scene_list_filename, allow_pickle=True)
            self.depth_path, self.camera_path, self.scene_info_path, self.seg_path, self.scene_gt_path = scene_list['depth_path'], scene_list['camera_path'], scene_list['scene_info_path'], scene_list['seg_path'], scene_list['scene_gt_path']
            if self.color:
                    self.color_path = scene_list['color_path']
        else:
            #normal collecting
            depth_path_gen = (self.render_data_path/self.mode).rglob('*/depth/*.png')
            self.depth_path, self.camera_path, self.scene_info_path, self.seg_path, self.scene_gt_path = [], [], [], [], []
            if self.color:
                self.color_path = []
            print('collecting scenes')
            for depth_path in tqdm(depth_path_gen):
                camera_path = self.replace_directory(depth_path, 'scene_camera.json', drop_file= True)
                scene_info_path = self.replace_directory(depth_path, 'scene_gt_info.json', drop_file= True)
                scene_gt_path = self.replace_directory(depth_path, 'scene_gt.json', drop_file= True)
                seg_path = self.replace_directory(depth_path, 'mask_visib', drop_suffix = True)
                seg_path = sorted(list(seg_path.parent.glob(seg_path.parts[-1]+'_*.png')))
                paths = [depth_path, camera_path, scene_info_path, scene_gt_path, seg_path]

                if self.color:
                    color_path = self.replace_directory(depth_path, 'rgb').with_suffix('.jpg')
                    paths.append(color_path)
                
                paths_exist = self.check_exists(paths)

                if paths_exist:
                    self.depth_path.append(depth_path)
                    self.camera_path.append(camera_path)
                    self.scene_info_path.append(scene_info_path)
                    self.scene_gt_path.append(scene_gt_path)
                    self.seg_path.append(seg_path)
                if self.color:
                    self.color_path.append(color_path)
            #saving scene list cache
            if cache: 
                print('saving scene cache')
                if self.color: 
                    np.savez(scene_list_filename, depth_path=self.depth_path, camera_path=self.camera_path, scene_info_path=self.scene_info_path, seg_path=self.seg_path, scene_gt_path=self.scene_gt_path, color_path=self.color_path)
                else:
                    np.savez(scene_list_filename, depth_path=self.depth_path, camera_path=self.camera_path, scene_info_path=self.scene_info_path, seg_path=self.seg_path, scene_gt_path=self.scene_gt_path)
                    
    def __getitem__(self, idx):

        depth_path = self.depth_path[idx]
        subscene_nr = str(int(depth_path.with_suffix('').parts[-1]))

        cam_file = self.camera_path[idx].open()
        cam_file = json.load(cam_file)[subscene_nr]

        scene_gt_file = self.scene_gt_path[idx].open()
        scene_gt_file = json.load(scene_gt_file)[subscene_nr]

        scene_info_file = self.scene_info_path[idx].open()
        scene_info_file = json.load(scene_info_file)[subscene_nr]

        seg_images = [np.asarray(Image.open(seg)) for seg in self.seg_path[idx]]
        depth_image = np.asarray(Image.open(depth_path))

        
        return_dict = {
        "depth" : depth_image,
        "camera" : cam_file ,
        "scene_gt" : scene_gt_file,
        "scene_info" : scene_info_file,
        "seg" : seg_images,
        "scene_nr":None,
        "subscene_nr": None
        }

        if self.color:
            return_dict.update({
                "color" : np.asarray(Image.open(self.color_path[idx])) 
            })
            
        return return_dict
    
    def __len__ (self):
        return len(self.depth_path)
