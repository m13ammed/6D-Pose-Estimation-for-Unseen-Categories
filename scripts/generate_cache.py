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

from dataset import base_object_dataset
import utils
import gin
import argparse

def psuedo_collate(inputs):

    return 0
if __name__ == '__main__':
    #gin.external_configurable(DPFMLoss)
    gin.external_configurable(torch.optim.Adam)
    gin.parse_config_file('/home/morashed/repo/6D-Pose-Estimation-for-Unseen-Categories/config/cache_gen.gin')
    utils.set_env_variables()
    parser = argparse.ArgumentParser()
    parser.add_argument('slice_idx', type=int, help='Index for list slicing')
    
    dataset = base_object_dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers = 0, collate_fn = psuedo_collate)


    for batch in loader:
        print(batch)