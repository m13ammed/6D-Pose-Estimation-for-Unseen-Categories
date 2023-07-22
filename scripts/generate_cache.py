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

from tqdm import tqdm

def psuedo_collate(inputs):

    return 0
if __name__ == '__main__':
    #gin.external_configurable(DPFMLoss)
    gin.external_configurable(torch.optim.Adam)
    gin.parse_config_file('./config/cache_gen.gin')
    utils.set_env_variables()
    #parser = argparse.ArgumentParser()
    #parser.add_argument('start_idx', type=int, help='Index for list slicing')
    #parser.add_argument('end_idx', type=int, help='Index for list slicing')
    torch.multiprocessing.set_start_method('spawn')
    dataset = base_object_dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers = 4, collate_fn = psuedo_collate)


    for i, batch in tqdm(enumerate(loader)):
        print(i)
