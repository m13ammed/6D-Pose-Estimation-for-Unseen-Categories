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

    try:
        gin.parse_config_file('./config/cache_gen.gin')
    except:
        gin.parse_config_file('../config/cache_gen.gin')

    utils.set_env_variables()

    torch.multiprocessing.set_start_method('spawn')
    dataset = base_object_dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(os.environ["BS"]), shuffle=False, num_workers = num_workers = int(os.environ["num_workers"]), collate_fn = psuedo_collate)


    for i, batch in tqdm(enumerate(loader)):
        print(i)
