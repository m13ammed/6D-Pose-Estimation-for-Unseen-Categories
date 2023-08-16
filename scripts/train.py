#import gin
import sys 
import os
import torch
from torch.utils.tensorboard import SummaryWriter
if '__file__' in vars():
    print("We are running the script non interactively")
    path = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path.append(path)    
else:
    print('We are running the script interactively')
    sys.path.append("..")


from models.dpfm import DPFMNet
from dataset import * #base_object_dataset
import utils
import gin.torch
from fmap2pointmap_solvers import *


from datetime import datetime

# Define a class for Tensorboard logging
class TensorboardLogger:
    def __init__(self, log_dir,comment):
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir = log_dir[:-1]if log_dir[-1] == '/' else log_dir

        self.log_dir = log_dir + "/" + current_time + "_"+comment
        self.writer = None
        self.global_step = 0
        self.epoch = 1
    def log_summaries(self, summaries):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)
        for tag, value in summaries.items():
            self.writer.add_scalar(tag, value, self.global_step)
            self.global_step += 1
    def log_summaries_epoch(self, summaries, epoch):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)
        
        for tag in summaries[0].keys():
            value = torch.Tensor([v[tag] for v in summaries])
            value = torch.mean(value)
            self.writer.add_scalar(tag+"_epoch", value, epoch)

@gin.configurable()
def train_net(model, criterion, optimizer, decay_iter, decay_factor, epochs, logging_dir, comment=""):
    # Check for CUDA availability and set device accordingly
    if torch.cuda.is_available() and os.environ["cuda"].lower() == 'true':
        device = torch.device(f'cuda:{os.environ["device"]}')
    else:
        device = torch.device("cpu")

    logger = TensorboardLogger(logging_dir, comment)

    # create dataset
    train_dataset = utils.prepare_train_datasets()
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(os.environ["BS"]), shuffle=True, collate_fn = collate, drop_last = True, num_workers = int(os.environ["num_workers"]))

    # define model
    if "pretrained_model" in os.environ.keys():
        if os.environ["pretrained_model"].lower() != "none":
            state_dict = torch.load(os.environ["pretrained_model"]) #"/data/unseen_object_data/tmp/logs/Jul25_02-36-42_hb_ycbv_all_100/192.pt" 
            model.load_state_dict(state_dict)
    model = model.to(device)

    optimizer = optimizer(model.parameters())
    criterion = criterion.to(device)

    fmap2pointmap_solver = choose_fmap2pointmap_solver()
    # Training loop
    print("start training")
    iterations = 0
    for epoch in range(1, epochs + 1):
        losses = []
        if epoch % decay_iter == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']*decay_factor

        model.train()
        for i, (CAD, PC, Obj) in enumerate(train_loader):
            print("start training")

            batch = {
                "shape1":CAD,
                "shape2":PC
            } 
            batch = shape_to_device(batch, device)

            # prepare iteration data
            map21 = Obj["P"]
            gt_partiality_mask12, gt_partiality_mask21 = Obj["overlap_12"].to(device), Obj["overlap_21"].to(device)
            
            # do iteration
            C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2, evecs_trans1, evecs_trans2 = model(batch)

            C_gt = torch.stack([utils.C_from_sparse_P(p.type(torch.int), cad[:,:30], pc[:,:30]) for p,cad,pc in zip(Obj["P"], CAD["evecs"], PC["evecs"])])
            loss, log_loss = criterion(C_gt, C_pred, map21, use_feat1, use_feat2,
                             overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)

            score_I = 0
            if loss > 1e5:
                print(Obj["obj_id"])
            
            for i in range(C_pred.shape[0]):
                non_zero_indices_CAD = torch.any(CAD["xyz"][i] != 0, dim=1)
                non_zero_indices_PC = torch.any(PC["xyz"][i] != 0, dim=1)


                p_pred = fmap2pointmap_solver(C_pred[i], CAD["evecs"][i][non_zero_indices_CAD,:30], PC["evecs"][i][non_zero_indices_PC,:30]) #spacial_filtering_fmap2pointmap(C_pred[i], CAD["evecs"][i][:,:30], PC["evecs"][i][:,:30]) #fmap2pointmap(C_pred[i], CAD["evecs"][i][:,:30], PC["evecs"][i][:,:30])
                score_I += utils.compute_inlier_ratio(p_pred.t(), CAD['xyz'][i], Obj["align_pc"][i].to(device), 0.1*Obj['diam_cad'][i])
            score_I = score_I/C_pred.shape[0]
            log_loss.update({
                "IR": score_I
            })

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()

            logger.log_summaries(log_loss)
            losses.append(log_loss)
            
            #   compute_inlier_ratio()
            # log
            iterations += 1
            if iterations % int(os.environ["log_interval"]) == 0:
                print(f"#epoch:{epoch}, #batch:{i + 1}, #iteration:{iterations}, loss:{loss}, IR: {score_I}")
                # save model
        if (epoch + 1) % int(os.environ["checkpoint_interval"]) == 0:
            torch.save(model.state_dict(), logger.log_dir+ f'/{epoch}.pt')
        logger.log_summaries_epoch(losses, epoch)



if __name__ == '__main__':
    #gin.external_configurable(DPFMLoss)
    gin.external_configurable(torch.optim.RMSprop) 
    gin.external_configurable(torch.optim.Adam) 


    try:
        gin.parse_config_file('./config/dpfm_orig.gin')
    except:
        gin.parse_config_file('../config/dpfm_orig.gin')
    utils.set_env_variables()
    
    train_net()

