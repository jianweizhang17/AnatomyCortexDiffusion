

import argparse
from omegaconf import OmegaConf
from torch import autograd
autograd.set_detect_anomaly(True)

from model import GaussianDiffusion
from dataset import BaseDataset

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import nibabel as nb

import numpy as np

import os
from tqdm import tqdm

from loguru import logger

import random

def _adapt_state_dict_keys(state_dict, want_module_prefix: bool):
    """
    Adapt checkpoint keys between DataParallel ('module.') and non-DataParallel formats.
    """
    if not state_dict:
        return state_dict
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    if want_module_prefix and not has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    if (not want_module_prefix) and has_module:
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(checkpoint_file: str, model, optimizer=None, scheduler=None, device: str | torch.device = "cpu"):
    """
    Load model (and optionally optimizer/scheduler) from a checkpoint.

    Returns:
        start_epoch (int): next epoch index to run.
    """
    if checkpoint_file is None:
        return 0
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    ckpt = torch.load(checkpoint_file, map_location=device)
    state = ckpt.get("model", ckpt)

    # Load into model, handling DataParallel prefix differences.
    want_module_prefix = any(n.startswith("module.") for n, _ in model.named_parameters())
    state = _adapt_state_dict_keys(state, want_module_prefix=want_module_prefix)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.warning(f"Checkpoint loaded with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    # ckpt['epoch'] is the last finished epoch index in this script; resume at next epoch.
    last_epoch = int(ckpt.get("epoch", -1))
    return max(0, last_epoch + 1)



def main():
    parser = argparse.ArgumentParser()
    
    # experiment parameters
    parser.add_argument('--devices',type=int,nargs='+',help='gpu ids or cpu')
    parser.add_argument("--log_dir",type=str,help="dir where log and models are saved")
    parser.add_argument("--exp_name",type=str,help="experiment name")
    parser.add_argument("--checkpoint_file",type=str,default=None,help="checkpoint file for resuming training")
    
    # model parameters
    parser.add_argument('--timesteps',type=int,default=1000,help='Input channel to the backbone')
    parser.add_argument('--sampling_timesteps',type=int,default=None,help='Input channel to the backbone')
    parser.add_argument('--objective',type=str,default='pred_v',help='Input channel to the backbone')
    parser.add_argument('--sec_objective',type=str,default='None',help='Input channel to the backbone')
    parser.add_argument('--beta_schedule',type=str,default='cosine',help='Input channel to the backbone')
    parser.add_argument('--schedule_fn_kwargs',type=int,default=dict(),help='Input channel to the backbone')
    parser.add_argument('--ddim_sampling_eta',type=float,default=0.0,help='Input channel to the backbone')
    parser.add_argument('--auto_normalize',type=bool,default=True,help='Input channel to the backbone')
    parser.add_argument('--offset_noise_strength',type=float,default=0.0,help='Input channel to the backbone')
    parser.add_argument('--min_snr_loss_weight',type=bool,default=False,help='Input channel to the backbone')
    parser.add_argument('--min_snr_gamma',type=int,default=5,help='Input channel to the backbone')
    
    parser.add_argument('--input_channel',type=int,help='Input channel to the backbone')
    parser.add_argument('--x_channel',type=int,default=0,help='Input data channel to the backbone, input_channel-x_channel is the condition channel')
    parser.add_argument('--output_channel',type=int,help='output channel to the backbone')
    parser.add_argument('--chs',type=int,nargs='+',default=[32,64,128],help='output channel to the backbone')
    parser.add_argument('--dropout_rate',type=int,default=0,help='dropout rate in backbone')
    parser.add_argument('--start_level',type=int,default=6,help='the starting ico level of input data')
    parser.add_argument('--norm_type',type=str,default='batch',help='the normaliztion layer used in backbone')
    parser.add_argument('--use_att',type=bool,nargs='+',default=[False,True,True],help='indicator if use self attention in a certain level')
    parser.add_argument('--attr_ch',type=int,default=256,help='the hidden dimension of attr condition, e.g. time,age,gender,etc.')
    parser.add_argument('--attr_condition_names',type=str,nargs='+',default=[],help='labels for attribute condtions') 
    parser.add_argument('--attr_condition_types',type=str,nargs='+',default=[],help='type for attribute condtions,cat_n for categorical data with n classes,num for numerical data')
    
    # dataset parameters
    parser.add_argument('--data_info_csv',type=str,help='data_info_csv')
    parser.add_argument('--use_balanced_dataset',type=int,default=0,help='balance dataset groups')
    parser.add_argument('--data_file_headers',type=str,nargs='+',default=[],help='data file headers used for loading data')
    parser.add_argument('--norm_mode',type=str,default='none',help='data normalization mode, none for raw data,standard for range [-1,1]')
    
    # train parameters
    parser.add_argument('--epoch',type=int,default=1000,help='Total number of epoch')
    parser.add_argument('--batch_size',type=int,default=8,help='batch size, should be divisible by number of devices')
    parser.add_argument('--sample_frq',type=int,default=1,help='Frequency in epoch to sample and save a new sample')
    parser.add_argument('--iter_loss_show_frq',type=int,default=100,help='Frequency in iteration to show iteration loss')
    parser.add_argument('--save_frq',type=int,default=1,help='Frequency in epoch for saving model to checkpoints')
    parser.add_argument('--cfg_rate',type=float,default=0.0,help='classifier free guidance rate in training, cfg_rate percentage of the time, the training will run wo condition')
    parser.add_argument('--cfg_w',type=float,default=0.0,help='weight for classifier free guidance')
    
    opt = parser.parse_args()
    
    os.makedirs(os.path.join(opt.log_dir,opt.exp_name),exist_ok=True)
    logger.add(os.path.join(opt.log_dir,opt.exp_name,'log.txt'),encoding='utf-8',level="DEBUG")
    
    
    dataset = BaseDataset(opt)
    model = GaussianDiffusion(opt)
    print (f'Number of data: {len(dataset)}')
    train_loader = DataLoader(dataset,batch_size=opt.batch_size,shuffle = True)
    
   
    optimizer = Adam(model.parameters(),lr=1e-5,weight_decay=0.001)
    scheduler = CosineAnnealingLR(optimizer,T_max = 10)
    optimizer.zero_grad()
    
    device = torch.device(f'cuda:{opt.devices[0]}')
    model = nn.DataParallel(model,device_ids=[torch.device(f'cuda:{i}') for i in opt.devices]).to(torch.device(f'cuda:{opt.devices[0]}'))
    print (f'\nNumber of parameters in backbone: {sum(p.numel() for p in model.parameters())}')

    # Resume from checkpoint if provided
    start_epoch = 0
    if opt.checkpoint_file is not None:
        start_epoch = load_checkpoint(opt.checkpoint_file, model, optimizer=optimizer, scheduler=scheduler, device=device)
        logger.info(f"Resumed from checkpoint {opt.checkpoint_file}, start_epoch={start_epoch}")
    
    # One sanity sample + initial checkpoint only when starting fresh
    if start_epoch == 0:
        for data in train_loader:
            x = data['x'].to(device).float()
            break
        noise = torch.randn_like(x).to(device) # pyright: ignore[reportPossiblyUnboundVariable]
        pred_x = model.module.ddim_sample(noise,{})
        nb.freesurfer.io.write_morph_data(os.path.join(opt.log_dir,opt.exp_name,f'sample_epoch_{-1}'),pred_x[-1].detach().cpu().numpy().reshape((-1,)))

        torch.save(
            {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': -1,
            },
            os.path.join(opt.log_dir,opt.exp_name,f'checkpoint_{0}.pt')
        )
    
    for epoch in range(start_epoch, opt.epoch):
        epoch_loss = 0.0 
        epoch_count = 0
        for data in train_loader:
            x = data['x'].to(device).float()
            # config attr conditions into a dict
            attr_conditions = {}
            for attr_condition_name,attr_condition_type in zip(opt.attr_condition_names,opt.attr_condition_types):
                if 'cat' in attr_condition_type: 
                    attr_conditions[attr_condition_name] = data[attr_condition_name].to(device).long()
                elif 'num' in attr_condition_type:
                    attr_conditions[attr_condition_name] = data[attr_condition_name].to(device).float()
                else:
                    raise ValueError(f'Unknown attr_condition_type {attr_condition_type}')
            # random pick to use cfg or not
            if random.random() < opt.cfg_rate:
                attr_conditions = {}
            pred_target,target = model(x,attr_conditions)
            loss = nn.functional.l1_loss(pred_target,target)
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_count += 1.0

            if epoch_count % opt.iter_loss_show_frq == 0:
                logger.info(f'epoch {epoch},iteration {epoch_count},iteration loss: {loss.item()}')
            
        logger.info(f'epoch {epoch}, epoch loss: {epoch_loss/epoch_count}')
        if (epoch+1) % opt.sample_frq == 0:
            noise = torch.randn_like(x).to(device)[0,:,:].unsqueeze(0) # pyright: ignore[reportPossiblyUnboundVariable]
            pred_x = model.module.ddim_sample(noise,{})[-1]
            if opt.norm_mode == 'standard':
                pred_x = (pred_x+1.0)/2.0*5.0
            
            pred_x = pred_x.detach().cpu().numpy().astype(np.float32)
            for i,header in enumerate(opt.data_file_headers):
                nb.freesurfer.io.write_morph_data(os.path.join(opt.log_dir,opt.exp_name,f'sample_{header}_epoch_{epoch}'),pred_x[0,i,:].reshape((-1,)))
                
                
            # fig = surf_tensor_2_im(pred_x,clim=[0,3])
            # fig.write_html(os.path.join(opt.log_dir,opt.exp_name,f'sample_epoch_{epoch}.html'))
        if (epoch+1) % opt.save_frq == 0:
            torch.save({'model':model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch':epoch},
                       os.path.join(opt.log_dir,opt.exp_name,f'checkpoint_{epoch+1}.pt'))
        
        
        
    
            
            

    
    
if __name__ == '__main__':
    main()
    


