

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


def main():
    parser = argparse.ArgumentParser()
    
    # train parameters
    parser.add_argument('--devices',type=int,nargs='+',help='gpu ids or cpu')
    parser.add_argument("--checkpoint_file",type=str,default=None,help="checkpoint file for testing")
    parser.add_argument("--log_dir",type=str,help="dir where log and models are saved")
    parser.add_argument("--exp_name",type=str,help="experiment name")
    
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
    parser.add_argument('--chs',type=int,nargs='+',default=[64,128,256],help='output channel to the backbone')
    parser.add_argument('--dropout_rate',type=int,default=0,help='dropout rate in backbone')
    parser.add_argument('--start_level',type=int,default=6,help='the starting ico level of input data')
    parser.add_argument('--norm_type',type=str,default='batch',help='the normaliztion layer used in backbone')
    parser.add_argument('--use_att',type=bool,nargs='+',default=[False,True,True],help='indicator if use self attention in a certain level')
    parser.add_argument('--attr_ch',type=int,default=256,help='the hidden dimension of attr condition, e.g. time,age,gender,etc.')
    parser.add_argument('--attr_condition_names',type=str,nargs='+',default=[],help='labels for attribute condtions')
    parser.add_argument('--attr_condition_types',type=str,nargs='+',default=[],help='type for attribute condtions,cat_n for categorical data with n classes,num for numerical data')
    
    # dataset parameters
    parser.add_argument('--data_dir',type=str,help='data_dir')
    parser.add_argument('--data_info_csv',type=str,help='data_info_csv')
    parser.add_argument('--batch_size',type=int,default=8,help='data_dir')
    parser.add_argument('--use_balanced_dataset',type=int,default=0,help='balance dataset groups')
    parser.add_argument('--norm_mode',type=str,default='none',help='data normalization mode, none for raw data,standard for range [-1,1]')

    parser.add_argument('--sample_steps',type=int,default=500,help='number timesteps to apply denoising model')
    parser.add_argument('--noise_steps',type=int,default=500,help='number of steps of noise applied')
    
    opt = parser.parse_args()
    
    harmonized_data_save_dir = os.path.join(opt.log_dir,f'{opt.exp_name}_{opt.sample_steps}_{opt.noise_steps}')
    os.makedirs(harmonized_data_save_dir,exist_ok=True)
    noisy_data_save_dir = os.path.join(opt.log_dir,f'{opt.exp_name}_{opt.sample_steps}_{opt.noise_steps}_noisy')
    os.makedirs(noisy_data_save_dir,exist_ok=True)
    dataset = BaseDataset(opt)
    model = GaussianDiffusion(opt)
    train_loader = DataLoader(dataset,batch_size=opt.batch_size,shuffle = True)
    
    
    
    device = torch.device(f'cuda:{opt.devices[0]}')
    model = nn.DataParallel(model,device_ids=[torch.device(f'cuda:{i}') for i in opt.devices]).to(torch.device(f'cuda:{opt.devices[0]}'))
    print (f'\nNumber of parameters in backbone: {sum(p.numel() for p in model.parameters())}')
    
    print(opt.checkpoint_file)
    model.module.load_state_dict(torch.load(opt.checkpoint_file,map_location=device)['model'])
    model.module.eval()
    
    for data in train_loader:
        x = data['x'].to(device).float()
        data_dirs = data['dir']
        
        if (any([not os.path.exists(os.path.join(harmonized_data_save_dir,i.split('/')[-1] + '.harmonized_2_adni')) for i in data_dirs])):
            # any(['4347_20220729_I1610440' in i for i in data_dirs])):
        
            t = torch.full((x.shape[0],),opt.noise_steps,device=device).long()
            
            x_noisy = model.module.q_sample(x,t,torch.randn_like(x))
            
            pred_x_list,_ = model.module.sample(x_noisy,{},sample_steps=opt.sample_steps)
            
            for i,data_dir in enumerate(data_dirs):
                save_data_file = data_dir.split('/')[-1] + '.harmonized_2_adni'
                save_noise_data_file = data_dir.split('/')[-1] + f'.{opt.noise_steps}'
                print (f'saving to {save_noise_data_file}')
                print (f'Saving to {save_data_file}')
                pred_x = pred_x_list[-1]
                pred_x = torch.clamp(pred_x,-1,1)
                pred_x_i = pred_x[i,:,:].detach().cpu().numpy().astype(np.float32).reshape((-1,))
                x_noisy_i = x_noisy[i,:,:].detach().cpu().numpy().astype(np.float32).reshape((-1,))
                if opt.norm_mode == 'standard':
                    pred_x_i = (pred_x_i+1.0)/2.0*5.0
                nb.freesurfer.io.write_morph_data(os.path.join(harmonized_data_save_dir,save_data_file),pred_x_i)
                nb.freesurfer.io.write_morph_data(os.path.join(noisy_data_save_dir,save_noise_data_file),x_noisy_i)
        else:
            print ('skipping:',data_dirs)
        
        
        
        
    
            
            

    
    
if __name__ == '__main__':
    main()
    

