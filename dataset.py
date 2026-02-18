import os
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nb
import numpy as np
import torch



class BaseDataset(Dataset):
    def __init__(self,opt):
        """
        """
        self.opt = opt
        
        data_info_csv = opt.data_info_csv
        attr_condition_names = opt.attr_condition_names
        data_file_headers = opt.data_file_headers
        
        self.data_info_df = pd.read_csv(data_info_csv)
        print (f'Using data info csv: {data_info_csv}')
        print (f'Using data file headers: {data_file_headers}')
        print (f'Using attribute condition names: {attr_condition_names}')
        print (f'Using use balanced dataset: {opt.use_balanced_dataset}')
        
        
        self.data_dirs = list(zip(*[self.data_info_df[i].tolist() for i in data_file_headers]))
        self.data_ids = self.data_info_df['data_id'].tolist()
        
        

        self.attr_condition_dict = {}
        for attr_condition_name in attr_condition_names:
            self.attr_condition_dict[attr_condition_name] = dict(zip(self.data_info_df['data_id'].tolist(),
                                                                self.data_info_df[attr_condition_name].tolist()))
            
            
            
            
    def __len__(self):
        return len(self.data_dirs)
              
    def __getitem__(self,idx):
        data_dirs = self.data_dirs[idx]
        # print (data_dirs)
        data = torch.cat([torch.from_numpy(nb.freesurfer.io.read_morph_data(i).reshape((1,-1)).astype(np.float32)) for i in data_dirs],dim=0)
        
        data = torch.clamp(data, min=0.0, max=5.0)
        if self.opt.norm_mode == 'standard':
            data = (data/5.0)*2.0-1.0
        
        data_dict = {}
        
        for attr_condition_name in self.attr_condition_dict:
            data_dict[attr_condition_name] = self.attr_condition_dict[attr_condition_name][self.data_ids[idx]]
        data_dict['x'] = data
        data_dict['dir'] = self.data_dirs[idx]
        data_dict['data_id'] = self.data_ids[idx]
        
        return data_dict



    