import numpy as np
import torch
import scipy
import os
from datetime import datetime
import pandas as pd
import json
import pickle

class ReplayBuffer():
    
    def __init__(self, size=100000):
        self.size = size
        self.paths = []
        
    def add_trajectories(self, paths):
        self.paths.extend(paths)
        self.paths = self.paths[-self.size:]
        
    def sample_buffer_random(self, num_trajectories):
        
        rand_idx = np.random.permutation(len(self.paths))[:num_trajectories]
        return [self.paths[i] for i in rand_idx]

class DataManager():
    def __init__(self, params, num_bands):
        self.rl_data = None
        self.dataset_type = params['dataset_type']
        self.data_file_path = params['data_file_path']
        self.sample_ratio = params['sample_ratio']
        
        self.num_bands = num_bands
        #load the data
        assert self.dataset_type in ('IndianPines', 'Botswana', 'SalientObjects'), f'{self.dataset_type} is not valid'
        #separating out in case any of the data requires unique pre-processig
        if self.dataset_type == 'IndianPines':
            self.load_indian_pine_data()
        elif self.dataset_type == 'Botswana':
            self.load_botswana_data()
        elif self.dataset_type == 'SalientObjects':
            self.load_salient_objects_data()
        #self.x_train = None
        #self.y_train = None
        #self.x_test = None
        #self.y_test = None
        
    def load_indian_pine_data(self):
        hyper_path = self.data_file_path
        hyper = scipy.io.loadmat(hyper_path)['x'][:, :self.num_bands]
        #hyper = np.load(hyper_path)
        # randomly sample for x% of the pixels
        indices = np.random.randint(0, hyper.shape[0], int(hyper.shape[0]*self.sample_ratio))
        self.rl_data = hyper[indices, :]
        print(self.rl_data.shape)
        
    def load_salient_objects_data(self):
        hyper_path = self.data_file_path
        hyper = np.load(hyper_path)
        print(hyper.shape)
        # randomly sample for x% of the pixels
        indices = np.random.randint(0, hyper.shape[0], int(hyper.shape[0]*self.sample_ratio))
        self.rl_data = hyper[indices, :]
        print(self.rl_data.shape)
        
    def load_botswana_data(self):
        self.rl_data = scipy.io.loadmat(self.data_file_path)
    #def load_salient_objects(self)

class LogManager():
    def __init__(self, params):
        
        self.logging_df = pd.DataFrame()
        self.dir_name = self._create_directory()
        self.log_param(params)

    def _create_directory(self):
        dir_name = f'output/Run - {datetime.now()}'
        os.mkdir(dir_name)
        return dir_name

    def log_df(self):
        self.logging_df.to_csv(f'{self.dir_name}/Results.csv')

    def log_param(self, params):
        with open (f'{self.dir_name}/config.json', 'w') as f:
            json.dump(params, f)

    def save_npy(self, file_name, np_array):
        with open(f'{self.dir_name}/{file_name}', 'wb') as f:
            np.save(f, np_array)





device = 'cpu'

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()