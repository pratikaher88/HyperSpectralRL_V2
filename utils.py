from re import L
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
        self.sample_ratio = params['sample_ratio']
        
        self.num_bands = num_bands
        #load the data
        assert self.dataset_type in ('IndianPines', 'Botswana', 'SalientObjects', 'PlasticFlakes', 'SoilMoisture'), f'{self.dataset_type} is not valid'
        #separating out in case any of the data requires unique pre-processig
        if self.dataset_type == 'IndianPines':
            self.load_indian_pine_data()
        elif self.dataset_type == 'Botswana':
            self.load_botswana_data()
        elif self.dataset_type == 'SalientObjects':
            self.load_salient_objects_data()
        elif self.dataset_type == 'PlasticFlakes':
            self.load_plastic_flakes_data()
        elif self.dataset_type == 'SoilMoisture':
            self.load_soil_moisture_data()
        #self.x_train = None
        #self.y_train = None
        #self.x_test = None
        #self.y_test = None
        
    def load_indian_pine_data(self):
        #hyper_path = self.data_file_path
        #hyper = scipy.io.loadmat(hyper_path)['x'][:, :self.num_bands]
        #hyper = np.load(hyper_path)
        # randomly sample for x% of the pixels
        #indices = np.random.randint(0, hyper.shape[0], int(hyper.shape[0]*self.sample_ratio))
        #self.rl_data = hyper[indices, :]
        #print(self.rl_data.shape)

        self.rl_data = self._stack('data/indian_pines/hyperspectral_imagery')
        self._sample()
        
        
    def load_salient_objects_data(self):
        
        self.rl_data = self._stack('data/salient_objects/hyperspectral_imagery')
        self._sample()

        #self.rl_data = np.load('')
        # randomly sample for x% of the pixels
        #indices = np.random.randint(0, hyper.shape[0], int(hyper.shape[0]*self.sample_ratio))
        #self.rl_data = hyper[indices, :]
        #print(self.rl_data.shape)

    def load_plastic_flakes_data(self):
        self.rl_data = self._stack('data/plastic_flakes/hyperspectral_imagery')
        self._sample()
        
    def load_botswana_data(self):
        self.rl_data = scipy.io.loadmat(self.data_file_path)
    #def load_salient_objects(self)

    def load_soil_moisture_data(self):
        self.rl_data = self._stack('data/soil_moisture/hyperspectral_imagery')
        self._sample()

    def _sample(self):
        indices = np.random.randint(0, self.rl_data.shape[0], int(self.rl_data.shape[0]*self.sample_ratio))
        self.rl_data[indices, :]

    def _stack(self, data_folder):
        data = None

        for _, _, files in os.walk(data_folder):
            for idx, file in enumerate(files):
                print(f'\rLoading {idx} out of {len(files)}', end='')
                file_data = np.load(os.path.join(data_folder, file))

                if isinstance(data, type(None)):
                    data = file_data
                else:
                    data = np.vstack((data, file_data))

        return data


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