from re import L
import numpy as np
import torch
import scipy
import os
from datetime import datetime
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from typing import Union

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
    'softmax': nn.Softmax(),
    'linear': nn.Linear(64, 64)
}

Activation = Union[str, nn.Module]

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

        #log data metadata
        self.data_metadata  = {}
        self.col_count = None
        self.full_row_count = None
        self.sample_row_count = None

        self.num_bands = num_bands
        #load the data
        assert self.dataset_type in ('IndianPines', 'Botswana', 'SalientObjects', 'PlasticFlakes', 'SoilMoisture', 'Foods'), f'{self.dataset_type} is not valid'
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
        elif self.dataset_type == 'Foods':
            self.load_foods_data()


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
        self.data_metadata['col_count'] = self.rl_data.shape[1]
        self.data_metadata['full_row_count'] = self.rl_data.shape[0]
        self._sample()
        

        
        
    def load_salient_objects_data(self):
        
        self.rl_data = self._stack('data/salient_objects/hyperspectral_imagery')
        self.data_metadata['col_count'] = self.rl_data.shape[1]
        self.data_metadata['full_row_count'] = self.rl_data.shape[0]
        self._sample()

        #self.rl_data = np.load('')
        # randomly sample for x% of the pixels
        #indices = np.random.randint(0, hyper.shape[0], int(hyper.shape[0]*self.sample_ratio))
        #self.rl_data = hyper[indices, :]
        #print(self.rl_data.shape)

    def load_plastic_flakes_data(self):
        self.rl_data = self._stack('data/plastic_flakes/hyperspectral_imagery')
        self.data_metadata['col_count'] = self.rl_data.shape[1]
        self.data_metadata['full_row_count'] = self.rl_data.shape[0]
        self._sample()
        
    def load_botswana_data(self):
        self.rl_data = scipy.io.loadmat(self.data_file_path)
    #def load_salient_objects(self)

    def load_soil_moisture_data(self):
        self.rl_data = self._stack('data/soil_moisture/hyperspectral_imagery')
        self.data_metadata['col_count'] = self.rl_data.shape[1]
        self.data_metadata['full_row_count'] = self.rl_data.shape[0]
        self._sample()

    def load_foods_data(self):
        self.rl_data = self._stack('data/foods/hyperspectral_imagery')
        self.data_metadata['col_count'] = self.rl_data.shape[1]
        self.data_metadata['full_row_count'] = self.rl_data.shape[0]
        self._sample()


    def _sample(self):
        indices = np.random.randint(0, self.rl_data.shape[0], int(self.rl_data.shape[0]*self.sample_ratio))
        self.rl_data = self.rl_data[indices, :]
        self.data_metadata['sample_row_count'] = self.rl_data.shape[0]

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
        self.log_json('config.json', params)

    def _create_directory(self):
        dir_name = f'output/Run - {datetime.now()}'
        os.mkdir(dir_name)
        return dir_name

    def log_final_data(self, band_selection_num=30):
        self.log_df()
        self.log_reward_plot(band_selection_num)


    def log_df(self):
        self.logging_df.to_csv(f'{self.dir_name}/Results.csv')

    def log_json(self, file_name, params):
        print(params)
        with open (f'{self.dir_name}/{file_name}', 'w') as f:
            json.dump(params, f)

    def save_npy(self, file_name, np_array):
        with open(f'{self.dir_name}/{file_name}', 'wb') as f:
            np.save(f, np_array)

    def log_reward_plot(self, band_selection_num):

        filter_df = self.logging_df[self.logging_df['Selected Band'] == band_selection_num-1]
        sns.lineplot(x='iter_num', y='Metric Next State', data=filter_df)
        plt.savefig(os.path.join(self.dir_name, 'reward.png'))


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)


device = 'cpu'

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()