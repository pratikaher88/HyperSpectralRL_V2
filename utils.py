import numpy as np
import torch
import scipy

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
        hyper_path = '/Users/pratikaher/FALL22/HyperSpectralRL/ForPratik/data_indian_pines_drl.mat'
        hyper = scipy.io.loadmat(hyper_path)['x'][:, :self.num_bands]
        #hyper = np.load(hyper_path)
        # randomly sample for x% of the pixels
        indices = np.random.randint(0, hyper.shape[0], int(hyper.shape[0]*self.sample_ratio))
        self.rl_data = hyper[indices, :]
        print(self.rl_data.shape)
        
    def load_salient_objects_data(self):
        hyper_path = '../data/salient_objects/hyperspectral_imagery/0001.npy'
        hyper = np.load(hyper_path)
        print(hyper.shape)
        # randomly sample for x% of the pixels
        indices = np.random.randint(0, hyper.shape[0], int(hyper.shape[0]*self.sample_ratio))
        self.rl_data = hyper[indices, :]
        print(self.rl_data.shape)
        
    def load_botswana_data(self):
        self.rl_data = scipy.io.loadmat(self.data_file_path)
    #def load_salient_objects(self)


device = 'cpu'

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()