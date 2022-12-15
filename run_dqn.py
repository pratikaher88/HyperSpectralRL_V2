from sklearn.linear_model import LogisticRegression
import scipy.io
import scipy
import numpy as np
import h5py
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
from scipy.stats.stats import pearsonr
import torch
from matplotlib.pyplot import figure
import pickle
from utils import from_numpy, to_numpy, DataManager, ReplayBuffer
from rl_trainer import RL_Trainer



params = {'agent':{
        'agent_class' : 'DQN',
        'n_iter':2001,
        'trajectory_sample_size': 10,
        'batch_size':10,
        'num_critic_updates':10,
        'num_bands':200,
        'reward_type':'mutual_info', #must be correlation or mutual info
        'exp_reward':True
        },
      'data':{
        'band_selection_num':30,
        'dataset_type':'IndianPines',
        'sample_ratio':0.1
        },
      'critic':{
        'gamma':0.99,
        'double_q':False
        },
      'policy':{
        'epsilon':0.99,
        'epsilon_decay':0.9999
        }
      }

if __name__ == "__main__":

    ## removing the data cache
    #with open('data/data_cache.pickle', 'rb') as handle:
    #    data_cache_loaded = pickle.load(handle)
    
    #  agent = DQNAgent(params, data_cache_loaded)
    # agent.runAgent()

    sample_map = {'IndianPines':1, 'SalientObjects':1, 'PlasticFlakes':1, 'SoilMoisture':1, 'Foods':1}
    num_bands_map = {'IndianPines':200, 'SalientObjects':81, 'PlasticFlakes':224, 'SoilMoisture':125, 'Foods':96}

    for dataset in ['Foods']:  #'IndianPines', 'SalientObjects', 'PlasticFlakes', 'SoilMoisture', Foods
      for double_q in [False]:
        for reward_type in ['mutual_info']:
          
          params['data']['dataset_type'] = dataset
          params['critic']['double_q'] = double_q
          params['agent']['reward_type'] = reward_type
          params['data']['sample_ratio'] = sample_map[dataset]
          params['agent']['num_bands'] = num_bands_map[dataset]

          rl_trainer = RL_Trainer(params)

          rl_trainer.run_training_loop()

          print(rl_trainer.LogManager.logging_df.head())
          rl_trainer.LogManager.log_final_data()


# import pickle

# data_cache = agent.cache

# with open('data_cache.pickle', 'wb') as handle:
#     pickle.dump(data_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data_cache.pickle', 'rb') as handle:
#     data_cache_loaded = pickle.load(handle)