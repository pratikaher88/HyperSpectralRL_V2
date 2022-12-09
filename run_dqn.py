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
from agents.dqn_agent import DQNAgent

params = {'agent':{
            'agent_class' : DQNAgent,
            'n_iter':10000,
            'trajectory_sample_size': 10,
            'batch_size':10,
            'num_critic_updates':10,
            'num_bands':200,
            'reward_type':'correlation'
            },
          'data':{
            'band_selection_num':30,
            'dataset_type':'IndianPines',
            'data_file_path':r'/Users/pratikaher/FALL22/HyperSpectralRL/ForPratik/data_indian_pines_drl.mat',
            'sample_ratio':0.1
            },
          'critic':{
            'gamma':0.99
            },
          'policy':{
            'epsilon':0.99,
            'epsilon_decay':0.9999
            }
         }


if __name__ == "__main__":

    with open('data/data_cache.pickle', 'rb') as handle:
        data_cache_loaded = pickle.load(handle)
    
    #  agent = DQNAgent(params, data_cache_loaded)
    # agent.runAgent()
    rl_trainer = RL_Trainer(params, data_cache_loaded)

    rl_trainer.run_training_loop()

    print(rl_trainer.logging_df.head())


# import pickle

# data_cache = agent.cache

# with open('data_cache.pickle', 'wb') as handle:
#     pickle.dump(data_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data_cache.pickle', 'rb') as handle:
#     data_cache_loaded = pickle.load(handle)