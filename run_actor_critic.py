import pickle
from rl_trainer_ac import RL_Trainer
from agents.ac_agent import ACAgent


params = {'agent':{
            'agent_class': ACAgent,
            'n_iter':10000,
            'trajectory_sample_size': 10,
            'batch_size':10,
            'num_critic_updates':10,
            'num_bands':200,
            'reward_type':'correlation',
            'num_critic_updates_per_agent_update': 1,
            'num_actor_updates_per_agent_update' : 1
            },
          'actor':{
            'num_bands':200,
            'learning_rate': 0.001
          },
          'critic':{
            'num_grad_steps_per_target_update' : 1,
            'num_target_updates' : 1,
            'num_bands':200,
            'gamma':0.99,
            'learning_rate': 0.001
            },
          'policy':{
            'epsilon':0.99,
            'epsilon_decay':0.9999
            },
            'data':{
            'band_selection_num':30,
            'dataset_type':'IndianPines',
            'data_file_path':r'/Users/pratikaher/FALL22/HyperSpectralRL/ForPratik/data_indian_pines_drl.mat',
            'sample_ratio':0.1
            },
         }


if __name__ == "__main__":

    with open('data/data_cache.pickle', 'rb') as handle:
        data_cache_loaded = pickle.load(handle)
    
    rl_trainer = RL_Trainer(params, data_cache_loaded)

    rl_trainer.run_training_loop()

    print(rl_trainer.logging_df.head())


# import pickle

# data_cache = agent.cache

# with open('data_cache.pickle', 'wb') as handle:
#     pickle.dump(data_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data_cache.pickle', 'rb') as handle:
#     data_cache_loaded = pickle.load(handle)