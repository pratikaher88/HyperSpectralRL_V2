import pickle
from rl_trainer import RL_Trainer
from agents.ac_agent import ACAgent


params = {'agent':{
            'agent_class': 'AC',
            'n_iter':2001,
            'trajectory_sample_size': 10,
            'batch_size':10,
            'num_critic_updates':10,
            'num_bands':200,
            'reward_type':'correlation',
            'num_critic_updates_per_agent_update': 1,
            'num_actor_updates_per_agent_update' : 1,
            'exp_reward': True
            },
          'actor':{
            'num_bands':200,
            'band_selection_num': 30,
            'learning_rate': 0.001,
            'epsilon': 1,
            'epsilon_decay':0.99999
          },
          'critic':{
            'num_grad_steps_per_target_update' : 1,
            'num_target_updates' : 1,
            'num_bands':200,
            'gamma':0.99,
            'learning_rate': 0.001,
            'double_q':False
            },
          'policy':{
            'epsilon':0.99,
            'epsilon_decay':0.9999
            },
            'data':{
            'band_selection_num':30,
            'dataset_type':'IndianPines',
            'sample_ratio':0.1
            },
         }


if __name__ == "__main__":

    #with open('data/data_cache.pickle', 'rb') as handle:
    #    data_cache_loaded = pickle.load(handle)

    sample_map = {'IndianPines':1, 'SalientObjects':1, 'PlasticFlakes':1, 'SoilMoisture':1, 'Foods':1}
    num_bands_map = {'IndianPines':200, 'SalientObjects':81, 'PlasticFlakes':224, 'SoilMoisture':125, 'Foods':96}

    for dataset in ['Foods']: #'IndianPines', SalientObjects, 'PlasticFlakes',  'SoilMoisture', 'Foods'
      for double_q in [True]:
        for reward_type in ['mutual_info']:
          
          params['data']['dataset_type'] = dataset
          params['critic']['double_q'] = double_q
          params['agent']['reward_type'] = reward_type
          params['data']['sample_ratio'] = sample_map[dataset]
          params['agent']['num_bands'] = num_bands_map[dataset]
          params['actor']['num_bands'] = num_bands_map[dataset]
          params['critic']['num_bands'] = num_bands_map[dataset]
    
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