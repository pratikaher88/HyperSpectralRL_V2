from critics.qcritic import QCritic
from critics.cql_critic import CQLCritic
from policies.rnd_policy import RNDPolicy
from policies.argmax_policy import ArgMaxPolicy
from utils import *

class ExplorationOrExploitationAgent():
    def __init__(self, parent, params):
        #super(ExplorationOrExploitationAgent, self).__init__(env, agent_params)

        self.agent_params = params['agent']
        self.num_bands = self.agent_params['num_bands']
        
        self.replay_buffer = parent.replay_buffer

        self.num_exploration_steps = self.agent_params['num_exploration_steps']
        self.offline_exploitation = False #params['offline_exploitation']

        self.exploitation_params = self.agent_params['exploitation']
        self.exploration_params = self.agent_params['exploration']
        self.exploitation_critic = CQLCritic(self.agent_params)
        self.exploration_critic = QCritic(self.agent_params, self.agent_params['num_bands'])
        
        self.exploration_model = RNDPolicy(self.num_bands)
        #self.explore_weight_schedule = self.agent_params['explore_weight_schedule']
        #self.exploit_weight_schedule = self.agent_params['exploit_weight_schedule']
        
        self.policy_params = params['policy']
        self.policy = ArgMaxPolicy(self.policy_params, self.exploration_critic)
        self.eval_policy = ArgMaxPolicy(self.policy_params, self.exploitation_critic)
        self.exploit_rew_shift = 0 #self.agent_params['exploit_rew_shift']
        self.exploit_rew_scale = 1 #self.agent_params['exploit_rew_scale']
        self.eps = 0.2 #self.agent_params['eps']

        self.running_rnd_rew_std = 1
        self.normalize_rnd = False #self.agent_params['normalize_rnd']
        self.rnd_gamma = params['critic']['gamma'] #self.agent_params['rnd_gamma']

        self.learning_starts = self.agent_params['learning_starts']
        self.learning_freq = 1
        self.target_update_freq = self.agent_params['target_update_freq']
        self.t = 0
        self.num_param_updates = 0

    def train(self):

        sampled_paths = self.replay_buffer.sample_buffer_random(self.agent_params['batch_size'])

        flat_sampled_path = [path for trajectory in sampled_paths for path in trajectory]
        obs = np.array([path['ob'] for path in flat_sampled_path])
        acs = np.array([path['ac'] for path in flat_sampled_path])
        obs_next = np.array([path['ob_next'] for path in flat_sampled_path])
        res = np.array([path['re'] for path in flat_sampled_path])
        terminals = np.array([path['terminal'] for path in flat_sampled_path])

        log = {}

        if self.t > self.num_exploration_steps:
            # TODO: After exploration is over, set the actor to optimize the extrinsic critic
            #HINT: Look at method ArgMaxPolicy.set_critic

            ####COME BACK TO THIS AND CHECK LATER
            self.policy.set_critic(self.exploitation_critic)
            #self.actor.set_critic(self.exploration_critic)

        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                #and self.replay_buffer.can_sample(self.batch_size)
        ):    
            
            explore_weight = 1
            exploit_weight = 0
            #####################################################


            #Part 3
            #explore_weight = self.explore_weight_schedule.value(self.t)
            #exploit_weight = self.exploit_weight_schedule.value(self.t)

            # Run Exploration Model #

            obs = check_tensor(obs)

            #exploration_pred = self.exploration_model(next_ob_no)
            print('Obs Shape: ', obs.shape)
            exploration_pred = self.exploration_model.forward(obs)
            exploration_pred_mean = exploration_pred.mean()
            exploration_pred_std = exploration_pred.std()
            ####THIS MAY BE WRONG - SHOULD WE USE THE MEAN TO NORMALIZE###########
            self.running_rnd_rew_std = self.rnd_gamma*self.running_rnd_rew_std + (1 - self.rnd_gamma)*exploration_pred_std
            exploration_bonus = normalize(exploration_pred, exploration_pred_mean, self.running_rnd_rew_std)
            

            # Reward Calculations #
            # TODO: Calculate mixed rewards, which will be passed into the exploration critic
            # HINT: See doc for definition of mixed_reward
            
            #print('In Explore/Exploit')
            #print('re_n type: ', type(re_n))
            #print('exploration_bonus type: ', type(exploration_bonus))
            #print('--------------------------------------')

            
            res = check_tensor(res)

            mixed_reward = exploit_weight*res + explore_weight*exploration_bonus

            # TODO: Calculate the environment reward
            # HINT: For part 1, env_reward is just 're_n'
            #       After this, env_reward is 're_n' shifted by self.exploit_rew_shift,
            #       and scaled by self.exploit_rew_scale
            ############# for part #1. ############
            #env_reward = re_n

            ######## for part 2 ###########
            env_reward = (res+self.exploit_rew_shift)*self.exploit_rew_scale


            # Update Critics And Exploration Model #

            # TODO 1): Update the exploration model (based off s')
            # TODO 2): Update the exploration critic (based off mixed_reward)
            # TODO 3): Update the exploitation critic (based off env_reward)
            expl_model_loss = self.exploration_model.update(obs_next)
            exploration_critic_loss = self.exploration_critic.update(obs, acs, obs_next, mixed_reward, terminals)
            exploitation_critic_loss = self.exploitation_critic.update(obs, acs, obs_next, env_reward, terminals)

            # Target Networks #
            if self.num_param_updates % self.target_update_freq == 0:
                # TODO: Update the exploitation and exploration target networks
                self.exploration_critic.update_target_network()
                self.exploitation_critic.update_target_network()


            # Logging #
            #log['Exploration Critic Loss'] = exploration_critic_loss['Training Loss']
            #log['Exploitation Critic Loss'] = exploitation_critic_loss['Training Loss']
            #log['Exploration Model Loss'] = expl_model_loss

            # TODO: Uncomment these lines after completing cql_critic.py
            #if self.exploitation_critic.alpha >= 0:
            #     log['Exploitation Data q-values'] = exploitation_critic_loss['Data q-values']
            #     log['Exploitation OOD q-values'] = exploitation_critic_loss['OOD q-values']
            #     log['Exploitation CQL Loss'] = exploitation_critic_loss['CQL Loss']

            self.num_param_updates += 1

        self.t += 1
        return log


    """
    def step_env(self):
        
            #Step the env and store the transition
            #At the end of this block of code, the simulator should have been
            #advanced one step, and the replay buffer should contain one more transition.
            #Note that self.last_obs must always point to the new latest observation.
       
        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        perform_random_action = np.random.random() < self.eps or self.t < self.learning_starts

        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            processed = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(processed)

        next_obs, reward, done, info = self.env.step(action)
        self.last_obs = next_obs.copy()

        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        if done:
            self.last_obs = self.env.reset()
    """