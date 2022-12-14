
import utils
import torch
import numpy as np
from collections import OrderedDict

class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=64)
        self.layer_2 = torch.nn.Linear(in_features=64, out_features=64)
        self.output_layer = torch.nn.Linear(in_features=64, out_features=output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_activation(self.output_layer(layer_2_output))
        return output

class SACAgent:

    ALPHA_INITIAL = 1.
    DISCOUNT_RATE = 0.99
    LEARNING_RATE = 10 ** -4
    SOFT_UPDATE_INTERPOLATION_FACTOR = 0.05

    def __init__(self, parent, params):
        
        self.agent_params = params['agent']

        self.state_dim = self.agent_params['num_bands']
        self.action_dim = self.agent_params['num_bands']
        self.actor_params = params['actor']
        self.epsilon = self.actor_params['epsilon']
        self.epsilon_decay = self.actor_params['epsilon_decay']

        self.num_critic_updates = self.agent_params['num_critic_updates']

        self.critic_local = Network(input_dimension=self.state_dim,
                                    output_dimension=self.action_dim)
        self.critic_local2 = Network(input_dimension=self.state_dim,
                                     output_dimension=self.action_dim)

        self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.LEARNING_RATE)
        self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.LEARNING_RATE)

        self.critic_target = Network(input_dimension=self.state_dim,
                                     output_dimension=self.action_dim)
        self.critic_target2 = Network(input_dimension=self.state_dim,
                                      output_dimension=self.action_dim)

        self.soft_update_target_networks(tau=1.)

        self.actor_local = Network(
            input_dimension=self.state_dim,
            output_dimension=self.action_dim,
            output_activation=torch.nn.Softmax(dim=1)
        )
        self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.LEARNING_RATE)

        # self.replay_buffer = ReplayBuffer(self.environment)
        self.replay_buffer = parent.replay_buffer

        self.target_entropy = 0.98 * -np.log(1 / self.action_dim)
        # self.log_alpha = torch.tensor(np.log(self.ALPHA_INITIAL), requires_grad=True)
        self.log_alpha = torch.tensor(np.log(self.ALPHA_INITIAL), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.LEARNING_RATE)

    def get_next_action(self, state):
        rand = np.random.rand()
        number_of_non_zeros = np.count_nonzero(state)

        selected_bands = np.argwhere(state != 0).squeeze(-1).tolist()

        # if rand < self.epsilon or number_of_non_zeros <= 1:
        # for on-policy SAC
        if False:
            # discrete_action = self.get_action_deterministically(state)
            # print(discrete_action)
            unselected_bands = np.squeeze(np.argwhere(state == 0))
            discrete_action = np.random.choice(unselected_bands)
            action_type = "Random"
        else:
            discrete_action = self.get_action_nondeterministically(state)
            # discrete_action = self.get_action_nondeterministically(state)
            unselected_bands = np.squeeze(np.argwhere(state == 0))

            # count = 0
            while discrete_action in selected_bands:
                discrete_action = self.get_action_nondeterministically(state)
                # print("here")
                # if count > 10:
                #     discrete_action = np.random.choice(unselected_bands)
                # count += 1
                # discrete_action = self.get_action_nondeterministically(state)

            # print("Discrete Action", discrete_action)
            action_type = "SAC Action"
        
        self.decay_epsilon()
        return discrete_action, action_type

    def get_action_nondeterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = np.random.choice(range(self.action_dim), p=action_probabilities)
        return discrete_action

    def get_action_deterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = np.argmax(action_probabilities)
        return discrete_action

    def train(self):

        # self.critic_optimiser.zero_grad()
        # self.critic_optimiser2.zero_grad()
        # self.actor_optimiser.zero_grad()
        # self.alpha_optimiser.zero_grad()

        loss = OrderedDict()
        for _ in range(self.num_critic_updates):

            self.critic_optimiser.zero_grad()
            self.critic_optimiser2.zero_grad()
            self.actor_optimiser.zero_grad()
            self.alpha_optimiser.zero_grad()

            sampled_paths = self.replay_buffer.sample_buffer_random(self.agent_params['batch_size'])

            flat_sampled_path = [path for trajectory in sampled_paths for path in trajectory]
            obs = np.array([path['ob'] for path in flat_sampled_path])
            acs = np.array([path['ac'] for path in flat_sampled_path])
            obs_next = np.array([path['ob_next'] for path in flat_sampled_path])
            res = np.array([path['re'] for path in flat_sampled_path])
            terminals = np.array([path['terminal'] for path in flat_sampled_path])

            obs, acs, obs_next, res, terminals = utils.from_numpy(obs), utils.from_numpy(acs), utils.from_numpy(obs_next), utils.from_numpy(res), utils.from_numpy(terminals)

            critic_loss, critic2_loss = self.critic_loss(obs, acs, res, obs_next, terminals)
            
            critic_loss.backward()
            critic2_loss.backward()
            self.critic_optimiser.step()
            self.critic_optimiser2.step()

            actor_loss, log_action_probabilities = self.actor_loss(obs)

            actor_loss.backward()
            self.actor_optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probabilities)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.soft_update_target_networks()

            loss['Actor_Loss'] = actor_loss.item()
            loss['Alpha_Loss'] = alpha_loss.item()
            loss['Critic_Loss1'] = critic_loss.item()
            loss['Critic_Loss2'] = critic2_loss.item()

        return loss

    def train_on_transition(self, state, discrete_action, next_state, reward, done):
        transition = (state, discrete_action, reward, next_state, done)
        self.train_networks(transition)

    def train_networks(self, transition):
        # Set all the gradients stored in the optimisers to zero.
        self.critic_optimiser.zero_grad()
        self.critic_optimiser2.zero_grad()
        self.actor_optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()
        # Calculate the loss for this transition.
        self.replay_buffer.add_transition(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        if self.replay_buffer.get_size() >= self.REPLAY_BUFFER_BATCH_SIZE:
            # get minibatch of 100 transitions from replay buffer
            minibatch = self.replay_buffer.sample_minibatch(self.REPLAY_BUFFER_BATCH_SIZE)
            minibatch_separated = list(map(list, zip(*minibatch)))

            # unravel transitions to get states, actions, rewards and next states
            states_tensor = torch.tensor(np.array(minibatch_separated[0]))
            actions_tensor = torch.tensor(np.array(minibatch_separated[1]))
            rewards_tensor = torch.tensor(np.array(minibatch_separated[2])).float()
            next_states_tensor = torch.tensor(np.array(minibatch_separated[3]))
            done_tensor = torch.tensor(np.array(minibatch_separated[4]))

            critic_loss, critic2_loss = \
                self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor)

            critic_loss.backward()
            critic2_loss.backward()
            self.critic_optimiser.step()
            self.critic_optimiser2.step()

            actor_loss, log_action_probabilities = self.actor_loss(states_tensor)

            actor_loss.backward()
            self.actor_optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probabilities)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.soft_update_target_networks()

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor):
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
            next_q_values_target = self.critic_target.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)
            soft_state_values = (action_probabilities * (
                    torch.min(next_q_values_target, next_q_values_target2) - self.alpha * log_action_probabilities
            )).sum(dim=1)

            # print("shapes critics",soft_state_values.shape, done_tensor.shape, (done_tensor*soft_state_values).shape)

            next_q_values = rewards_tensor + done_tensor * self.DISCOUNT_RATE*soft_state_values

        soft_q_values = self.critic_local(states_tensor).gather(1, actions_tensor.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        soft_q_values2 = self.critic_local2(states_tensor).gather(1, actions_tensor.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        
        # critic_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values)
        # critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)
        
        critic_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)
        # weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic_square_error, critic2_square_error)]
        # self.replay_buffer.update_weights(weight_update)
        # TODO : fix this later

        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic_loss, critic2_loss

    def actor_loss(self, states_tensor,):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.critic_local(states_tensor)
        q_values_local2 = self.critic_local2(states_tensor)
        inside_term = self.alpha.detach() * log_action_probabilities - torch.min(q_values_local, q_values_local2)

        # print(action_probabilities.shape, inside_term.shape,(action_probabilities * inside_term).shape)
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        # print( (log_action_probabilities + self.target_entropy).shape)
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss

    def get_action_info(self, states_tensor):
        action_probabilities = self.actor_local.forward(states_tensor)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)

        return action_probabilities, log_action_probabilities

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probabilities = self.actor_local.forward(state_tensor)

        return action_probabilities.squeeze(0).detach().numpy()

    def soft_update_target_networks(self, tau=SOFT_UPDATE_INTERPOLATION_FACTOR):
        self.soft_update(self.critic_target, self.critic_local, tau)
        self.soft_update(self.critic_target2, self.critic_local2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def predict_q_values(self, state):

        if isinstance(state, np.ndarray):
            state = utils.from_numpy(state)

        q_values = self.critic_local(state)
        q_values2 = self.critic_local2(state)
        return torch.min(q_values, q_values2)

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay 