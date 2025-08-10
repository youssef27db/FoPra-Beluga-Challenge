import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    """!
    @brief Memory buffer for storing PPO training experiences
    
    This class manages the storage and retrieval of experiences for PPO training,
    including states, actions, probabilities, values, rewards, and done flags.
    """
    
    def __init__(self, batch_size):
        """!
        @brief Initialize the PPO memory buffer
        @param batch_size Size of batches for training
        """
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        """!
        @brief Generate randomized training batches from stored experiences
        @return Tuple containing arrays of states, actions, probabilities, values, rewards, dones, and batch indices
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions),\
            np.array(self.probs), np.array(self.values),\
            np.array(self.rewards), np.array(self.dones),\
            batches

    def store_memory(self, state, action, probs, values, reward, done):
        """!
        @brief Store a single experience in the memory buffer
        @param state Current state observation
        @param action Action taken
        @param probs Action probability from policy
        @param values State value estimate
        @param reward Received reward
        @param done Episode termination flag
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """!
        @brief Clear all stored experiences from memory
        """
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []


class ActorNetwork(nn.Module):
    """!
    @brief Actor network for PPO agent
    
    Neural network that outputs action probabilities given a state.
    Uses a categorical distribution for discrete action spaces.
    """
    
    def __init__(self, n_actions, input_dims, alpha, fc1dims=256, fc2dims=256, fc3dims=128, chkpt_dir='tmp/ppo', name = 'ppo'):
        """!
        @brief Initialize the actor network
        @param n_actions Number of possible actions
        @param input_dims Dimension of input state space
        @param alpha Learning rate for optimization
        @param fc1dims Number of neurons in first hidden layer
        @param fc2dims Number of neurons in second hidden layer  
        @param fc3dims Number of neurons in third hidden layer
        @param chkpt_dir Directory for saving checkpoints
        @param name Model name for checkpoint files
        """
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_' + name)
        self.actor = nn.Sequential(nn.Linear(input_dims, fc1dims), nn.ReLU(),
                                   nn.Linear(fc1dims, fc2dims), nn.ReLU(),
                                   nn.Linear(fc2dims, fc3dims), nn.ReLU(),
                                   nn.Linear(fc3dims, n_actions), nn.Softmax(dim=-1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """!
        @brief Forward pass through the actor network
        @param state Input state tensor
        @return Categorical distribution over actions
        """
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        """!
        @brief Save the current model state to checkpoint file
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """!
        @brief Load model state from checkpoint file
        """
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    """!
    @brief Critic network for PPO agent
    
    Neural network that estimates the value function V(s) for a given state.
    Used to compute advantages in the PPO algorithm.
    """
    
    def __init__(self, input_dims, alpha, fc1dims=256, fc2dims=256, fc3dims=128, chkpt_dir='tmp/ppo', name = 'ppo'):
        """!
        @brief Initialize the critic network
        @param input_dims Dimension of input state space
        @param alpha Learning rate for optimization
        @param fc1dims Number of neurons in first hidden layer
        @param fc2dims Number of neurons in second hidden layer
        @param fc3dims Number of neurons in third hidden layer
        @param chkpt_dir Directory for saving checkpoints
        @param name Model name for checkpoint files
        """
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_' + name)
        self.critic = nn.Sequential(nn.Linear(input_dims, fc1dims), nn.ReLU(),
                                    nn.Linear(fc1dims, fc2dims), nn.ReLU(),
                                    nn.Linear(fc2dims, fc3dims), nn.ReLU(),
                                    nn.Linear(fc3dims, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """!
        @brief Forward pass through the critic network
        @param state Input state tensor
        @return Value estimate for the given state
        """
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        """!
        @brief Save the current model state to checkpoint file
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """!
        @brief Load model state from checkpoint file
        """
        self.load_state_dict(T.load(self.checkpoint_file))


class PPOAgent:
    """!
    @brief Proximal Policy Optimization (PPO) agent implementation
    
    This class implements the PPO algorithm for reinforcement learning.
    It uses an actor-critic architecture with clipped probability ratios
    to ensure stable policy updates.
    """
    
    def __init__(self, input_dims, n_actions, gamma=0.99, alpha=0.0005, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=128, N=1024, n_epochs=5, model_name ='ppo'):
        """!
        @brief Initialize the PPO agent
        @param input_dims Dimension of the state space
        @param n_actions Number of possible actions
        @param gamma Discount factor for future rewards
        @param alpha Learning rate for both actor and critic networks
        @param gae_lambda Lambda parameter for Generalized Advantage Estimation
        @param policy_clip Clipping parameter for PPO objective
        @param batch_size Size of training batches
        @param N Number of steps to collect before learning
        @param n_epochs Number of training epochs per learning step
        @param model_name Name for saving/loading model checkpoints
        """
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha, name = model_name)
        self.critic = CriticNetwork(input_dims, alpha, name = model_name)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, values, reward, done):
        """!
        @brief Store an experience in the agent's memory
        @param state Current state observation
        @param action Action taken
        @param probs Log probability of the action
        @param values Value estimate for the state
        @param reward Reward received
        @param done Whether the episode is finished
        """
        self.memory.store_memory(state, action, probs, values, reward, done)

    def save_models(self):
        """!
        @brief Save both actor and critic model checkpoints
        """
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        """!
        @brief Load both actor and critic model checkpoints
        """
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        """!
        @brief Choose an action based on the current observation
        @param observation Current state observation
        @return Tuple of (action, log_probability, value_estimate, action_distribution)
        """
        # Convert observation to tensor
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value, dist


    def learn(self):
        """!
        @brief Perform PPO learning update using stored experiences
        
        Implements the PPO algorithm with clipped probability ratios and 
        Generalized Advantage Estimation (GAE). Updates both actor and 
        critic networks for multiple epochs.
        """
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, values_arr,\
            reward_arr, done_arr, batches = self.memory.generate_batches()

            values = values_arr
            advantages = np.zeros(len(reward_arr), dtype=np.float32)

            # Calculate advantages using GAE
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                   a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(done_arr[k])) - values[k])
                   discount *= self.gamma*self.gae_lambda
                advantages[t] = a_t

            advantage = T.tensor(advantages).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            # Train on each batch
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                # Calculate probability ratio and apply clipping
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Calculate critic loss
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                # Combined loss and backpropagation
                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

