'''
Class definitions for the ddpg agent, noise model, and replay buffer.

Some methods taken from DDPG file in Udacity Deep Learning course
Coded by: Jeremy Vila
11/13/18
'''

import numpy as np
import random
import copy
from collections import namedtuple, deque
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim

params = {
'buffer_size': int(1e6),  # replay buffer size
'batch_size': 256,        # minibatch size. Increased batch size for efficieny
'gamma': 0.95,            # discount factor. Reduced gamma for more discout
'tau': 1e-3,              # for soft update of target parameters
'actor_lr': 1e-3,         # learning rate of the actor. Set larger to learn fast
'critic_lr': 1e-3}        # learning rate for the critic.  Set larger to learn fast

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Class definition for the agent
# Taken from Udacity deep learning nanodegree ddpg function
class Agent():
    
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Networks; one is a target
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=params['actor_lr'])

        # Critic Networks; one is a target
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=params['critic_lr'])

        # Noise process
        # Taken from Udacity deep learning nanodegree ddpg function
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        # Taken from Udacity deep learning nanodegree ddpg function
        self.memory = ReplayBuffer(action_size, params['buffer_size'], params['batch_size'], random_seed)

    # Method to load the weights
    def load_weights(self, critic_path='cc_critic.pth', actor_path='cc_actor.pth', load_opt=True):
        actor = torch.load(actor_path)
        critic = torch.load(critic_path)

        self.actor_local.load_state_dict(actor['local'])
        self.actor_target.load_state_dict(actor['target'])

        self.critic_local.load_state_dict(critic['local'])
        self.critic_target.load_state_dict(critic['target'])

        if load_opt:
            self.actor_optimizer.load_state_dict(actor['opt'])
            self.critic_optimizer.load_state_dict(critic['opt'])
    
    def step(self, state, action, reward, next_state, done):
        # Save experience
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > params['batch_size']:
            experiences = self.memory.sample()
            self.learn(experiences, params['gamma'])

    # Get actions according to policy
    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        # Add noise for randomness
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    # update the agent from a batch
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Update critic
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Update based on the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Update based on the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target network (Seems to work every iteration)
        self.soft_update(self.critic_local, self.critic_target, params['tau'])
        self.soft_update(self.actor_local, self.actor_target, params['tau'])                     

    # Update the target network in a soft manner
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# Taken from Udacity deep learning nanodegree ddpg function
# Ornstein-Uhlenbeck process
class OUNoise:

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()  # Reset the noise

    def reset(self):
        self.state = copy.copy(self.mu)

    # Generate random sample of the noise
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

# Taken from Udacity deep learning nanodegree ddpg function
class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        # method to add a tuple to the buffer
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # Set the random seed
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        # Add dictionary element to the buffer
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        # Random dample of experiences
        experiences = random.sample(self.memory, k=self.batch_size)

        # Unwrap states, actions, rewards, next states, and dones
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)