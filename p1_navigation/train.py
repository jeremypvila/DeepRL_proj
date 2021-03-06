'''
Main Code to train the agent for project p1 for Udacity Deep RL Nanodegree

Parts of this code were adapted from the dqn lesson of the course

8/26/18
'''

import gym
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from agent import Agent
from unityagents import UnityEnvironment


# please do not modify the line below
env = UnityEnvironment(file_name="Banana_Linux_NoVis/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name];

# number of actions and observations
action_size = brain.vector_action_space_size
state_size = brain.vector_observation_space_size

#Hyperparameters to tune
batch_size = 64
gamma=0.99  # discount factor
lr=1e-4  # learning rate of the Adam optimizer (Optimizing MSE loss)
agent = Agent(state_size=state_size, action_size=action_size, gamma=gamma, batch_size=batch_size, lr=lr, seed=0)
# agent.load_weights()

# Main DQN loop pulled from dqn section of udacity course.  Adapted to the new "BananaBrain" environment
min_score = 13.0
def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, gamma=0.99):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0] # Initialize state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)

            # must adapt to unit ML environment
            # Take action and get next_state, reward, and done status
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # get the status

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=min_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save({'Q_est': agent.qnetwork_local.state_dict(), 
                        'Q_target': agent.qnetwork_target.state_dict(),
                        'opt': agent.optimizer.state_dict()}, 'navigation.pth')
            break
    return scores

scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("trained_scores.png")