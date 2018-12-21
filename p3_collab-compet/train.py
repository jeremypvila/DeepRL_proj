'''
Main script to train the multi agent ddpg algoirthm on the Tennis environment

Some methods taken from DDPG file in Udacity Deep Learning course and from my previous p2 project
MADDPG implementation based on Udacity Deep Learning course and MADDPG paper

Coded by:
Jeremy Vila
12/15/18

'''

from unityagents import UnityEnvironment
from maddpg_agent import MADDPG
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch

# Load the environment 
env = UnityEnvironment(file_name='Tennis_Linux_NoVis/Tennis.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# get number of agenets.  Confirm there is one
num_agents = len(env_info.agents)

# Get action size
action_size = brain.vector_action_space_size

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]

agent = MADDPG(state_size=state_size, action_size=action_size, num_agents=num_agents, seed=42)
# agent.load_weights()

episodes = 10000  # Number of episodes
max_time = 1000  # Max number of time steps per episode
max_score = 0.6  # Average score to beat over length 100 window

# Score lists
scores_deque = deque(maxlen=100)
all_scores = []
all_scores_mean = []
all_scores_std = []

# Main training loop
for ep in range(0, episodes):
	env_info = env.reset(train_mode=True)[brain_name]
	states = env_info.vector_observations
	scores = np.zeros(num_agents) 

	# state = env.reset()
	agent.reset()  # Resets the noise in the agent
	scores = np.zeros(num_agents)
	# Step through time steps and learn the actor and critic
	for t in range(max_time):
		actions = agent.act(states)									# Get actions from policy (for each agent)
		env_info = env.step(actions)[brain_name]           			# Perform actions in environment
		next_states = env_info.vector_observations					# get next state (for each agent)
		rewards = env_info.rewards                         			# get reward (for each agent)
		dones = env_info.local_done									# get dones (for each agent)
		agent.step(states, actions, rewards, next_states, dones)	# Add experience to buffer
		states = next_states										# Reset states
		scores += rewards											# Accumulate rewards									
		if np.any(dones):
			break 

	scores_deque.append(np.max(scores))
	all_scores.append(np.max(scores))  # Add to total list of scores
	all_scores_mean.append(np.mean(scores_deque))
	all_scores_std.append(np.std(scores_deque))

	# Print results as they are computed
	mn_score = np.mean(scores_deque)
	print('\rEpisode {}\tAverage Score: {:.3f}\tScore: {:.3f}'.format(ep+1, mn_score, np.max(scores)), end="")
	if ep+1 % 100 == 0 or mn_score > max_score:
		agent.save()
	if mn_score > max_score:
		break

# plot the scores
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True, sharey=False)

ind = list(range(500,len(all_scores)))
N = len(ind)

# plot the sores by episode
ax1.plot(ind, np.asarray(all_scores)[ind])
ax1.set_xlim(ind[0], ind[-1]+1)
ax1.set_ylabel('Score')
ax1.set_xlabel('Episode #')
ax1.set_title('Episodic Scores')

mn = np.asarray(all_scores_mean)[ind]
std = np.asarray(all_scores_std)[ind]
# plot the average of these scores

ax2.axhline(y=max_score, xmin=0.0, xmax=1.0, color='r', linestyle='--', linewidth=0.7, alpha=0.9)
ax2.plot(ind, np.asarray(all_scores_mean)[ind])
ax2.fill_between(ind, mn+std, mn-std, facecolor='gray', alpha=0.2)
ax2.set_ylabel('Score')
ax2.set_xlabel('Episode')
ax2.set_title('Average score with uncertainty')

f.tight_layout()
plt.savefig("trained_scores.png")