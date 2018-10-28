'''
Main script that was used for laoding the environment

'''

from unityagents import UnityEnvironment
from ddpg_agent import Agent
import numpy as np 

# Load the environment 
env = UnityEnvironment(file_name='/home/jeremy/projects/DeepRL_proj/p2_continuous_control/Reacher_Linux_NoVis/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=0)

episodes = 2
max_time = 300

scores_deque = deque(maxlen=100)
scores = []
max_score = -np.Inf
for ep in range(1, episodes+1):
	state = env.reset()
	agent.reset()
	score = 0
	for t in range(max_time):
		action = agent.act(state)
		next_state, reward, done, _ = env.step(action)
		agent.step(state, action, reward, next_state, done)
		state = next_state
		score += reward
		if done:
			break 
		scores_deque.append(score)
		scores.append(score)
		print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
		if i_episode % 100 == 0:
			torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
			torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
			print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque))) 

# Load in the ddpg agent

'''
env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
'''

env.close()