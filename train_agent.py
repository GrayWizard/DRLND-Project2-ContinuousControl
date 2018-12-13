# Note: this code is heavily based on the lecture DDPG code from Udacity DRN nanodegree
from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent

# set up the environment
env = UnityEnvironment(file_name='d:\Courses\DRLND\DRLND-Project2-ContinuousControl\Reacher_Windows_x86_One\Reacher.exe', no_graphics=False)

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

# initialize the agent
agent = Agent(state_size=state_size, action_size=action_size, random_seed=42)

def ddpg(n_episodes=1000, max_t=3000, print_every=100):
    """Deep Deterministic Policy Gradients Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): frequency of printing/saving the state of the learning
    """    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=print_every)  # last 'print_every' scores
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        agent.reset()                                      # reset the agent 
        state = env_info.vector_observations[0]            # get the state
        score = 0
        for t in range(max_t):
            action = agent.act(state)                      # get action from the agent
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # check if done
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward                                # add the reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        # Save the learning state every 'print_every' episodes
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        # If the desired score is achieved, save the learning state and exit
        if np.mean(scores_window)>=30.0:
            solved=True
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_30.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_30.pth')
            break
    return scores

# train
scores = ddpg()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()