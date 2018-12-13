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

# initialize the agent
agent = Agent(state_size=33, action_size=4, random_seed=42)

# load the weights from files
agent.actor_local.load_state_dict(torch.load('checkpoint_actor_30.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic_30.pth'))

env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]             # get the current state    
score = 0                                           # initialize the score
while True:
    action = agent.act(state)                      # get action from the agent
    env_info = env.step(action)[brain_name]        # send the action to the environment
    state = env_info.vector_observations[0]        # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    print('\rCurrent Score: {:.2f}'.format(score), end="") # print current score
    if done:                                       # exit loop if episode finished
        break
print('Total score: {}'.format(score))
env.close()