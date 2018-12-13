[//]: # (Image References)

[image1]: ./Figure_1.png "Rewards Plot"

# Project 2: Continuous Control - Report

### Solution

#### Approach

My initial approach was to try a vanilla DDPG learning algorithm described in Lectures 14-17 of Lesson 5. The training code (adapted from the lectures/Udacity's Deep Reinforcement Learning Github Repository) is contained in `train_agent.py` and the agent and network model in `ddpg_agent.py` and `model.py` accordingly.

#### Neural Network Architecture

##### Actor
The Actor network is a 4-layer fully connected neural network (with ReLu activations for the hidden layers and a Tanh activation for the output layer) with 33 units in the input layer, 128 units in each of the hidden layers and 4 units in the output layer. The output of the first hidden layer is batch-normalized for performance and stability improvement (based on a recommendation from the DRLND Slack channel).

##### Critic
The Critic network is a 4-layer fully connected neural network (with ReLu activations) with 33 units in the input layer, 128 units in each of the hidden layers and 1 unit in the output layer. The output of the first hidden layer is batch-normalized for performance and stability improvement (based on a recommendation from the DRLND Slack channel).

#### Hyperparameters
The initial approach used the following hyperparameters:

DDPG:
- n_episodes (int): maximum number of training episodes: *1000*
- max_t (int): maximum number of timesteps per episode: *3000*

Agent:
- BUFFER_SIZE = *int(1e5)*  # replay buffer size
- BATCH_SIZE minibatch size: *128*
- GAMMA: discount factor: *0.99*
- TAU: for soft update of target parameters: *1e-3*
- LR_ACTOR: learning rate of the actor: *2e-4*
- LR_CRITIC: learning rate of the critic: *2e-4*
- WEIGHT_DECAY: L2 weight decay: *0*

#### Results

After some experiments with the structure of actor and critic networks and learning rates, the basic vanilla DDPG with parameters described above performed adequately and trained the agent to solve the environment with average score of +30 in 216 episodes.

See the rewards plot below:

![Rewards Plot][image1]

Other network configurations (listed below) yielded considerably worse results:
- Actor and Critic with 2 hidden layers of 64 neurons each - score 1.72 after 1000 episodes
- Actor and Critic with 2 hidden layers of 256 neurons each - score 24.27 after 1000 episodes
- Actor and Critic with 1st hidden layers of 64 neurons and 2nd hidden layer with 128 neurons - score 18.82 after 1000 episodes
- Actor and Critic with 1st hidden layers of 128 neurons and 2nd hidden layer with 64 neurons - score 1.3 after 1000 episodes
- Actor and Critic with 3 hidden layers of 64 neurons each - score 1.67 after 1000 episodes

### Future work
As mentioned before, it seems that even a standard DDPG performs adequately. However, if we wanted to experiment further, a set of non-interacting, parallel copies of the same agent, coupled with an algorithm like PPO, A3C, or D4PG might be a way to go.