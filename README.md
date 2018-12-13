# Project 2: Continuous Control

### Introduction

The goal of this project is to train an agent to operate in the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

### Instructions

The project requires the installation of the environment provided by Udacity; see the detailed instructions [here](https://classroom.udacity.com/nanodegrees/nd893/parts/286e7d2c-e00c-4146-a5f2-a490e0f23eda/modules/089d6d51-cae8-4d4b-84c6-9bbe58b8b869/lessons/5b822b1d-5c89-4fd5-9b52-a02ddcfd3385/concepts/2303cf3b-d5dc-42b0-8d15-e379fa76c6d5). The following Python 3.5 libraries are required as well (if not provided by the Udacity DRLND environment): `unityagents`,`numpy`,`torch`,`matplotlib`.

After the enviromnent is set up and activated, run `python train_agent.py` to train the agent and `python demo_trained_agent.py` to see how the trained agent performs.