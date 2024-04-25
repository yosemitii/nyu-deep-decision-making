"""
    This file will run REINFORCE or PPO code
    with the input seed and environment.
"""

import gymnasium as gym
import os
import argparse

# Import ppo files
from ppo.ppo import PPO
from ppo.reinforce import REINFORCE
from ppo.network import FeedForwardNN
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
def train_ppo(args):
    """
        Trains with PPO on specified environment.

        Parameters:
            args - the arguments defined in main.

        Return:
            None
    """
    # Store hyperparameters and total timesteps to run by environment
    hyperparameters = {}
    total_timesteps = 0
    if args.env == 'Pendulum-v1':
        hyperparameters = {'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 200, 'gamma': 0.99, 'n_updates_per_iteration': 10,
                            'lr': 3e-4, 'clip': 0.2, 'save_freq': 1e6, 'seed': args.seed}
        total_timesteps = 2005000
    elif args.env == 'BipedalWalker-v3':
        hyperparameters = {'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 1600, 'gamma': 0.99, 'n_updates_per_iteration': 10,
                            'lr': 2.5e-4, 'clip': 0.2, 'save_freq': 1e6, 'seed': args.seed}
        total_timesteps = 6405000
    elif args.env == 'LunarLanderContinuous-v2':
        hyperparameters = {'timesteps_per_batch': 1024, 'max_timesteps_per_episode': 1000, 'gamma': 0.999, 'n_updates_per_iteration': 4,
                            'lr': 2.5e-4, 'clip': 0.2, 'save_freq': 1e6, 'seed': args.seed}
        total_timesteps = 6005000
    else:
        raise ValueError("Unrecognized environment, please specify the hyperparameters first.")

    # Make the environment and model, and train
    env = gym.make(args.env)
    model = PPO(FeedForwardNN, env, **hyperparameters)
    model.learn(total_timesteps)
    return model

def train_reinforce(args):
    """
        Trains with REINFORCE on specified environment.

        Parameters:
            args - the arguments defined in main.

        Return:
            None
    """
    # Store hyperparameters and total timesteps to run by environment
    hyperparameters = {}
    total_timesteps = 0
    if args.env == 'Pendulum-v1':
        hyperparameters = {'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 200, 'gamma': 0.99, 'n_updates_per_iteration': 1,
                            'lr': 3e-4, 'clip': 0.2, 'save_freq': 1e6, 'seed': args.seed}
        total_timesteps = 2005000
    elif args.env == 'BipedalWalker-v3':
        hyperparameters = {'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 1600, 'gamma': 0.99, 'n_updates_per_iteration': 1,
                            'lr': 2.5e-4, 'clip': 0.2, 'save_freq': 1e6, 'seed': args.seed}
        total_timesteps = 6405000
    elif args.env == 'LunarLanderContinuous-v2':
        hyperparameters = {'timesteps_per_batch': 1024, 'max_timesteps_per_episode': 1000, 'gamma': 0.999, 'n_updates_per_iteration': 1,
                            'lr': 2.5e-4, 'clip': 0.2, 'save_freq': 1e6, 'seed': args.seed}
        total_timesteps = 6005000
    else:
        raise ValueError("Unrecognized environment, please specify the hyperparameters first.")

    # Make the environment and model, and train
    env = gym.make(args.env)
    model = REINFORCE(FeedForwardNN, env, **hyperparameters)
    model.learn(total_timesteps)
    return model

def main(args):
    """
        An intermediate function that will call either REINFORCE learn or PPO learn.

        Parameters:
            args - the arguments defined below

        Return:
            None
    """
    res = {}
    x_timestamp = {}
    y_reward = {}
    # for env in ['Pendulum-v1', 'BipedalWalker-v3', 'LunarLanderContinuous-v2']:

    plt.figure(figsize=(10, 8))
    for i in range(3):
        seed = datetime.now().minute * 60 + datetime.now().second
        args.seed = seed
        if args.alg == 'PPO':
            model = train_ppo(args)
        elif args.alg == 'reinforce':
            model = train_reinforce(args)
        else:
            raise ValueError(f'Algorithm {args.alg} not defined; options are reinforce or PPO.')
        res['time'+str(i)] = model.logger['time']
        res['rewards'+str(i)] = model.logger['rewards']
        plt.plot(res['time' + str(i)], res['rewards' + str(i)], label=f'seed={seed}')
    plt.legend()
    plt.grid(True)
    plt.show()

    max_len = 0
    max_id = -1
    for i in range(3):
        if len(res['time' + str(i)]) > max_len:
            max_len = len(res['time' + str(i)])
            max_id = i

    to_stack = []
    for i in range(3):
        curr_len = len(res['rewards'+str(i)])
        to_stack.append(np.pad(res['rewards'+str(i)], (0, max_len-curr_len), 'constant', constant_values=np.nan))
    stacked = np.stack(to_stack)
    mean_reward = np.nanmean(stacked, axis=0)

    time_stamp = res['time'+str(max_id)]
    plt.figure(figsize=(10, 8))
    plt.plot(time_stamp, mean_reward, label=f'mean reward {args.alg}')
    plt.title(f'mean reward vs timestamp {args.env}')
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', dest='alg', type=str, default='reinforce')        # Formal name of our algorithm
    parser.add_argument('--seed', dest='seed', type=int, default=None)             # An int for our seed
    parser.add_argument('--env', dest='env', type=str, default='')                 # Formal name of environment

    args = parser.parse_args()

    main(args)
