"""
    The file contains the PPO class to train with.
    NOTE: Original PPO pseudocode can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gymnasium as gym
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal

from .base_alg import BasePolicyGradient


class PPO(BasePolicyGradient):
    """
        This is the PPO class we will use as our model in main.py
    """

    def __init__(self, policy_class, env, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """
        # self.env = env
        # self.obs_dim = env.observation_space.shape[0]
        # self.act_dim = env.action_space.shape[0]

        super().__init__(policy_class, env, **hyperparameters)
        # TODO: Initialize critic network
        # self.env = env
        # self.obs_dim = env.observation_space.shape[0]
        # self.act_dim = env.action_space.shape[0]
        # Initialize actor and critic networks
        self.critic = policy_class(self.obs_dim, 1)

        # TODO: Initialize optimizers critic
        # self.critic_optim = None
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        print(
            f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(
            f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        while t_so_far < total_timesteps:
            # We're collecting our batch simulations here
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # TODO: Calculate advantage
            # A_k = None
            V, _ = self.evaluate(batch_obs, batch_acts, batch_rtgs)
            A_k = batch_rtgs - V.detach().squeeze()
            # A_k = batch_rtgs

            # One of the only tricks we use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):
                # TODO: figure out the actor and the critic losses for our algorithm
                # actor_loss = None
                # critic_loss = None
                V_k, log_probs = self.evaluate(batch_obs, batch_acts, batch_rtgs)
                r_k_theta = (log_probs - batch_log_probs).exp()

                clipped_r_k = torch.clamp(r_k_theta, 1-self.clip, 1 + self.clip)

                # L_CLIP
                surr1 = r_k_theta * A_k
                surr2 = clipped_r_k * A_k
                actor_loss = -torch.min(surr1, surr2).mean()
                # actor_loss = - (torch.min(r_k_theta, clipped_r_k) * A_k).mean()
                # critic_loss = A_k + V_k
                critic_loss = F.mse_loss(V_k.squeeze(), batch_rtgs)

                # actor_loss = None
                # critic_loss = None

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def evaluate(self, batch_obs, batch_acts, batch_rtgs):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
                batch_rtgs - the rewards-to-go calculated in the most recently collected
                                batch as a tensor. Shape: (number of timesteps in batch)
        """
        # TODO: Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        # V = None
        V = self.critic(batch_obs)

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs
