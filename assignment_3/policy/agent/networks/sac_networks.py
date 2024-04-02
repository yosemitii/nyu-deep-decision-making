import torch
from torch import nn

import utils

class SacActor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, action_shape),
            nn.Tanh()
        )

        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, action_shape),
        )
        self.apply(utils.weight_init)

    def forward(self, obs):
        # TODO: Define the forward pass
        # mu = None
        mu = self.mu_head(self.policy(obs))
        sigma = self.sigma_head(self.policy(obs))
        sigma = torch.clamp(sigma, 1e-6, 1)
        # std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, sigma)
        return dist


class Value(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        # TODO: Define the Q network
        # self.Q = None
        self.Q = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),  # Concatenate state and action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # TODO: Define the forward pass
        # Hint: Pass the state and action through the network and return the Q value
        # q = None
        q = self.Q(torch.cat([obs, action], dim=-1))
        return q