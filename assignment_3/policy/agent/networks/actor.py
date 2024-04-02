import torch
from torch import nn

import utils

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        # TODO: Define the policy network
        # self.policy = None
        self.policy = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_shape),
            nn.Tanh()
        )
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        # TODO: Define the forward pass
        # mu = None
        mu = self.policy(obs)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist



