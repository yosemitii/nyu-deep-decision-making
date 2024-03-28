import numpy as np
import torch
import torch.nn as nn
from torch import distributions as pyd

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class ExpertBuffer:
    """
    An expert buffer class. This class stores the demonstration (obs, action) set from the expert.
    While this class is completely created for you, you should still pay attention to how it's coded.
    Inspired by https://github.com/denisyarats/pytorch_sac
    """
    def __init__(self, max_length, obs_shape, action_shape):
        # Creates a buffer to hold all the expert demonstrations
        self._obs_buffer = np.empty(shape=(max_length, *obs_shape), dtype=np.float64)
        self._goal_buffer = np.empty(shape=(max_length, *obs_shape), dtype=np.float64)
        self._expert_action_buffer = np.empty(shape=(max_length, *action_shape), dtype=np.float64)
        self._current_index = 0
        self._is_full = False
        self.capacity = max_length

    def insert(self, obs, goal, action):
        # Insert an image observation along with the expert action in the buffer.
        insert_idx = self._current_index
        np.copyto(self._obs_buffer[insert_idx], obs)
        np.copyto(self._goal_buffer[insert_idx], goal)
        np.copyto(self._expert_action_buffer[insert_idx], action)
        self._current_index = (self._current_index + 1) % self.capacity
        if self._current_index == 0:
            self._is_full = True

    def __len__(self):
        return self.capacity if self._is_full else self._current_index

    def sample(self, batch_size=256):
        # Sample a batch of size batch_size of observations and expert actions.
        current_length = self.__len__()
        batch_indices = np.random.randint(low=0, high=current_length, size=batch_size)

        batch_obs = self._obs_buffer[batch_indices]
        batch_goal = self._goal_buffer[batch_indices]
        batch_action = self._expert_action_buffer[batch_indices]
        return batch_obs, batch_goal, batch_action
