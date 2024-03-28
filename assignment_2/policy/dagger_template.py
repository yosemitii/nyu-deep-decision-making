"""
[CSCI-GA 3033-090] Special Topics: Deep Decision Making & Reinforcement Learning

Homework - 2, DAgger
Deadline: March 8, 2024 11:59 PM.

Complete the code template provided in dagger.py, with the right 
code in every TODO section, to implement DAgger. Attach the completed 
file in your submission.
"""

import tqdm
import hydra
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import gym
import particle_envs

from utils import weight_init, ExpertBuffer
from video import VideoRecorder

from matplotlib import pyplot as plt

class Actor(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_dim):
		super(Actor, self).__init__()
		# TODO define your own network
		self.policy = nn.Sequential(
			nn.Linear(input_dim*2, hidden_dim),
			nn.ReLU(inplace=True),
			# nn.Linear(hidden_dim, hidden_dim),
			# nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, action_dim),
			nn.Tanh()
		)
		self.apply(weight_init)

	def forward(self, obs, goal):
		# TODO pass it forward through your network.
		output_action = self.policy(torch.cat((goal, obs), dim=-1))
		return output_action


def initialize_model_and_optim(cfg):
	# TODO write a function that creates a model and associated optimizer
	# given the config object.
	# pass
	model = Actor(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim)
	optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
	return model, optimizer


class Workspace:
	def __init__(self, cfg):
		self._work_dir = os.getcwd()
		print(f'workspace: {self._work_dir}')

		self.cfg = cfg

		self.device = torch.device(cfg.device)
		self.train_env = gym.make('particle-v0', height=cfg.height, width=cfg.width, step_size=cfg.step_size, reward_type='dense')
		self.train_env = gym.make('particle-v0', height=cfg.height, width=cfg.width, step_size=cfg.step_size, reward_type='dense')

		self.expert_buffer = ExpertBuffer(cfg.experience_buffer_len, 
										  self.train_env.observation_space.shape,
										  self.train_env.action_space.shape)
		
		self.model, self.optimizer = initialize_model_and_optim(cfg)

		# TODO: define a loss function
		# self.loss_function = None
		self.loss_function = nn.MSELoss()

		# init video recorder
		self.video_recorder = VideoRecorder(self._work_dir)
		
	def eval(self, ep_num):
		# A function that evaluates the 
		# Set the DAgger model to evaluation
		self.model.eval()
		# self.model.eval().to(self.device)

		avg_eval_reward = 0.
		avg_episode_length = 0.
		successes = 0
		for ep in range(self.cfg.num_eval_episodes):
			eval_reward = 0.
			ep_length = 0.
			obs_np = self.train_env.reset(reset_goal=True)
			goal_np = self.train_env.goal
			if ep == 0:
				self.video_recorder.init(self.train_env, enabled=True)
			# Need to be moved to torch from numpy first
			obs = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
			goal = torch.from_numpy(goal_np).float().to(self.device).unsqueeze(0)
			with torch.no_grad():
				action = self.model(obs, goal)
			done = False
			while not done:
				# Need to be moved to numpy from torch
				action = action.squeeze().detach().cpu().numpy()
				obs, reward, done, info = self.train_env.step(action)
				self.video_recorder.record(self.train_env)
				obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
				with torch.no_grad():
					action = self.model(obs, goal)
				eval_reward += reward
				ep_length += 1.
			avg_eval_reward += eval_reward
			avg_episode_length += ep_length
			if info['is_success']:
				successes += 1
		avg_eval_reward /= self.cfg.num_eval_episodes
		avg_episode_length /= self.cfg.num_eval_episodes
		success_rate = successes / self.cfg.num_eval_episodes
		self.video_recorder.save(f'eval_{ep_num}.mp4')
		return avg_eval_reward, avg_episode_length, success_rate


	def model_training_step(self):
		# A function that optimizes the model self.model using the optimizer 
		# self.optimizer using the experience  stored in self.expert_buffer.
		# Number of optimization step should be self.cfg.num_training_steps.

		# Set the model to training.
		self.model.train()
		# For num training steps, sample data from the training data.
		avg_loss = 0.
		iterable = tqdm.trange(self.cfg.num_training_steps)
		for _ in iterable:
			# TODO write the training code.
			obs_np, goal_np, expert_act_np = self.expert_buffer.sample(self.cfg.batch_size)
			self.optimizer.zero_grad()
			obs = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
			goal = torch.from_numpy(goal_np).float().to(self.device).unsqueeze(0)
			expert_act = torch.from_numpy(expert_act_np).float().to(self.device).unsqueeze(0)
			# predicted_actions = self.model(torch.tensor(obs, dtype=torch.float32),
			# 							   torch.tensor(goal, dtype=torch.float32))
			predicted_actions = self.model(obs, goal)
			loss = self.loss_function(predicted_actions, expert_act)
			loss.backward()

			self.optimizer.step()
			avg_loss += loss.item()

		avg_loss /= self.cfg.num_training_steps
		return avg_loss


	def run(self):
		train_loss, eval_reward, episode_length = None, 0, 0
		iterable = tqdm.trange(self.cfg.total_training_episodes)

		decay = 0.99
		threshold = 1

		num_expert_calls = []
		success_rate_series = []
		ep_series = []
		for ep_num in iterable:
			iterable.set_description('Collecting exp')
			# Set the DAGGER model to evaluation
			self.model.eval()
			ep_train_reward = 0.
			ep_length = 0.

			# TODO write the training loop.
			# 1. Roll out your current model on the environment.
			# 2. On each step, after calling either env.reset() or env.step(), call 
			#    env.get_expert_action() to get the expert action for the current 
			#    state of the environment.
			# 3. Store that observation alongside the expert action in the buffer.
			# 4. When you are training, use the stored obs and expert action.

			# Hints:
			# 1. You will need to convert your obs to a torch tensor before passing it
			#    into the model.
			# 2. You will need to convert your action predicted by the model to a numpy
			#    array before passing it to the environment.
			# 3. Make sure the actions from your model are always in the (-1, 1) range.
			# 4. Both the environment observation and the expert action needs to be a
			#    numpy array before being added to the environment.
			
			# TODO training loop here.
			# train_env
			obs = self.train_env.reset()
			goal = self.train_env.goal
			while ep_length < self.train_env.spec.max_episode_steps:

				rand_x = np.random.uniform(0.0, 1.0)

				expert_action = self.train_env.get_expert_action()
				self.expert_buffer.insert(obs, goal, expert_action)
				if rand_x <= threshold:
					action = expert_action
				else:
					action = self.model(torch.tensor(obs), torch.tensor(goal)).detach().cpu().numpy()
				obs, reward, done, info = self.train_env.step(action)
				ep_length += 1



			train_reward = ep_train_reward
			train_episode_length = ep_length

			if (ep_num+1) % self.cfg.train_every == 0:
				# Reinitialize model every time we are training
				iterable.set_description('Training model')
				# TODO train the model and set train_loss to the appropriate value.
				# Hint: in the DAgger algorithm, when do we initialize a new model?
				self.model, self.optimizer = initialize_model_and_optim(self.cfg)
				train_loss = self.model_training_step()
				threshold *= decay

			if ep_num % self.cfg.eval_every == 0:
				# Evaluation loop
				iterable.set_description('Evaluating model')
				eval_reward, episode_length, success_rate = self.eval(ep_num)

				num_expert_calls.append(self.train_env.expert_calls)
				success_rate_series.append(success_rate)
				ep_series.append(ep_num)


			iterable.set_postfix({
				'Train loss': train_loss,
				'Train reward': train_reward,
				'Eval reward': eval_reward
			})

		plt.plot(num_expert_calls, success_rate_series, label='success rate')
		plt.xlabel('num queries')
		plt.ylabel('success rate')
		plt.title(f'p = {decay} num queries vs success rate')
		plt.legend()
		plt.grid()
		plt.show()





			

@hydra.main(config_path='.', config_name='train')
def main(cfg):
	# In hydra, whatever is in the train.yaml file is passed on here
	# as the cfg object. To access any of the parameters in the file,
	# access them like cfg.param, for example the learning rate would
	# be cfg.lr
	workspace = Workspace(cfg)
	workspace.run()


if __name__ == '__main__':
	main()