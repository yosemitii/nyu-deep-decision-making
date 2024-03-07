import torch
from torch import nn

import torchvision.transforms as T
from torch.nn import functional as F
import torch.distributions as D

import utils
from agent.networks.encoder import Encoder
from agent.networks.kmeans_discretizer import KMeansDiscretizer

class FocalLoss(nn.Module):
	
	def __init__(self, gamma: float = 0, size_average: bool = True):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.size_average = size_average

	def forward(self, input, target):
		"""
		Args:
			input: (N, B), where B = number of bins
			target: (N, )
		"""
		logpt = F.log_softmax(input, dim=-1)
		logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
		pt = logpt.exp()

		loss = -1 * (1 - pt) ** self.gamma * logpt
		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()

class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape, hidden_dim, nbins):
		super().__init__()

		self._output_dim = action_shape[0]

		# TODO: Define the policy network
		# Hint: There must be a common trunk followed by two heads - one for binning and one for offsets
		self.trunk = nn.Sequential(
			nn.Linear(repr_dim * 2, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True),
		)

		self.bin_head = nn.Sequential(
			nn.Linear(hidden_dim, nbins),
			# nn.Tanh()
			nn.Softmax(dim=1)
		)

		self.offset_head = nn.Sequential(
			nn.Linear(hidden_dim, action_shape[0]),
			nn.Tanh()
		)

		self.apply(utils.weight_init)

	def forward(self, obs, std, cluster_centers=None):
		# TODO: Implement the forward pass, return bin_logits, and offset
		# reshaped_centroids = cluster_centers.reshape(1, cluster_centers.shape[0] * cluster_centers.shape[1])
		# input_repr = torch.cat((obs, reshaped_centroids.expand(obs.shape[0], -1)), dim=1)
		features = self.trunk(obs)
		bin_logits = torch.tanh(self.bin_head(features))
		offset = self.offset_head(features)
		return bin_logits, offset

class Agent:
	def __init__(self, obs_shape, action_shape, device, lr, hidden_dim, stddev_schedule, 
	      		 stddev_clip, use_tb, obs_type, nbins, kmeans_iters, offset_weight,
				 offset_loss_weight):
		self.device = device
		self.lr = lr
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.use_tb = use_tb
		self.use_encoder = True if obs_type=='pixels' else False
		self.nbins = nbins
		self.kmeans_iters = kmeans_iters
		self.offset_weight = offset_weight
		self.offset_loss_weight = offset_loss_weight
		
		# actor parameters
		self._act_dim = action_shape[0]

		# discretizer
		self.discretizer = KMeansDiscretizer(num_bins=self.nbins, kmeans_iters=self.kmeans_iters)

		# TODO: Define the encoder (for pixels)
		if self.use_encoder:
			self.encoder = None
			repr_dim = self.encoder.repr_dim
		else:
			# TODO: Define the representation dimension for non-pixel observations
			repr_dim = obs_shape[0]

		# TODO: Define the actor
		self.actor = Actor(repr_dim, action_shape, hidden_dim, nbins)

		# loss
		self.criterion = FocalLoss(gamma=2.0)

		# TODO: Define optimizers
		if self.use_encoder:
			self.encoder_opt = None
		# self.actor_opt = None
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

		# data augmentation
		if self.use_encoder:
			self.aug = utils.RandomShiftsAug(pad=4)

		self.train()

	def __repr__(self):
		return "bet"
	
	def train(self, training=True):
		self.training = training
		if training:
			if self.use_encoder:
				self.encoder.train(training)
			self.actor.train(training)
		else:
			if self.use_encoder:
				self.encoder.eval()
			self.actor.eval()
	
	def compute_action_bins(self, actions):
		# Compute nbins bin centers using k-nearest neighbors algorithm 
		actions = torch.as_tensor(actions, device=self.device).float()
		self.discretizer.fit(actions)
		self.cluster_centers = self.discretizer.bin_centers.float().to(self.device)
		
	def find_closest_cluster(self, actions) -> torch.Tensor:
		# TODO: Return the index of closest cluster center for each action in actions
		distances = torch.cdist(actions, self.cluster_centers, p=2)
		closest_cluster_center = distances.argmin(dim=1, keepdim=True)
		return closest_cluster_center
		

	def act(self, obs, goal, step):
		obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
		goal = torch.as_tensor(goal, device=self.device).float().unsqueeze(0)
		
		# TODO: Obtain bin_logits and offset from the actor
		stddev = utils.schedule(self.stddev_schedule, step)
		bin_logits, offset = self.actor.forward(torch.cat((obs, goal), dim=1), stddev, self.cluster_centers)

		# TODO: Compute base action (Hint: Use the bin_logits)
		argmax_id = torch.argmax(bin_logits).item()

		base_action = self.cluster_centers[argmax_id]
		# print(argmax_id, base_action)
		action = base_action + self.offset_weight * offset
		return action.cpu().numpy()[0]

	def update(self, expert_replay_iter, step):
		metrics = dict()

		batch = next(expert_replay_iter)
		obs, action, goal = utils.to_torch(batch, self.device)
		obs, action, goal = obs.float(), action.float(), goal.float()

		# augment
		if self.use_encoder:
			# TODO: Augment the observations and encode them (for pixels)
			pass

		# TODO: Compute bin_logits and offset from the actor
		stddev = utils.schedule(self.stddev_schedule, step)
		bin_logits, offset = self.actor.forward(torch.cat((obs, goal), dim=1), stddev, self.cluster_centers)

		# TODO: Compute discrete loss on bins and offset loss
		argmax_id = torch.argmax(bin_logits, dim=1)
		ground_truth_logits = self.find_closest_cluster(action)
		residual_action = action - self.cluster_centers[argmax_id]
		# discrete_loss = self.criterion(self.cluster_centers, self.cluster_centers[argmax_id])
		discrete_loss = self.criterion(bin_logits, ground_truth_logits)
		mse_loss_fn = nn.MSELoss()
		offset_loss = mse_loss_fn(residual_action, offset)

		# actor loss
		actor_loss = discrete_loss + self.offset_loss_weight * offset_loss

		# TODO: Update the actor (and encoder for pixels)
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['discrete_loss'] = discrete_loss.mean().item()
			metrics['offset_loss'] = offset_loss.mean().item() * self.offset_loss_weight
			metrics['logits_entropy'] = D.Categorical(logits=bin_logits).entropy().mean().item()

		return metrics

	def save_snapshot(self):
		keys_to_save = ['actor', 'cluster_centers']
		if self.use_encoder:
			keys_to_save += ['encoder']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload):
		for k, v in payload.items():
			self.__dict__[k] = v

		# Update optimizers
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
