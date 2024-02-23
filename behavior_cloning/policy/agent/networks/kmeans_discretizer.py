import tqdm
import torch
from sklearn.cluster import KMeans

class KMeansDiscretizer:
	"""
	Simplified and modified version of KMeans algorithm from sklearn.

	Code borrowed from https://github.com/notmahi/miniBET/blob/main/behavior_transformer/bet.py
	"""

	def __init__(
		self,
		num_bins: int = 100,
		kmeans_iters: int = 50,
	):
		super().__init__()
		self.n_bins = num_bins
		self.kmeans_iters = kmeans_iters

	def fit(self, input_actions: torch.Tensor) -> None:
		self.bin_centers = KMeansDiscretizer._kmeans(
			input_actions, nbin=self.n_bins, niter=self.kmeans_iters
		)

	@classmethod
	def _kmeans(cls, x: torch.Tensor, nbin: int = 512, niter: int = 50):
		"""
		Function implementing the KMeans algorithm.

		Args:
			x: torch.Tensor: Input data - Shape: (N, D)
			nbin: int: Number of bins
			niter: int: Number of iterations
		"""

		# TODO: Implement KMeans algorithm to cluster x into nbin bins. Return the bin centers - shape (nbin, x.shape[-1])
		# bin_centers = None
		x_numpy = x.numpy()  # Convert PyTorch tensor to NumPy array

		# Initialize KMeans with the desired parameters
		kmeans = KMeans(n_clusters=nbin, max_iter=niter, random_state=0)

		# Fit the KMeans algorithm to the data
		kmeans.fit(x_numpy)

		# Get the cluster labels (assignments) for each data point
		cluster_labels = kmeans.labels_

		# Optional: Convert cluster labels back to a PyTorch tensor
		cluster_labels_tensor = torch.tensor(cluster_labels)

		# If you need the cluster centers as a PyTorch tensor
		cluster_centers = kmeans.cluster_centers_
		cluster_centers_tensor = torch.tensor(cluster_centers)
		return cluster_centers_tensor
		# N, D = x.shape  # Number of samples and dimensionality
		#
		#
		# # initial_indices = torch.randperm(N)[:nbin]
		# # bin_centers = x[initial_indices]
		# bin_centers = torch.empty((nbin, D), dtype=x.dtype, device=x.device)
		# initial_index = torch.randint(N, (1,)).item()
		# bin_centers[0] = x[initial_index]
		#
		# for k in range(1, nbin):
		# 	distances = torch.cdist(x, bin_centers[:k]).min(dim=1)[0]
		# 	probabilities = distances / distances.sum()
		# 	next_center_index = torch.multinomial(probabilities, 1).item()
		# 	bin_centers[k] = x[next_center_index]
		#
		# for iteration in range(niter):
		# 	distances = torch.cdist(x, bin_centers)  # Compute all distances between data points and centroids
		# 	closest = distances.argmin(dim=1)  # Find the closest centroid for each data point
		#
		# 	# Update step - Compute new centroids as the mean of the points in each cluster
		# 	for k in range(nbin):
		# 		if (closest == k).any():
		# 			bin_centers[k] = x[closest == k].mean(dim=0)
		#
		# return bin_centers
		# return bin_centers