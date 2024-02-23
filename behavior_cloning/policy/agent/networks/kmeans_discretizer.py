import tqdm
import torch

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
		N, D = x.shape  # Number of samples and dimensionality


		# initial_indices = torch.randperm(N)[:nbin]
		# bin_centers = x[initial_indices]
		bin_centers = torch.empty((nbin, D), dtype=x.dtype, device=x.device)
		initial_index = torch.randint(N, (1,)).item()
		bin_centers[0] = x[initial_index]

		for k in range(1, nbin):
			distances = torch.cdist(x, bin_centers[:k]).min(dim=1)[0]
			probabilities = distances / distances.sum()
			next_center_index = torch.multinomial(probabilities, 1).item()
			bin_centers[k] = x[next_center_index]

		for iteration in range(niter):
			distances = torch.cdist(x, bin_centers)  # Compute all distances between data points and centroids
			closest = distances.argmin(dim=1)  # Find the closest centroid for each data point

			# Update step - Compute new centroids as the mean of the points in each cluster
			for k in range(nbin):
				if (closest == k).any():
					bin_centers[k] = x[closest == k].mean(dim=0)
		# for iteration in range(niter):
		#
		# 	distances = torch.cdist(x, bin_centers)  # Compute all distances between data points and centroids
		# 	closest = distances.argmin(dim=1)  # Find the closest centroid for each data point
		#
		# 	# Step 3: Update step - Compute new centroids as the mean of the points in each cluster
		# 	new_centroids = torch.stack([x[closest == k].mean(dim=0) for k in range(nbin)])
		#
		# 	# Check for convergence (if centroids do not change)
		# 	if torch.allclose(bin_centers, new_centroids, atol=1e-4):
		# 		break
		#
		# 	bin_centers = new_centroids

		return bin_centers
		# return bin_centers