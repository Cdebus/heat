import sys
import math
import random
import numpy as np

import heat as ht


class KMeans:
    def __init__(self, n_clusters, max_iter=1000, tol=1e-4, random_state=42):
        # TODO: document me
        # TODO: extend the parameter list
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def initialize_centroids(k, dimensions, seed, device):
        # TODO: document me
        # TODO: extend me with further initialization methods
        # zero-centered uniform random distribution in [-1, 1]
        ht.random.set_gseed(seed)
        return ht.random.uniform(low=-1.0, high=1.0, size=(1, dimensions, k), device=device)

    @staticmethod
    def initialize_centroids_databased(k, dimensions, data):
        # Initialize centroids with random samples from the dataset
        # Samples will be equally distributed drawn from all involved processes
        # Each rank draws it samples, stores them into sub_centroids tensor and then distributes them amongst all processes.

        nproc = data.comm.size
        rank = data.comm.rank

        if data.split == 0:

            if rank < (k % nproc):
                num_samples = k // nproc + 1
            else:
                num_samples = k // nproc

        else:
            raise NotImplementedError('Not implemented for other splitting-axes')

        local_centroids = ht.empty((num_samples, dimensions))

        for i in range(num_samples):
            x = random.randint(0, data.lshape[0] - 1)
            local_centroids._tensor__array[i, :] = data._tensor__array[x, :]

        recv_counts = np.full((nproc,), k // nproc)
        recv_counts[:k % nproc] += 1

        recv_displs = np.zeros((nproc,), dtype=recv_counts.dtype)
        np.cumsum(recv_counts[:-1], out=recv_displs[1:])

        gathered = ht.empty((k, data.gshape[1]))
        print("(Rank: {:2d})  ".format(rank), tuple(recv_counts), tuple(recv_displs))

        data.comm.Allgatherv(local_centroids._tensor__array,
                             (gathered._tensor__array, tuple(recv_counts), tuple(recv_displs),), recv_axis=0)

        out = ht.transpose(gathered)
        out = out.expand_dims(axis=0)
        
        return out



    def fit(self, data):
        # TODO: document me
        data = data.expand_dims(axis=2)

        # initialize the centroids randomly
        centroids = self.initialize_centroids_databased(self.n_clusters, data.shape[1], data)
            
        new_centroids = centroids.copy()

        for epoch in range(self.max_iter):
            # calculate the distance matrix and determine the closest centroid
            distances = ((data - centroids) ** 2).sum(axis=1, keepdim=True)
            matching_centroids = distances.argmin(axis=2, keepdim=True)

            # update the centroids
            for i in range(self.n_clusters):
                selection = (matching_centroids == i).astype(ht.int64)
                new_centroids[:, :, i:i + 1] = ((data * selection).sum(axis=0, keepdim=True) /
                                                selection.sum(axis=0, keepdim=True).clip(1.0, sys.maxsize))

            # check whether centroid movement has converged
            epsilon = ((centroids - new_centroids) ** 2).sum()
            centroids = new_centroids.copy()
            if self.tol is not None and epsilon <= self.tol:
                break

        return centroids
