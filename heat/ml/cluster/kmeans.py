import sys
import torch
from heat.core.communication import MPI_WORLD
import random
import numpy as np
import matplotlib.pyplot as plt
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

        if (data.split == None) or (data.split == 0):
            if rank < (k % nproc):
                num_samples = k // nproc + 1
            else:
                num_samples = k // nproc

        else:
            raise NotImplementedError('Not implemented for other splitting-axes')

        local_centroids = torch.empty((num_samples, dimensions))
        for i in range(num_samples):
            x = random.randint(0, data.lshape[0] - 1)
            local_centroids[i, :] = data._DNDarray__array[x, :, 0]


        recv_counts = np.full((nproc,), k // nproc)
        recv_counts[:k % nproc] += 1

        recv_displs = np.zeros((nproc,), dtype=recv_counts.dtype)
        np.cumsum(recv_counts[:-1], out=recv_displs[1:])

        device = ht.devices.sanitize_device(data.device)
        gathered = torch.empty((k, data.gshape[1]),  device=device.torch_device)

        data.comm.Allgatherv(local_centroids,
                             (gathered, tuple(recv_counts), tuple(recv_displs),), recv_axis=0)

        gathered = torch.transpose(gathered, 0, 1)
        out = ht.dndarray.DNDarray(gathered, (data.gshape[1],k), ht.types.canonical_heat_type(data.dtype), None, device, MPI_WORLD)
        out = out.expand_dims(axis=0)

        return out



    def fit(self, data):
        # TODO: document me
        data = data.expand_dims(axis=2)

        # initialize the centroids randomly
        centroids = self.initialize_centroids_databased(self.n_clusters, data.shape[1], data)
        #centroids = self.initialize_centroids(self.n_clusters, data.shape[1], self.random_state, data.device)
        #if data.comm.rank == 0:
        #     np_centroids = centroids._DNDarray__array.numpy().reshape((1024,185, 7 ))
        #     for i in range(7):
        #         img = np_centroids[:, :, i]
        #         file = 'Initial_Centroid_'+str(i)+'.png'
        #         plt.imshow(img)
        #         plt.savefig(file)

        new_centroids = centroids.copy()


        for epoch in range(self.max_iter):
            # calculate the distance matrix and determine the closest centroid

            if data.comm.rank == 0:
                print("Epoch = ", epoch)

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

        # save matching final centroid for each datapoint
        np_matching_centroids = matching_centroids._DNDarray__array.numpy()
        tmp = np_matching_centroids[:,:,0]
        np.savetxt("/home/debu_ch/src/heat/results/Matching_Centroids_Rank"+str(data.comm.rank)+".csv", tmp, delimiter=",")

        return centroids
