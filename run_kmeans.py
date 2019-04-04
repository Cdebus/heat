import heat as ht
import numpy as np
import torch
import os
import random

def main():
    print("Starting TestRun...")


    k = 7

    data = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), 'data', split=0)
    dimensions = data.lshape[1]
    nproc = data.comm.size
    rank = data.comm.rank
    split_axis = data.split


    if split_axis == 0:

        if rank < (k%nproc):
            num_samples = k//nproc + 1
        else:
            num_samples = k//nproc


    else:
        raise NotImplementedError('Not implemented for other splitting-axes')

    local_centroids = ht.empty((num_samples, dimensions))

    for i in range(num_samples):
        x = random.randint(0, data.lshape[0] - 1)
        local_centroids._tensor__array[ i, :] = data._tensor__array[x,:]



    recv_counts = np.full((nproc,), k // nproc)
    recv_counts[:k % nproc] += 1

    recv_displs = np.zeros((nproc,), dtype=recv_counts.dtype)
    np.cumsum(recv_counts[:-1], out=recv_displs[1:])


    gathered = ht.empty((k, data.gshape[1]))
    print("(Rank: {:2d})  ".format(rank),tuple(recv_counts), tuple(recv_displs))

    data.comm.Allgatherv(local_centroids._tensor__array, (gathered._tensor__array, tuple(recv_counts), tuple(recv_displs),), recv_axis=0)

    return_tensor =  ht.transpose(gathered)
    return_tensor = return_tensor.expand_dims(axis=0)


    #kmeans = ht.ml.cluster.KMeans(n_clusters=k)

    #centroids = kmeans.fit(data)


if __name__ == "__main__":
    main()
