import heat as ht
import time
import os
import numpy as np
import matplotlib.pyplot as plt


class Timing:
    def __init__(self, name, verbosity=1):
        self.verbosity = verbosity
        self.name = name

    def __enter__(self):
        if self.verbosity > 0:
            self.start = time.perf_counter()

    def __exit__(self, *args):
        if self.verbosity > 0:
            stop = time.perf_counter()
            print("Time {}: {:4.4f}s".format(self.name, stop - self.start))



def main():

    k = 3
    kmeans = ht.ml.cluster.KMeans(n_clusters=k, init="kmeans++")
    test = 'test243'

    #data = ht.load_hdf5(os.path.join(os.getcwd(), "/home/debu_ch/src/heat-Phillip/heat/datasets/data/iris.h5"),"data",split=0)
    #data = ht.load_hdf5(os.path.join(os.getcwd(), '/home/debu_ch/src/heat-Phillip/heat/datasets/data/snapshot_matrix_'+test+'.h5'), 'snapshots', split=0)
    data = ht.load_hdf5(os.path.join(os.getcwd(), '/home/debu_ch/data/snapshot_matrix_'+test+'.h5'),'snapshots', split=0)
    if(data.comm.rank==0):
        print("Data Loaded: ", test)
        start = time.perf_counter()
    kmeans.fit(data)
    if data.comm.rank==0:
        stop = time.perf_counter()
        print("Number of Iterations: {}".format(kmeans.n_iter_))
        print("Kmeans clustering duration: {:4.4f}s".format( stop - start))
    cluster = kmeans.predict(data)
    centroids = kmeans.cluster_centers_

    #
    #np_centroids = centroids.numpy().reshape((k,1024,185 ))
    #np_centroids = centroids.numpy().reshape((3,2,2 ))
    np_cluster = cluster._DNDarray__array.numpy()

    ht.save_hdf5(centroids, '/home/debu_ch/result/results/kmeans/Exp1_1/' + test + '_Centroids.h5', 'centroids')

    # Experiment 1: Writing
    # out final Centroids to images
    if data.comm.rank == 0:
        np.savetxt('/home/debu_ch/result/results/kmeans/Exp1_1/'+test+'_Prediction_2.csv', np_cluster, delimiter=",")

        # for i in range(np_centroids.shape[0]):
        #     img = np_centroids[i, :, :]
        #     file = '/home/debu_ch/result/results/kmeans/'+test+'_Final_Centroid_'+str(i)+'.png'
        #     plt.imshow(img)
        #     plt.savefig(file)
        #     plt.clf()

    # Experiment 2: write out first centroid for each rank
    # np_centroids = centroids._DNDarray__array.numpy().reshape((1024, 185, 7))
    # img = np_centroids[:, :, 0]
    # file = '/home/debu_ch/src/heat/results/Final_Centroid_Rank' + str(data.comm.rank) + '.png'
    # plt.imshow(img)
    # plt.savefig(file)

    # Experiment 3-4:
    # write centroids to csv
    # if data.comm.rank == 0:
    #    np_centroids = centroids._DNDarray__array.numpy().reshape((1024, 185, 7))
    #    for i in range(7):
    #        img = np_centroids[:, :, i]
    #        np.savetxt("/home/debu_ch/src/heat/results/Final_Centroid_" + str(i) + ".csv", img, delimiter=",")


if __name__ == "__main__":
    main()
