import heat as ht
import time
import os

#ht.use_device("gpu")

class Timing:
    def __init__(self,name,verbosity=1):
        self.verbosity = verbosity
        self.name = name

    def __enter__(self):
        if self.verbosity > 0:
            self.start = time.perf_counter()

    def __exit__(self,*args):
        if self.verbosity > 0:
            stop = time.perf_counter()
            print("Time {}: {:4.4f}s".format(self.name,stop-self.start))

def main():
    #print("Running benchmark:")
    batch_size = 100000
    data = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/twitter.h5'), 'DBSCAN', split=0)
    #data = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/snapshot_matrix_test284.h5'), 'snapshots', split=0)
    #if data.comm.rank == 0:
    print("Size of data: ", data.shape)
    # fit the clusters
    k = 10
    kmeans = ht.ml.cluster.KMeans(n_clusters=k)

    with Timing("(Rank {:2d}) Fitting with KMeans".format(data.comm.rank)):
        centroids = kmeans.fit(data)

if __name__=="__main__":
    main()
