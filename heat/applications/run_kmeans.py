import heat as ht
import os

def main():

    #data = ht.load_hdf5(os.path.join(os.getcwd(), '/home/debu_ch/src/heat-Phillip/heat/datasets/data/iris.h5'), 'data', split=0)
    data = ht.load_hdf5(os.path.join(os.getcwd(), '/home/debu_ch/src/heat-Phillip/heat/datasets/data/snapshot_matrix_test284.h5'), 'snapshots', split=0)
    k = 7
    kmeans = ht.ml.cluster.KMeans(n_clusters=k)
    centroids = kmeans.fit(data)
    print("FINAL CENTROIDS CALCULATED ")




if __name__ == "__main__":
    main()
