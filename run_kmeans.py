import heat as ht
import os

def main():
        print("Starting TestRun...")

        data = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), 'data', split=0)

        k = 3
        kmeans = ht.ml.cluster.KMeans(n_clusters=k)

        centroids = kmeans.fit(data)


if __name__ == "__main__":
    main()
