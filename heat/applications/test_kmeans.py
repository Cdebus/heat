import heat as ht
import time
import os

class Timing:
	def __init__(self, name, verbosity=1):
		self.verbosity = verbosity
		self.name = name

	def __enter__(self):
		if self.verbosity > 0:
			self.start = time.perf_counter()

	def __exit__(self, *args):
		if  self.verbosity > 0: 
			stop = time.perf_counter()
			print("Time {}: {:4.4f}s".format(self.name, stop-self.start))

def main():
	print("Starting testrun...")	
	data = ht.load_hdf5('/home/debu_ch/src/heat/heat/datasets/data/iris.h5', 'data', split=0)
	
	if data.comm.rank == 0:
		print("size of data: ", data.shape)

	k = 3
	kmeans = ht.ml.cluster.KMeans(n_clusters=k)

	print("Starting Kmeans...")
	#with Timing("(Rank {:2d}) Fitting with Kmeans".format(data.comm.rank)):
	centroids = kmeans.fit(data)
	
	print("Done!")


if __name__=="__main__":
	main()
