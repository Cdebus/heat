#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 3
#SBATCH -n 150
#SBATCH -t 16:00:00
#SBATCH -J Heat_Affinity
#SBATCH --mem=350000
#SBATCH --error=/home/debu_ch/src/heat/Logs/Similarity-%j.out
#SBATCH --output=/home/debu_ch/src/heat/Logs/Similarity-%j.out

set -eu; set -o pipefail

export PYTHONPATH=`pwd`

function runkmeans()
{
	#echo "Running kmeans clustering (Databased Centroid Initialization : Test 4C )"
	echo "Running Spectral Clustering"
	echo $(date)
	export OMP_NUM_TASK=$SLURM_NTASKS
	echo "Number of nodes: ${OMP_NUM_TASK}"

	mpirun -n ${OMP_NUM_TASK} --mca mpi_cuda_support 0 --mca mpi_warn_on_fork 0 python /home/debu_ch/src/heat/testing_LA.py
}

runkmeans
