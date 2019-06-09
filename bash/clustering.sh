#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1 
#SBATCH -n 20
#SBATCH -t 01:00:00
#SBATCH -J Heat_kMeans
#SBATCH --mem=350000
#SBATCH --error=Logs/kmeans-run-%j.out
#SBATCH --output=Logs/kmeans-run-%j.out

set -eu; set -o pipefail

export PYTHONPATH=`pwd`

function runkmeans()
{
	echo "Running kmeans clustering (Databased Centroid Initialization)"
	export OMP_NUM_TASK=$SLURM_NTASKS
	echo "Number of nodes: ${OMP_NUM_TASK}"

	mpirun -n ${OMP_NUM_TASK} --mca mpi_cuda_support 0 --mca mpi_warn_on_fork 0 python heat/clustering/run_kmeans.py
}

runkmeans
