#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 4
#SBATCH -n 224
###SBATCH -p gpu
#SBATCH -t 01:00:00
#SBATCH -J Heat_benchmark
#SBATCH --error=Logs/bench-run-%j.out
#SBATCH --output=Logs/bench-run-%j.out

#SBATCH --error=/home/debu_ch/src/heat/Logs/Similarity-run-%j.out
#SBATCH --output=/home/debu_ch/src/heat/Logs/Similarity-run-%j.out

set -eu; set -o pipefail

source ~/src/activate.sh
export PYTHONPATH=`pwd`

function runbenchmark()
{
    echo ""
    echo "Running Benchmark with following parameters:"
    export OMP_NUM_THREADS=$1
    NUM_SOCKETS_PER_NODE=4 #$(($2/4+1))
    TOTAL_NUM_SOCKETS=$2 #$((${NUM_SOCKETS_PER_NODE}/4))
    echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
    echo "maximal NUM_RANKS_PER_NODE: ${NUM_SOCKETS_PER_NODE}"
    echo "TOTAL_NUM_RANKS: ${TOTAL_NUM_SOCKETS}"
    mpirun -n ${TOTAL_NUM_SOCKETS} -N $NUM_SOCKETS_PER_NODE \
           --mca mpi_cuda_support 0 --mca mpi_warn_on_fork 0 \
           python heat/benchmarks/bench_kmeans.py
}

for i in `seq 14`; do
    for j in `seq 1 16`; do
        runbenchmark $i $j
    done
done

