#!/bin/bash
#SBATCH --job-name=example1
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --account=eecs587f23_class
#SBATCH --partition=gpu




#make sure to load the cuda module before running
#module load cuda
#make sure to compile your program using nvcc
#./a.out $2
nvcc -o colortree treecoloring.cu
nvprof --log-file output_$1.log ./colortree $1 >> output_$1.txt
rm random*
