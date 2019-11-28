#!/bin/sh

#BSUB -q gpuk80
#BSUB -gpu "num=1"
#BSUB -J DQN_Train_1
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -B
#BSUB -N
#BSUB -u axelnj44@gmail.com
#BSUB -o logs/%J.out
#BSUB -e lobs/%J.err

module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.4.2.24-prod-cuda-10.0

echo "Running Script"
python3 train.py

