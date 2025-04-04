#!/bin/bash
#BSUB -q gpuv100
#BSUB -J Geodesics_PartA
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:30
#BSUB -R "rusage[mem=4GB]"
#BSUB -u s203557@dtu.dk
#BSUB -B
#BSUB -env "LSB_JOB_REPORT_MAIL=N"
#BSUB -N
#BSUB -o %J.out
#BSUB -e %J.err

# Load necessary modules
module load python3/3.11.4
module load cuda/11.3

# Activate virtual environment
source /dtu/blackhole/0e/154958/miniconda3/bin/activate adlcv-ex1


python ensemble_vae.py geodesics --device cuda --batch-size 2048