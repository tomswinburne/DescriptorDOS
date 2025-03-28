#!/bin/bash -f

#SBATCH --cpus-per-task=32

mpirun -np ${SLURM_NTASKS} python run.py
