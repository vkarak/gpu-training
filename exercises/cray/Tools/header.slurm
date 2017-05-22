#!/bin/bash
#SBATCH --job-name="APPLICATION" 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:03:00 
#SBATCH --output="APPLICATION.log"
#SBATCH --constraint=gpu
