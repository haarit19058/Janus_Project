#!/bin/bash
#SBATCH --job-name=job1905
# Partition:
#SBATCH --partition=gpu
# Number of nodes:
#SBATCH --nodes=1
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
# Processors per task:
# Always at least twice the number of GPUs (gpu2 and GTX2080TI in gpu2)
# Four times the number for TITAN and V100 in gpu3 and A5000 in gpu4
# Eight times the number for A40 in gpu3
#SBATCH --cpus-per-task=2
# Wall clock limit:
#SBATCH --time=24:30:00
#SBATCH --error=janus_test_%J.err
#SBATCH --output=janus_test_%J.out
#SBATCH -v

cd /home/AnirbanMondal_grp/23110077/mycode
source /home/AnirbanMondal_grp/23110077/myvenv/bin/activate
python3 /home/AnirbanMondal_grp/23110077/Transformer/train.py
python3 /home/AnirbanMondal_grp/23110077/Transformer/new.py > jan.out

