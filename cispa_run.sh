#!/bin/bash

#SBATCH --job-name=str-master
#SBATCH --output=/home/c01adga/CISPA-projects/sparse_lottery-2022/STR-master/jobs/job-%j.out

#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu
#SBATCH --time=8640


JOBDATADIR=`ws create work --space "$SLURM_JOB_ID" --duration "7 00:00:00"`
JOBTMPDIR=/tmp/job-"$SLURM_JOB_ID"

# test for the credentials files
srun test -f ~/CISPA-home/.config/enroot/.credentials

srun mkdir "$JOBTMPDIR"

srun mkdir -p "$JOBDATADIR" "$JOBTMPDIR"/models

srun --container-image=projects.cispa.saarland:5005#css/ngc/pytorch:21.12-py3 --container-mounts="$JOBTMPDIR":/tmp python3 $HOME/CISPA-projects/sparse_lottery-2022/STR-master/run.py \
--data-dir $HOME/CISPA-scratch/CIFAR10 --model-dir /tmp/models

srun mv /tmp/job-"$SLURM_JOB_ID".out "$JOBDATADIR"/out_resnet_init.txt
srun mv "$JOBTMPDIR"/ "$JOBDATADIR"/data
