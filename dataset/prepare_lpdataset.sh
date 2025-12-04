#!/bin/bash
#SBATCH --job-name=LPR
##SBATCH --partition=hopper
#SBATCH --nodelist=hpe160
#SBATCH --nodes=1                    # 노드 수 (필요시 수정)
#SBATCH --gres=gpu:0                 # 노드당 GPU 수 (필요시 수정)
#SBATCH --ntasks-per-node=8          # 노드당 태스크 수 (보통 GPU 수와 동일)
#SBATCH --cpus-per-task=14
#SBATCH --mem=2000G
#SBATCH --comment="LPR model training"
#SBATCH --output=dataset_%A.log

echo "Run started at:- "
date
hostname -I;

srun python prepare_lpdataset.py