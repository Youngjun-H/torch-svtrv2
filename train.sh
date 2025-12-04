#!/bin/bash
#SBATCH --job-name=LPR
#SBATCH --partition=hopper
#SBATCH --nodes=1                    # 노드 수 (필요시 수정)
#SBATCH --gres=gpu:8                 # 노드당 GPU 수 (필요시 수정)
#SBATCH --ntasks-per-node=8          # 노드당 태스크 수 (보통 GPU 수와 동일)
#SBATCH --cpus-per-task=14
#SBATCH --mem=2000G
#SBATCH --comment="LPR_training"
#SBATCH --output=model_%A.log

# =============================================================================
# 학습 설정 정보 출력
# =============================================================================
echo "================================================================"
echo "Job name: $SLURM_JOB_NAME"
echo "Nodelist: $SLURM_JOB_NODELIST"
echo "Number of nodes: ${SLURM_NNODES:-1}"
echo "GPUs per node: ${SLURM_NTASKS_PER_NODE:-8}"
echo "Total GPUs: $((${SLURM_NNODES:-1} * ${SLURM_NTASKS_PER_NODE:-8}))"
echo "================================================================"

echo "Run started at:- "
date
hostname -I;

# =============================================================================
# 학습 실행
# PyTorch Lightning이 SLURM 환경 변수를 자동으로 감지하여 분산 학습 설정
# =============================================================================

# SLURM에서 실제 할당된 노드 수와 GPU 수 확인
# SLURM 환경 변수를 직접 사용 (가장 정확함)
ACTUAL_NODES=${SLURM_NNODES:-1}
ACTUAL_GPUS_PER_NODE=${SLURM_NTASKS_PER_NODE:-8}

echo "Starting training with:"
echo "  - Nodes: $ACTUAL_NODES"
echo "  - GPUs per node: $ACTUAL_GPUS_PER_NODE"
echo "  - Total GPUs: $((ACTUAL_NODES * ACTUAL_GPUS_PER_NODE))"
echo ""

srun python svtrv2/train.py \
    --config svtrv2/configs/svtrv2_rctc.yml \
    --num_nodes ${ACTUAL_NODES} \
    --devices ${ACTUAL_GPUS_PER_NODE} \