#!/bin/bash
#SBATCH --job-name=AIGVDetEval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=A100,L40S,A40
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclude=node50

# -------- shell hygiene --------
# Exit immediately if a command exits with a non-zero status.
set -euo pipefail
# Enable debugging output
#set -x
umask 077
mkdir -p logs

# -------- print job header --------
echo "================= SLURM JOB START ================="
echo "Job:    $SLURM_JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Node:   ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-unknown}  (${SLURM_JOB_GPUS:-not-set})"
echo "Start:  $(date)"
echo "==================================================="

datetime="$(date '+%Y%m%d_%H%M%S')"
result_dir="results/${datetime}_AIGVDet"
mkdir -p "${result_dir}"
data_root="/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_videos_only"
#data_root="/home/infres/ziyliu-24/data/FakeParts2DataMock"
data_entry_csv="/projects/hi-paris/DeepFakeDataset/videos_index.csv"
done_csv=("results/")

model_original="checkpoints/optical.pth"
model_optical_flow="checkpoints/original.pth"
raft_model="raft_model/raft-things.pth"

source /home/infres/ziyliu-24/miniconda3/etc/profile.d/conda.sh
conda activate antifake310
export PYTHONPATH="${PYTHONPATH:-}:${PWD}/core"

srun python3 -Wignore AIGVDetEval.py \
         --data_root "${data_root}" \
         --data_csv "${data_entry_csv}" \
         --done_csv_list "${done_csv[@]}" \
         --pred_csv "${result_dir}/predictions.csv" \
         --model_original_path "${model_original}" \
         --model_optical_flow_path "${model_optical_flow}" \
         --raft_model "${raft_model}"

EXIT_CODE=$?

echo "================== SLURM JOB END =================="
echo "End:   $(date)"
echo "Exit:  ${EXIT_CODE}"
echo "==================================================="
exit "${EXIT_CODE}"