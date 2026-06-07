#!/bin/bash
# Usage:
#   sbatch submit_check_compatibility.sh meta-llama/Llama-3.1-8B-Instruct
#   sbatch submit_check_compatibility.sh google/gemma-2-9b-it
#   sbatch submit_check_compatibility.sh openai/gpt-oss-20b
#
# Check output:
#   tail -f /home/ana.serpa/slurm/<jobid>.out

#SBATCH --job-name=check-compat
#SBATCH --partition=l40s,h100,rtx8000,a5000,rtx5000
#SBATCH --output=/home/ana.serpa/slurm/%j.out
#SBATCH --error=/home/ana.serpa/slurm/%j.err
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-user=a165880@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

PYTHON="/home/ana.serpa/venvs/neural-steering/bin/python3"
export HF_HOME="/home/ana.serpa/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1

MODEL="${1:-}"
if [[ -z "${MODEL}" ]]; then
    echo "ERROR: model name required."
    echo "Usage: sbatch submit_check_compatibility.sh <huggingface-model>"
    exit 1
fi

cd /home/ana.serpa/neural-steering

echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Node:      ${SLURM_NODELIST}"
echo "Model:     ${MODEL}"
echo ""

echo "=== Cluster GPU Status ==="
sinfo -o "%12P %5a %6D %8T %G" --noheader | grep gpu
echo "==========================="
echo ""

echo "=== Allocated GPU ==="
nvidia-smi
echo "====================="
echo ""

echo "=== Running check_compatibility.py ==="
$PYTHON check_compatibility.py --model "${MODEL}" --level all --no-color
