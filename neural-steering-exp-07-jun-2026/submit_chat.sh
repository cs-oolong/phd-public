#!/bin/bash
# Interactive GPU chat launcher — auto-selects an available partition.
#
# Usage:
#   ./submit_chat.sh                                          # Llama int8, auto partition
#   ./submit_chat.sh Qwen/Qwen2.5-7B-Instruct                # Qwen int8, auto partition
#   ./submit_chat.sh meta-llama/Llama-3.1-8B-Instruct int4    # Llama int4, auto partition
#   ./submit_chat.sh meta-llama/Llama-3.1-8B-Instruct int8 l40s  # Force l40s partition

PYTHON="/home/ana.serpa/venvs/neural-steering/bin/python3"
export HF_HOME="/home/ana.serpa/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1

cd /home/ana.serpa/neural-steering

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
QUANT="${2:-int8}"

# Default order: SLURM picks the first partition with free resources.
DEFAULT_PARTITIONS="l40s,rtx5000,a5000,p5000,h100,rtx8000"

if [[ -n "$3" ]]; then
    PARTITIONS="$3"
else
    PARTITIONS="$DEFAULT_PARTITIONS"
fi

echo "=== Cluster GPU Status ==="
sinfo -o "%12P %5a %6D %8T %G" --noheader | grep gpu
echo "==========================="
echo "Requesting GPU on: $PARTITIONS"
echo ""

srun --pty \
    --partition="$PARTITIONS" \
    --time=02:00:00 \
    --mem=32G \
    --cpus-per-task=4 \
    --gres=gpu:1 \
    "$PYTHON" chat.py --model "$MODEL" --quantize "$QUANT"
