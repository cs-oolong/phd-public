#!/bin/bash
# Submit all experiments for meta-llama/Llama-3.1-8B-Instruct
# Usage: ./submit_all_llama.sh [model_override]

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting all experiments for: $MODEL"
echo ""

echo "=== Cluster GPU Status ==="
sinfo -o "%12P %5a %6D %8T %G" --noheader | grep gpu
echo "==========================="
echo ""

declare -A JOBS

submit_job() {
    local name="$1"
    local script="$2"
    local output
    output=$(sbatch "${SCRIPT_DIR}/${script}" "$MODEL")
    local job_id
    job_id=$(echo "$output" | grep -oP '(?<=Submitted batch job )\d+')
    JOBS["$name"]="$job_id"
    echo "  $name: job $job_id"
}

echo "Submitting jobs..."
submit_job "refusal_comparison"    "submit_refusal_comparison.sh"
submit_job "language_comparison"   "submit_language_comparison.sh"
submit_job "moral_comparison"      "submit_moral_comparison.sh"
submit_job "entity_comparison"     "submit_entity_comparison.sh"
submit_job "knowledge_baseline"    "submit_knowledge_baseline.sh"
echo ""

echo "=== Summary ==="
echo "Model: $MODEL"
for name in refusal_comparison language_comparison moral_comparison entity_comparison knowledge_baseline; do
    echo "  $name: ${JOBS[$name]}"
done
echo ""

echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f /home/ana.serpa/slurm/<jobid>.out"
