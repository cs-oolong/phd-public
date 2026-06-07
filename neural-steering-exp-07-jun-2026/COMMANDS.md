# Neural Steering — Command Reference

## Setup

```bash
git clone https://github.com/NousResearch/neural-steering.git
cd neural-steering
python -m venv venv
source venv/bin/activate
pip install torch transformers accelerate bitsandbytes huggingface_hub sentencepiece protobuf
pip install -e .
```

## Downloading Models

```bash
# Run from headnode (needs internet, compute nodes are offline)
/home/ana.serpa/venvs/neural-steering/bin/python3 download_only.py <model_name> /home/ana.serpa/.cache/huggingface/hub/

# Models used in experiments
/home/ana.serpa/venvs/neural-steering/bin/python3 download_only.py meta-llama/Llama-3.1-8B-Instruct /home/ana.serpa/.cache/huggingface/hub/
/home/ana.serpa/venvs/neural-steering/bin/python3 download_only.py Qwen/Qwen2.5-7B-Instruct /home/ana.serpa/.cache/huggingface/hub/
/home/ana.serpa/venvs/neural-steering/bin/python3 download_only.py mistralai/Mistral-7B-Instruct-v0.3 /home/ana.serpa/.cache/huggingface/hub/
/home/ana.serpa/venvs/neural-steering/bin/python3 download_only.py tencent/Hunyuan-7B-Instruct /home/ana.serpa/.cache/huggingface/hub/
```

## Compatibility Check

```bash
# Config only (no GPU, no download)
python check_compatibility.py --model <model_name> --level config

# Full check on cluster
sbatch submit_check_compatibility.sh <model_name>
```

## Interactive Chat

```bash
# Auto partition, int8 quantization (default)
./submit_chat.sh

# Specific model and quantization
./submit_chat.sh Qwen/Qwen2.5-7B-Instruct int4

# No quantization
./submit_chat.sh meta-llama/Llama-3.1-8B-Instruct none

# Force partition
./submit_chat.sh meta-llama/Llama-3.1-8B-Instruct int8 l40s
```

## Quickstart (Refusal Ablation Demo)

```bash
sbatch submit_quickstart.sh
```

## Experiments — Individual

```bash
# Each accepts an optional model argument (default: Llama-3.1-8B-Instruct)
sbatch submit_refusal_comparison.sh [model]
sbatch submit_language_comparison.sh [model]
sbatch submit_moral_comparison.sh [model]
sbatch submit_entity_comparison.sh [model]
sbatch submit_knowledge_baseline.sh [model]
```

## Experiments — Batch (All 5 per Model)

```bash
./submit_all_llama.sh
./submit_all_qwen.sh
./submit_all_mistral.sh
./submit_all_hunyuan.sh
```

## Cluster Monitoring

```bash
# GPU availability planner
./job_planner.sh
./job_planner.sh --gpus 2 --mem 64G --cpus 8

# Cluster status
sinfo -o "%12P %5a %6D %8T %G" --noheader | grep gpu

# My running/pending jobs
squeue -u $USER

# Recent job history (last 2 hours)
sacct -u ana.serpa --starttime now-2hours --format=JobID,JobName,Partition,State,Elapsed,ExitCode,NodeList -X

# Recent job history (last 24 hours)
sacct -u ana.serpa --starttime now-1day --format=JobID,JobName,State,Elapsed,ExitCode -X

# Follow job output
tail -f /home/ana.serpa/slurm/<jobid>.out

# Check drained nodes
sinfo -N -l --states=drain,drained
sinfo -N -o "%12N %12P %6T %20E"

# Check what a specific job is using
scontrol show job <jobid> | grep -E "NumCPUs|MinMemory|Gres|Partition|NodeList"
```

## Copying Results to Local Machine

```bash
# From your local machine (zsh — needs quotes around globs)
rsync -avP "ana.serpa@headnode:~/neural-steering/{submit_*.sh,job_planner.sh,chat.py,quickstart.py,*_comparison.py,knowledge_baseline.py,check_compatibility.py,download_only.py}" ./

# Copy all results
rsync -avP ana.serpa@headnode:~/neural-steering/results/ ./results/
```
