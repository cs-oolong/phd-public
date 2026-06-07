#!/usr/bin/env python3
"""Download HuggingFace models — APENAS download, zero memória RAM usada."""
import sys
import os
from huggingface_hub import snapshot_download

if len(sys.argv) < 2:
    print("Usage: python download_model.py <model_name> [cache_dir]")
    sys.exit(1)

model_name = sys.argv[1]
# Use filesystem compartilhado do cluster (ajuste para o seu)
cache_dir = sys.argv[2] if len(sys.argv) > 2 else "~/.cache/huggingface/hub/"

print(f"⏳ Baixando arquivos para: {model_name}")
print(f"📁 Destino: {cache_dir}")

snapshot_download(
    repo_id=model_name,
    cache_dir=os.path.expanduser(cache_dir),
    resume_download=True,
)

print(f"✅ Download completo: {model_name}")
