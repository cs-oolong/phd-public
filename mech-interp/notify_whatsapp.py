#!/usr/bin/env python3
"""
Envia notificacoes de progresso do CNA para WhatsApp
Roda como cronjob ou manualmente para verificar status
"""

import os
import glob
import json
from datetime import datetime

def read_latest_status():
    """Le o arquivo de status mais recente"""
    status_file = "/home/shimeji/monorepo/phd/ignore/mech-interp/cna_status.txt"
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            lines = f.readlines()
        return lines[-10:] if lines else []
    return []

def read_latest_results():
    """Le o resultado JSON mais recente"""
    json_files = glob.glob("/home/shimeji/monorepo/phd/ignore/mech-interp/cna_subset_*.json")
    if not json_files:
        return None
    
    latest = max(json_files, key=os.path.getmtime)
    with open(latest, "r") as f:
        return json.load(f)

def format_message():
    """Formata mensagem para WhatsApp"""
    lines = read_latest_status()
    result = read_latest_results()
    
    msg = "📊 *Status CNA*\n\n"
    
    if lines:
        msg += "*Ultimos updates:*\n"
        for line in lines[-5:]:
            msg += f"• {line.strip()}\n"
    
    if result:
        msg += f"\n*Subset atual:* {result['execution_metadata']['subset_size']}\n"
        msg += f"*Tempo total:* {result['timing']['total_seconds']:.0f}s\n"
        msg += f"*Circuito:* {result['model_metadata']['circuit_size']} neuronios\n\n"
        msg += "*Refusal rates:*\n"
        for strength, data in result['results'].items():
            msg += f"  Strength {strength}: {data['rate']:.1f}%\n"
    
    # Verifica se ainda esta rodando
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_cna_subsets.py"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            msg += "\n⏳ *Status:* Rodando..."
        else:
            msg += "\n✅ *Status:* Concluido!"
    except:
        msg += "\n❓ *Status:* Desconhecido"
    
    return msg

if __name__ == "__main__":
    print(format_message())
