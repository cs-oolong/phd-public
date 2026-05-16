#!/usr/bin/env python3
"""
Roda o CNA e envia logs para WhatsApp via Hermes.
"""

import subprocess
import sys
import os
import time

def send_whatsapp_update(message):
    """Envia mensagem para WhatsApp via Hermes CLI"""
    try:
        # Usa o hermes send-message para enviar para o WhatsApp
        subprocess.run(
            ["hermes", "send-message", "--platform", "whatsapp", "--message", message],
            capture_output=True,
            timeout=10
        )
    except Exception as e:
        print(f"Erro ao enviar WhatsApp: {e}")

def main():
    # Envia mensagem inicial
    send_whatsapp_update("🚀 Iniciando reprodução CNA!\nModelo: Qwen2.5-1.5B-Instruct\nGPU: RTX 4060\n\nVou enviar updates a cada etapa...")
    
    # Roda o script CNA capturando output
    process = subprocess.Popen(
        [sys.executable, "reproduce_cna.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd="/home/shimeji/monorepo/phd/mech-interp"
    )
    
    log_lines = []
    last_update = time.time()
    
    for line in process.stdout or []:
        print(line, end='')  # Print local
        log_lines.append(line)
        
        # Envia update a cada 30 segundos ou em marcos importantes
        current_time = time.time()
        if current_time - last_update > 30 or any(keyword in line for keyword in ["Carregando", "FASE", "Circuito", "Refusal", "CONCLUIDA"]):
            # Pega as últimas 5 linhas relevantes
            recent = ''.join(log_lines[-5:])
            if len(recent) > 500:
                recent = recent[:500] + "..."
            send_whatsapp_update(f"📊 CNA Update:\n```\n{recent}\n```")
            last_update = current_time
            log_lines = []  # Limpa para não repetir
    
    # Espera terminar
    return_code = process.wait()
    
    if return_code == 0:
        send_whatsapp_update("✅ CNA CONCLUÍDO COM SUCESSO!\n\nVerificando resultados...")
    else:
        send_whatsapp_update(f"❌ CNA falhou com código {return_code}\n\nVerifique os logs em cna_output.log")
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())
