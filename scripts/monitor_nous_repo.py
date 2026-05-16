#!/usr/bin/env python3
"""
Monitora se o repositório da Nous Research foi liberado.
Checa https://github.com/NousResearch/neural-steering diariamente.
"""

import urllib.request
import urllib.error
import sys

def check_repo():
    url = "https://github.com/NousResearch/neural-steering"
    try:
        req = urllib.request.Request(url, method='HEAD')
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200:
                print(f"🎉 REPOSITÓRIO DISPONÍVEL! {url}")
                print("O código do paper 'Targeted Neuron Modulation via Contrastive Pair Search' foi liberado!")
                return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"⏳ Repositório ainda não disponível (404). URL: {url}")
            print("   O paper prometeu open-source mas o repo ainda não está público.")
        else:
            print(f"⚠️ Erro HTTP {e.code} ao checar o repositório.")
    except Exception as e:
        print(f"⚠️ Erro ao checar: {e}")
    return False

if __name__ == "__main__":
    available = check_repo()
    sys.exit(0 if available else 1)
