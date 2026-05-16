#!/bin/bash
# Script para verificar progresso do CNA

echo "=== STATUS DO CNA ==="
echo ""

# Verifica se o processo está rodando
if pgrep -f "reproduce_cna.py" > /dev/null; then
    echo "✅ Processo CNA está RODANDO"
    echo ""
    
    # Mostra últimas linhas do log
    if [ -f cna_output.log ]; then
        echo "📊 Últimas linhas do log:"
        echo "---"
        tail -20 cna_output.log
        echo "---"
    else
        echo "⏳ Log ainda não criado (pode estar baixando o modelo)"
    fi
    
    # Mostra uso da GPU
    echo ""
    echo "🎮 Uso da GPU:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
    
else
    echo "❌ Processo CNA NÃO está rodando"
    
    if [ -f cna_output.log ]; then
        echo ""
        echo "📄 Log completo:"
        cat cna_output.log
    fi
fi

echo ""
echo "=== FIM ==="
