#!/bin/bash
# Roda CNA em tmux com notificacoes WhatsApp

SESSION="cna_subsets"
LOGFILE="/home/shimeji/monorepo/phd/mech-interp/cna_tmux.log"
STATUSFILE="/home/shimeji/monorepo/phd/mech-interp/cna_status.txt"

# Limpa status anterior
> "$STATUSFILE"
echo "$(date '+%H:%M:%S') - Iniciando experimentos CNA..." >> "$STATUSFILE"

# Mata sessao anterior se existir
tmux kill-session -t "$SESSION" 2>/dev/null

# Cria nova sessao
tmux new-session -d -s "$SESSION" \
    "cd /home/shimeji/monorepo/phd/mech-interp && \
     /home/shimeji/monorepo/phd/mech-interp/baby_steps/.venv/bin/python run_cna_subsets.py 2>&1 | tee $LOGFILE"

echo "Sessao tmux criada: $SESSION"
echo "Para acompanhar: tmux attach -t $SESSION"
echo "Para ver logs: tail -f $LOGFILE"
