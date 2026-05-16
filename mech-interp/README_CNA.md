# Guia de Uso - CNA Reproduction

**Commit atual:** `e6a16c8`

---

## 📁 Arquivos do Projeto

| Arquivo | Descrição |
|---------|-----------|
| `reproduce_cna.py` | Script original (versão completa) |
| `reproduce_cna_v2.py` | Versão simplificada com metadados |
| `reproduce_cna_jbb.py` | Versão com JBB-Behaviors dataset |
| `run_cna_subsets.py` | **Principal** - subsets progressivos com timing |
| `run_cna_tmux.sh` | Roda em tmux (não morre se desconectar) |
| `TUTORIAL_CNA.md` | Tutorial completo para compartilhar |
| `README_CNA.md` | Este guia de uso rápido |

---

## 🚀 Como Usar

### Opção 1: Rápido (um comando)

```bash
cd /home/shimeji/monorepo/phd/mech-interp
bash run_cna_tmux.sh
```

Isso roda em tmux e salva logs em `cna_tmux.log`.

### Opção 2: Com notificações WhatsApp

```bash
# 1. Iniciar experimento
cd /home/shimeji/monorepo/phd/mech-interp
bash run_cna_tmux.sh

# 2. Criar cronjob de notificações (a cada 5 min)
hermes cronjob create --name "CNA Status" --schedule "*/5 * * * *" --script notify_cna_status.py --no-agent

# 3. Quando acabar, remover cronjob
hermes cronjob list
hermes cronjob remove --job-id <ID>
```

### Opção 3: Manual (sem tmux)

```bash
cd /home/shimeji/monorepo/phd/mech-interp
/home/shimeji/monorepo/phd/mech-interp/baby_steps/.venv/bin/python run_cna_subsets.py
```

---

## 📊 O que o script faz

1. **Carrega JBB-Behaviors** (100 prompts harmful)
2. **Roda subsets progressivos:** 10 → 20 → 50 → 100 prompts
3. **Para cada subset:**
   - Discovery: encontra 224 neurônios (0.1%)
   - Evaluation: testa 3 strengths (0.0, 0.5, 1.0)
   - Mede tempo de cada fase
   - Salva resultados em JSON
4. **Gera resumo comparativo**

---

## 📁 Arquivos de Saída

| Padrão | Conteúdo |
|--------|----------|
| `cna_subset_10_*.json` | Resultado subset 10 prompts |
| `cna_subset_20_*.json` | Resultado subset 20 prompts |
| `cna_subset_50_*.json` | Resultado subset 50 prompts |
| `cna_subset_100_*.json` | Resultado subset 100 prompts |
| `cna_summary_*.json` | Resumo comparativo de todos |
| `cna_tmux.log` | Logs em tempo real |
| `cna_status.txt` | Status para notificações |

---

## 🔍 Como verificar resultados

### Ver último resultado:
```bash
ls -lt cna_subset_*.json | head -1
```

### Ver resumo de todos:
```bash
for f in cna_subset_*.json; do
  echo "=== $(basename $f) ==="
  python3 -c "import json; d=json.load(open('$f')); print(f'Subset: {d[\"execution_metadata\"][\"subset_size\"]}'); print(f'Tempo: {d[\"timing\"][\"total_seconds\"]:.0f}s'); print(f'Refusal S0.0: {d[\"results\"][\"0.0\"][\"rate\"]:.1f}%'); print(f'Refusal S0.5: {d[\"results\"][\"0.5\"][\"rate\"]:.1f}%'); print(f'Refusal S1.0: {d[\"results\"][\"1.0\"][\"rate\"]:.1f}%')"
done
```

### Ver status em tempo real:
```bash
tail -f cna_tmux.log
```

---

## ⚙️ Configurar subsets

Editar `run_cna_subsets.py`, linha 24:
```python
SUBSET_SIZES = [10, 20, 50, 100]  # Alterar aqui
```

---

## 🐛 Troubleshooting

### Erro: "No module named jailbreakbench"
```bash
/home/shimeji/monorepo/phd/mech-interp/baby_steps/.venv/bin/pip install jailbreakbench
```

### Erro: "CUDA out of memory"
- Reduzir `max_length` no script (padrão: 512)
- Ou usar modelo menor

### Processo morreu
```bash
tmux attach -t cna_subsets  # Ver o que aconteceu
```

---

## 📈 Resultados esperados

| Subset | Tempo | S 0.0 | S 0.5 | S 1.0 |
|--------|-------|-------|-------|-------|
| 10 | ~30s | ~20% | ~40% | ~50% |
| 20 | ~60s | ~5% | ~40% | ~50% |
| 50 | ~160s | ~6% | ~54% | ~62% |
| 100 | ~320s | ~3% | ~37% | ~57% |

**Interpretação:** S 0.0 = ablado (menos refusal), S 1.0 = baseline

---

## 🔗 Links úteis

- Paper: arXiv:2605.12290
- JBB-Behaviors: https://github.com/JailbreakBench/jailbreakbench
- Repositório Nous (ainda não público): https://github.com/NousResearch/neural-steering
