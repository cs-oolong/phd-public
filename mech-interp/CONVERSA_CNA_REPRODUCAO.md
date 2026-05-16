# Reprodução CNA - Resumo da Conversa
# Data: 2026-05-14
# Paper: Targeted Neuron Modulation via Contrastive Pair Search (arXiv:2605.12290)

## O que foi feito nesta sessão

### 1. Script de Reprodução CNA Criado
- **Arquivo:** `/home/shimeji/monorepo/phd/mech-interp/reproduce_cna.py`
- Implementa o método CNA (Contrastive Neuron Attribution) do zero
- Usa HuggingFace transformers com hooks manuais (transformer-lens não suporta Qwen2.5)
- Modelo alvo: `Qwen/Qwen2.5-1.5B-Instruct` (ou Llama 3.2 1B)
- Fase 1: Discovery (captura ativações MLP, calcula diferença contrastiva, seleciona top 0.1%)
- Fase 2: Evaluation (mede refusal rate com/sem ablation)

### 2. Script de Monitoramento do Repositório Oficial
- **Arquivo:** `~/.hermes/scripts/monitor_nous_repo.py`
- Checa diariamente se https://github.com/NousResearch/neural-steering está disponível
- **Cronjob agendado:** Todo dia 10h da manhã, avisa no WhatsApp quando sair

### 3. Driver NVIDIA - Instalação em Andamento
- ✅ Driver `nvidia-driver-595-open` instalado
- ✅ Blacklist do nouveau configurado
- ✅ Initramfs atualizado
- 🔄 **PENDENTE:** Reiniciar o computador para ativar o driver

## Especificações do PC
- CPU: Intel i7-13650HX (14 cores / 20 threads)
- RAM: 16 GB
- GPU: NVIDIA RTX 4060 Max-Q / Mobile (8GB VRAM)
- OS: Linux Mint (kernel 6.14.0-37)
- Status GPU: Driver instalado, aguardando reboot

## Capacidade do PC para Mech Interp
| Modelo | Cabe em 8GB? | Uso |
|--------|-------------|-----|
| GPT-2 small (124M) | ✅ Fácil | Prototipagem |
| Qwen2.5 1.5B Instruct | ✅ Confortável | **Nosso alvo agora** |
| Llama 3 8B | ⚠️ 4-bit justo | Com quantização |
| 14B+ | ❌ Não cabe | Precisa de cluster/cloud |

## Próximos Passos (após reboot)
1. Verificar `nvidia-smi` funciona
2. Rodar o script: `python reproduce_cna.py`
3. Verificar resultados (refusal rate baseline vs ablated)
4. Se funcionar: escalar para modelo maior no cluster do lab

## Opções de Cloud para Escalação
| Serviço | Tipo | Preço | Ideal para |
|---------|------|-------|-----------|
| Colab Pro | Notebook gerenciado | R$ 58/mês | Experimentos regulares |
| Vast.ai | Mercado GPU | US$ 0.30-2/hr | Testes pontuais baratos |
| RunPod | Cloud ML | US$ 0.44-2.5/hr | Sessões com persistência |
| Lambda | Servidor dedicado | US$ 800-3000/mês | Projeto longo (PhD) |

## Diferença Base vs Instruct (do paper)
- **Base model:** Não conversa, não recusa. Abla neurônios = só muda conteúdo
- **Instruct model:** Conversa e recusa. Abla neurônios = refusal cai 50%+
- Fine-tuning "transforma" estrutura pré-existente em gate de segurança

## Tempo estimado para reproduzir
- Discovery (1.5B, 16 prompts): ~5-10 minutos na RTX 4060
- Evaluation (10 prompts x 5 strengths): ~15-20 minutos
- **Total:** ~30 minutos para validar o método

## Comandos úteis
```bash
# Verificar GPU
nvidia-smi

# Rodar o script (no venv existente)
cd /home/shimeji/monorepo/phd/mech-interp
/home/shimeji/monorepo/phd/mech-interp/baby_steps/.venv/bin/python reproduce_cna.py

# Ver cronjobs
hermes cron list
```

## Links importantes
- Paper: arXiv:2605.12290
- Repo oficial (ainda não disponível): https://github.com/NousResearch/neural-steering
- Script local: `/home/shimeji/monorepo/phd/mech-interp/reproduce_cna.py`
