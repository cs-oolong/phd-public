# Mechanistic Interpretability Papers with Code

> Curated for your PhD research | Searched: 2026-06-29
> Covers: Circuit tracing, activation patching, SAEs, neural steering, automated circuit discovery

---

## 1. Activation Patching & Circuit Tracing

### Core Papers (Must-read)

| Paper | Authors | Year | Citations | Link |
|-------|---------|------|-----------|------|
| **Towards best practices of activation patching in language models: Metrics and methods** | F Zhang, N Nanda | 2024 | 252 | [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html) |
| **Attribution patching outperforms automated circuit discovery** | A Syed, C Rager, A Conmy | 2024 | 181 | [BlackboxNLP](https://aclanthology.org/2024.blackboxnlp-1.25/) |
| **Circuit-tracer: A new library for finding feature circuits** | M Hanna, M Piotrowski, J Lindsey | 2025 | 14 | [BlackboxNLP](https://aclanthology.org/2025.blackboxnlp-1.14/) |
| **Finding transformer circuits with edge pruning** | A Bhaskar, A Wettig, D Friedman | 2024 | 55 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/20fdaf67581e6d7157376d1ed584040a-Abstract-Conference.html) |
| **Transformer circuit evaluation metrics are not robust** | J Miller, B Chughtai, W Saunders | 2024 | 25 | [OpenReview](https://openreview.net/forum?id=zSf8PJyQb2) |

**Key takeaways for your CNA work:**
- Zhang & Nanda (2024) is the definitive reference on activation patching best practices — directly relevant to your `phase2_activation_patching.py` and CNA reproduction scripts
- Syed et al. (2024) shows attribution patching (EAP) beats ACDC in most settings — suggests EAP might be worth integrating into your `mi_research_package`
- Circuit-tracer (Hanna et al., 2025) is a new library with constrained patching and direct-effects patching — could complement your existing toolkit

---

## 2. Automated Circuit Discovery (ACDC)

| Paper | Authors | Year | Citations | Link |
|-------|---------|------|-----------|------|
| **Towards automated circuit discovery for mechanistic interpretability (ACDC)** | A Conmy, A Mavor-Parker, A Lynch | 2023 | 738 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html) |
| **Efficient automated circuit discovery in transformers using contextual decomposition (CD-T)** | A Hsu, G Zhou, Y Cherapanamjeri | 2025 | 17 | [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/916ee60e315531d6b3954af8a8dc3437-Abstract-Conference.html) |
| **Knowledge circuits in pretrained transformers** | Y Yao, N Zhang, Z Xi, M Wang | 2024 | 98 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d6df31b1be98e04be48af8bedb95b499-Abstract-Conference.html) |
| **Finding transformer circuits with edge pruning (EAP)** | A Bhaskar, A Wettig, D Friedman | 2024 | 55 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/20fdaf67581e6d7157376d1ed584040a-Abstract-Conference.html) |

**Code repositories to check:**
- ACDC: `https://github.com/ArthurConmy/Automatic-Circuit-Discovery`
- EAP: `https://github.com/Abhi1gupta/edge-pruning` 

**Relevance to your work:** Your `reproduce_cna*.py` scripts are doing circuit analysis — ACDC and EAP are the two main frameworks. CD-T (Hsu et al., 2025) claims 97% ROC AUC with better runtime than ACDC.

---

## 3. Sparse Autoencoders (SAEs) for Interpretability

### Highly Cited

| Paper | Authors | Year | Citations | Link |
|-------|---------|------|-----------|------|
| **Sparse autoencoders find highly interpretable features in language models** | H Cunningham, A Ewart, L Riggs, R Huben | 2023 | 1221 | [arXiv](https://arxiv.org/abs/2309.08600) |
| **Scaling and evaluating sparse autoencoders** | L Gao, T Dupre la Tour, H Tillman, G Goh | 2025 | 712 | [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/42ef3308c230942d223c411adf182c88-Abstract-Conference.html) |
| **Towards principled evaluations of sparse autoencoders for interpretability and control** | A Makelov, G Lange, N Nanda | 2025 | 78 | [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/53356aebeea8ffd40a8ac3bb66243162-Abstract-Conference.html) |
| **Saebench: A comprehensive benchmark for sparse autoencoders in language model interpretability** | A Karvonen, C Rager, J Lin, C Tigges, J Bloom | 2025 | 92 | [arXiv](https://arxiv.org/abs/2503.09532) |
| **Interpreting attention layer outputs with sparse autoencoders** | C Kissane, R Krzyzanowski, JI Bloom, A Conmy | 2024 | 62 | [arXiv](https://arxiv.org/abs/2406.17759) |

**Code resources:**
- SAEBench: `https://github.com/jbloomAus/SAEBench`
- OpenAI's SAE implementation: `https://github.com/openai/sparse_autoencoder`
- `sae_experiments.py` in your repo already covers this — consider benchmarking with SAEBench

---

## 4. Neural Steering & Representation Engineering

| Paper | Authors | Year | Citations | Link |
|-------|---------|------|-----------|------|
| **Representation engineering for large-language models: Survey and research challenges** | L Bartoszcze, S Munshi, B Sukidi, J Yen, Z Yang | 2025 | 32 | [arXiv](https://arxiv.org/abs/2502.17601) |
| **Angular steering: Behavior control via rotation in activation space** | MH Vu, T Nguyen | 2026 | 29 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b0223cad0e73b793f31eb6cc41cefceb-Abstract-Conference.html) |
| **Steering large language models using conceptors** | J Postmus, S Abreu | 2024 | 30 | [arXiv](https://arxiv.org/abs/2410.16314) |
| **Steering knowledge selection behaviours in LLMs via sae-based representation engineering** | Y Zhao, A Devoto, G Hong, X Du | 2025 | 17 | [NAACL](https://aclanthology.org/2025.naacl-long.264/) |
| **Learning to steer: Input-dependent steering for multimodal llms** | J Parekh, P Khayatan, M Shukor | 2026 | 9 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ea491e2d1c46686b8db5cd11154f5d2c-Abstract-Conference.html) |
| **Steer2adapt: Dynamically composing steering vectors elicits efficient adaptation of llms** | P Han, X Xu, K Xuan, P Song | 2026 | 6 | [arXiv](https://arxiv.org/abs/2602.07276) |

**Direct relevance:** Your `neural-steering-exp-07-jun-2026/` folder with refusal comparison, moral comparison, entity comparison, and language comparison scripts maps directly to this literature.

**Key code resources:**
- Representation Engineering repo: `https://github.com/victorhad/RepE`
- Conceptor steering: `https://github.com/JJPostmus/conceptor-steering`

---

## 5. Benchmarks & Evaluation

| Paper | Authors | Year | Citations | Link |
|-------|---------|------|-----------|------|
| **Mib: A mechanistic interpretability benchmark** | A Mueller, A Geiger, S Wiegreffe, D Arad | 2025 | 49 | [arXiv](https://arxiv.org/abs/2504.13151) |
| **Interpbench: Semi-synthetic transformers for evaluating mechanistic interpretability techniques** | R Gupta, I Arcuschin, T Kwa | 2024 | 19 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a8f7d43ae092d9a5295775eb17f3f4f7-Abstract-Datasets_and_Benchmarks_Track.html) |
| **Find: A function description benchmark for evaluating interpretability methods** | S Schwettmann, T Shaham | 2023 | 37 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ef0164c1112f56246224af540857348f-Abstract-Datasets_and_Benchmarks.html) |
| **Tracr: Compiled transformers as a laboratory for interpretability** | D Lindner, J Kramár, S Farquhar | 2023 | 133 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/771155abaae744e08576f1f3b4b7ac0d-Abstract-Conference.html) |

---

## 6. Tools & Frameworks with Code

| Tool/Paper | What it does | Link |
|------------|-------------|------|
| **Prisma** (Joseph et al., 2025) | Open-source toolkit for vision MI | [arXiv](https://arxiv.org/abs/2504.19475) |
| **MechIR** (Parry et al., 2025) | MI framework for Information Retrieval | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-88720-8_16) |
| **nnterp** (Dumas, 2025) | Standardized interface for MI across architectures | [arXiv](https://arxiv.org/abs/2511.14465) |
| **SAEBench** (Karvonen et al., 2025) | Comprehensive SAE benchmarking suite | [arXiv](https://arxiv.org/abs/2503.09532) |
| **Circuit-tracer** (Hanna et al., 2025) | Library for finding feature circuits | [BlackboxNLP](https://aclanthology.org/2025.blackboxnlp-1.14/) |

---

## 7. Recommended Next Steps

Based on your current work in `mech-interp/` and `neural-steering-exp-07-jun-2026/`:

1. **For CNA reproduction**: Compare your `reproduce_cna.py` against the EAP implementation (Bhaskar et al., 2024) — edge pruning is faster and often more accurate than ACDC
2. **For SAE experiments**: Benchmark your `sae_experiments.py` against SAEBench (Karvonen et al., 2025) for standardized evaluation
3. **For neural steering**: The angular steering paper (Vu & Nguyen, 2026) introduces rotation-based steering which could complement your current vector-addition approach in `refusal_comparison.py` and related scripts
4. **Cross-model generalization**: "Universal sparse autoencoders" (Thasarathan et al., 2025) explores cross-model concept alignment — highly relevant to your multi-model experiments (Llama, Mistral, Qwen, Hunyuan)

---

*Generated from 71 unique papers across 5 targeted searches on Google Scholar*
