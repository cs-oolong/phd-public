# Mechanistic Interpretability for LLMs: A Comprehensive Research Guide

*Compiled April 2026 — focused on 2024-2026 literature and tooling*

---

## Table of Contents

1. [What Is Mechanistic Interpretability?](#1-what-is-mechanistic-interpretability)
2. [Key Techniques](#2-key-techniques)
   - 2.1 Sparse Autoencoders (SAEs)
   - 2.2 Logit Lens & Tuned Lens
   - 2.3 Activation Patching / Causal Tracing
   - 2.4 Circuit Discovery & Attribution Graphs
   - 2.5 Probing Classifiers
   - 2.6 Representation Engineering / Activation Steering
   - 2.7 Patchscopes
3. [Comparison Table: Pros & Cons](#3-comparison-table-pros--cons)
4. [Tooling Ecosystem](#4-tooling-ecosystem)
5. [Key Papers (2024-2026)](#5-key-papers-2024-2026)
6. [Hardware Requirements & Practical Advice](#6-hardware-requirements--practical-advice)
7. [Demo Script Overview](#7-demo-script-overview)

---

## 1. What Is Mechanistic Interpretability?

Mechanistic Interpretability (MI) is the subfield of AI safety/interpretability that seeks to **reverse-engineer the internal computations** of neural networks — particularly transformer-based LLMs — into human-understandable algorithms. Unlike post-hoc explainability (e.g., SHAP, saliency maps) which correlates inputs to outputs, MI aims to uncover **why and how** the model computes its answer.

Named one of **MIT Technology Review's 10 Breakthrough Technologies for 2026**, MI has evolved from studying toy models into scalable methods applied to frontier models like Claude 3.5 and GPT-4.

**The mental model:** Think of a transformer as a huge program compiled by training, not written by humans. MI tries to:
1. Find the internal variables that matter (**features**, not necessarily individual neurons)
2. Trace how those variables influence each other (**circuits**, paths, attribution graphs)
3. Test causality by **intervening** (patching, ablations, steering) and checking what changes
4. Compress the story into a faithful, human-usable explanation

---

## 2. Key Techniques

### 2.1 Sparse Autoencoders (SAEs)

**The Big Idea:** Individual neurons in LLMs are *polysemantic* — they activate for multiple unrelated concepts. This happens because models use **superposition** to represent more features than they have neurons. SAEs decompose these dense, polysemantic activations into **sparse, monosemantic features** (each feature = one concept).

**How it works:**
- An SAE is a shallow autoencoder (encoder + decoder) with a sparsity constraint (L1 penalty, TopK, JumpReLU, etc.)
- It is trained on millions of activation vectors extracted from a specific layer/component of the LLM
- The encoder maps a d-dimensional activation to a much larger (e.g., 16K-16M) sparse feature space
- Each dimension of this larger space ideally represents one interpretable concept

**Why it's hardware-friendly:**
- You do NOT need to retrain or fine-tune the LLM itself
- Pre-trained SAEs are freely available for models like GPT-2, Gemma 2, Pythia, Llama
- You just load the LLM, run it on some text, extract activations, and pass them through the SAE
- Works on CPU for small models (GPT-2 Small = ~124M params, runs on a laptop)

**Key variants (2024-2025):**
- **TopK SAEs** (OpenAI, Gao et al., ICLR 2025): Directly control sparsity by keeping only the top-k features active. Cleaner scaling laws.
- **JumpReLU SAEs** (DeepMind/Gemma Scope): Use a learnable threshold for activation.
- **Gated SAEs**: Separate gating and magnitude estimation for better reconstruction.
- **Cross-Layer Transcoders (CLTs)** (Anthropic, 2025): Instead of per-layer SAEs, these predict MLP outputs across layers, enabling richer circuit tracing.

**Pros:**
- Most scalable unsupervised feature discovery method
- Works on any model with accessible activations
- Pre-trained SAEs available; no need to train from scratch for common models
- Features are often genuinely monosemantic and interpretable

**Cons:**
- Training SAEs from scratch is compute-intensive (millions of activation vectors)
- Not all features are easily interpretable — some remain opaque
- Reconstruction is imperfect; SAEs miss some model behavior
- Feature splitting/absorption at different scales can complicate analysis

---

### 2.2 Logit Lens & Tuned Lens

**The Big Idea:** Apply the model's output (unembedding) matrix to intermediate layer hidden states to see what the model would "predict" if processing stopped at each layer.

**How it works:**
- At each layer l, compute: `LogitLens(h_l) = LayerNorm(h_l) @ W_unembed`
- This gives a vocabulary distribution at every layer, revealing how predictions refine layer by layer
- The **Tuned Lens** (Belrose et al., 2023) trains a small affine probe per layer to correct for basis changes, producing more reliable results in early layers

**Pros:**
- Extremely simple to implement (a few lines of code)
- No training required (logit lens); minimal training for tuned lens
- Great for building intuition about information flow
- Works on any autoregressive transformer

**Cons:**
- Purely observational — shows what information *exists*, not what the model *uses*
- Logit lens is unreliable in early layers (basis mismatch)
- Cannot establish causal claims
- Limited to vocabulary-space projections

---

### 2.3 Activation Patching / Causal Tracing

**The Big Idea:** Replace a specific activation in one model run with the corresponding activation from a different run, and measure the effect on output. This is the primary tool for establishing **causal** claims about model internals.

**How it works:**
1. Run the model on a **clean** prompt (correct behavior)
2. Run the model on a **corrupted** prompt (different behavior)
3. At a specific component (layer, attention head, MLP, etc.), replace the corrupted activation with the clean one
4. If output changes back toward correct behavior, that component is **causally important**

**Variants:**
- **Activation Patching** (Heimersheim & Nanda, 2024): Replace activations between clean/corrupted runs
- **Attribution Patching** (Nanda, 2023; Syed et al., BlackboxNLP 2024): Uses gradients for a linear approximation — much faster (2 forward + 1 backward pass vs. N forward passes)
- **Path Patching**: Tests the causal effect of direct paths from specific senders to receivers in isolation

**Pros:**
- Gold standard for causal claims about model components
- Can precisely identify which components are responsible for specific behaviors
- Works at multiple granularities (layers, heads, MLPs, individual neurons)

**Cons:**
- Computationally expensive (one forward pass per activation patched — attribution patching mitigates this)
- Requires careful experimental design (choosing clean/corrupted pairs)
- Results can be sensitive to choice of corruption method and metric
- Combinatorial explosion when trying to find full circuits

---

### 2.4 Circuit Discovery & Attribution Graphs

**The Big Idea:** Identify the minimal subnetwork (circuit) within the model responsible for a specific behavior, and visualize it as a graph of interacting features.

**Landmark work — Anthropic's Circuit Tracing (March 2025):**
- Uses Cross-Layer Transcoders (CLTs) to replace MLPs with interpretable features
- Builds **attribution graphs** showing how features interact to produce outputs
- Applied to Claude 3.5 Haiku — first time circuits were traced in a production-scale model
- Open-sourced as the `circuit-tracer` library (2.7K stars on GitHub)
- Demonstrated genuine multi-step reasoning (e.g., "capital of the state containing Dallas" → Texas → Austin)

**Other approaches:**
- **ACDC Algorithm** (Conmy et al.): Greedy iterative circuit discovery
- **Edge Attribution Patching** (Syed et al., 2024): Prune edges by importance scores — outperforms ACDC
- **Automated Circuit Discovery** via gradient-based methods

**Pros:**
- Most detailed view of model computation
- Can reveal genuine multi-step reasoning, planning, and knowledge retrieval
- Attribution graphs provide visual, explorable representations
- Open-source tooling available

**Cons:**
- Currently takes hours of human effort to understand circuits even for short prompts
- Missing attention circuits in some approaches (e.g., CLT-based tracing)
- Reconstruction errors can introduce artifacts
- Scaling to longer prompts and larger models remains challenging

---

### 2.5 Probing Classifiers

**The Big Idea:** Train a small classifier (usually linear) on top of frozen model activations to test whether specific information (e.g., part-of-speech, entity type, sentiment) is encoded at a given layer.

**How it works:**
- Extract activations from a specific layer for a labeled dataset
- Train a linear classifier (or small MLP) to predict the label from the activations
- High accuracy → the information is linearly encoded at that layer

**Pros:**
- Simple, well-understood methodology
- Can test for any concept you have labels for
- Computationally cheap
- Good for hypothesis-driven investigation

**Cons:**
- Correlation, not causation — high probe accuracy doesn't mean the model *uses* that information
- Risk of the probe itself learning the task (especially non-linear probes)
- Limited to concepts you can label
- Doesn't reveal the computational mechanism

---

### 2.6 Representation Engineering / Activation Steering

**The Big Idea:** Identify a direction in activation space corresponding to a concept (e.g., "honesty", "refusal", "happiness") and add/subtract that direction during inference to steer model behavior.

**How it works:**
1. Collect activation vectors for prompts with/without a target concept
2. Compute the "concept direction" as the difference of mean activations (or via PCA, etc.)
3. During inference, add a scaled version of this direction to the residual stream
4. The model's behavior shifts accordingly (e.g., becomes more/less honest)

**Key work:**
- **Representation Engineering** (Zou et al., 2023): Foundational paper establishing the framework
- **RepE taxonomy** (Wehner et al., 2025): Comprehensive survey covering identification, operationalization, and control
- **Goodfire** (startup, 2024-2025): Commercial platform for feature steering using SAE-derived features
- **Activation Addition / CAA** (Turner et al., 2023): Contrastive Activation Addition for steering

**Pros:**
- Direct, practical intervention on model behavior
- No retraining required — works at inference time
- Can control abstract concepts (honesty, style, safety)
- Complementary to SAEs (SAE features can serve as steering vectors)

**Cons:**
- Steering vectors can have unintended side effects
- Magnitude calibration is tricky — too much steering degrades output quality
- Concept directions may be oversimplified (linear assumption)
- Managing multiple steering vectors simultaneously is an open problem

---

### 2.7 Patchscopes

**The Big Idea:** Use the LLM itself to explain its own internal representations by "patching" hidden states into carefully designed prompts.

**How it works (Geva et al., 2024):**
1. Run the model on a **source prompt** and extract a hidden state at layer l
2. Design a **target prompt** with few-shot examples that encourage the model to describe/decode representations
3. Patch the source hidden state into the target prompt at a specific position
4. The model's completion of the target prompt serves as a natural-language explanation of the hidden state

**Pros:**
- Leverages the model's own capabilities for explanation
- More expressive than vocabulary-space projection (logit lens)
- Can use a more capable model to explain a smaller model's representations
- Unifies several prior methods (logit lens, probing, CausalTracing) under one framework

**Cons:**
- Relatively new — less validated than established methods
- Results depend heavily on target prompt design
- May be unreliable for early layers or complex concepts
- Harder to automate at scale

---

## 3. Comparison Table: Pros & Cons

| Technique | Causal? | Hardware Cost | Ease of Use | Scalability | Best For |
|---|---|---|---|---|---|
| **SAEs** | No (observational) | Low-Medium | Medium | High | Feature discovery, understanding representations |
| **Logit Lens** | No | Very Low | Very Easy | High | Quick intuition, layer-by-layer prediction tracking |
| **Activation Patching** | Yes | Medium-High | Medium | Medium | Identifying causally important components |
| **Circuit Tracing** | Yes | High | Hard | Low-Medium | Deep understanding of specific behaviors |
| **Probing** | No | Low | Easy | High | Hypothesis testing for specific information |
| **RepE / Steering** | Interventional | Low | Medium | High | Controlling model behavior at inference |
| **Patchscopes** | Partial | Low-Medium | Medium | Medium | Natural-language explanations of internals |

---

## 4. Tooling Ecosystem

### Core Libraries

| Tool | Description | Install | GitHub Stars | Best For |
|---|---|---|---|---|
| **TransformerLens** | Purpose-built MI library by Neel Nanda. Reimplements 50+ architectures with hooks at every activation. The de facto standard. | `pip install transformer_lens` | ~5K | General MI research, activation caching, circuit analysis |
| **SAELens** | Training and analyzing SAEs. Maintains a registry of pre-trained SAEs for GPT-2, Gemma, Pythia, etc. | `pip install sae-lens` | ~1.3K | SAE training, loading pre-trained SAEs, feature analysis |
| **nnsight** | General-purpose library for inspecting/intervening on any PyTorch model. Supports remote execution on large models via NDIF. | `pip install nnsight` | ~870 | Large model research (70B+), any PyTorch architecture |
| **nnterp** | Lightweight wrapper around nnsight providing a unified TransformerLens-like interface across 50+ architectures without reimplementation. | `pip install nnterp` | New | Cross-architecture research, quick experiments |
| **circuit-tracer** | Anthropic's open-source tool for attribution graphs using cross-layer transcoders. | `pip install circuit-tracer` | ~2.7K | Circuit tracing, attribution graph visualization |

### Visualization & Exploration

| Tool | Description | URL |
|---|---|---|
| **Neuronpedia** | Interactive web platform for exploring SAE features and attribution graphs. No installation needed. | https://www.neuronpedia.org |
| **SAE-Vis** | Generate feature dashboards locally. | Part of SAELens ecosystem |

### Other Notable Tools

| Tool | Description |
|---|---|
| **Interpreto** | Python library for post-hoc explainability (attributions + concept-based). Works with HuggingFace models from BERT to LLMs. `pip install interpreto` |
| **TDHook** | Lightweight, generic interpretability framework based on tensordict. Handles complex composed models. |
| **sae-auto-interp** (EleutherAI) | Automated interpretation of SAE features using LLMs. Generates text explanations and scores them. |
| **tuned-lens** | Per-layer affine probes for more reliable intermediate predictions than logit lens. `pip install tuned-lens` |
| **mech-interp-toolkit** | Organized collection of MI methods (logit lens, activation patching, path patching, ACDC) on TransformerLens. |

---

## 5. Key Papers (2024-2026)

### Foundational & Surveys

1. **"Open Problems in Mechanistic Interpretability"** — Sharkey, Chughtai, Batson et al. (TMLR, Sep 2025)
   - The definitive roadmap of open problems. By researchers from Apollo, Anthropic, Google DeepMind, Eleuther, MIT, and more.
   - https://openreview.net/forum?id=91H76m9Z94

2. **"Locate, Steer, and Improve: A Practical Survey of Actionable MI in LLMs"** — Zhang et al. (Jan 2026)
   - Organized around the pipeline: Locate (diagnose) → Steer (intervene) → Improve (alignment, capability, efficiency).
   - https://arxiv.org/abs/2601.14004

3. **"Bridging the Black Box: A Survey on MI in AI"** — ACM Computing Surveys (Feb 2026)
   - High-level synthesis across neurons, circuits, and algorithms abstraction layers.
   - https://dl.acm.org/doi/10.1145/3787104

4. **"A Practical Review of MI for Transformer-Based Language Models"** — Rai et al. (updated Oct 2025)
   - Task-centric taxonomy; great roadmap for beginners.
   - https://arxiv.org/abs/2407.02646

### Sparse Autoencoders

5. **"Scaling and Evaluating Sparse Autoencoders"** — Gao, Dupré la Tour et al. (OpenAI, ICLR 2025)
   - TopK SAEs, clean scaling laws, trained 16M-latent SAE on GPT-4. Introduced new evaluation metrics.
   - https://openreview.net/forum?id=tcsZt9ZNKD

6. **"A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of LLMs"** — Shu et al. (EMNLP Findings, Nov 2025)
   - Comprehensive SAE survey: architectures, training strategies, evaluation methods, applications.
   - https://aclanthology.org/2025.findings-emnlp.89/

7. **"Revising and Falsifying SAE Feature Explanations"** — Ma et al. (NeurIPS 2025)
   - Introduces similarity-based evaluation and tree-based iterative explanation for SAE features.
   - https://neurips.cc/virtual/2025/poster/118303

8. **"Mechanistic Interpretability with SAE Neural Operators"** — Tolooshams et al. (Sep 2025, updated Feb 2026)
   - Extends SAEs from vector-valued to functional representations using Fourier neural operators.
   - https://arxiv.org/abs/2509.03738

### Circuit Discovery

9. **"Circuit Tracing: Revealing Computational Graphs in Language Models"** — Ameisen et al. (Anthropic, Mar 2025)
   - Introduced cross-layer transcoders and attribution graphs. Applied to Claude 3.5 Haiku.
   - https://transformer-circuits.pub/2025/attribution-graphs/methods.html

10. **"On the Biology of a Large Language Model"** — Lindsey et al. (Anthropic, Mar 2025)
    - Companion paper exploring fascinating circuit-level findings: multi-step reasoning, planning, multilingual representations.
    - https://transformer-circuits.pub/2025/attribution-graphs/biology.html

11. **"Attribution Patching Outperforms Automated Circuit Discovery"** — Syed, Rager, Conmy (BlackboxNLP 2024)
    - Shows attribution patching (gradient-based) beats ACDC and other methods for circuit recovery.
    - https://aclanthology.org/2024.blackboxnlp-1.25/

### Activation Patching & Interventions

12. **"How to Use and Interpret Activation Patching"** — Heimersheim & Nanda (Apr 2024)
    - Practical guide with best practices for activation patching.
    - https://arxiv.org/abs/2404.15255

### Representation Engineering

13. **"Taxonomy, Opportunities, and Challenges of Representation Engineering for LLMs"** — Wehner et al. (Mar 2025)
    - First comprehensive RepE survey: identification, operationalization, and control.
    - https://arxiv.org/abs/2502.19649

### Tools & Infrastructure

14. **"nnterp: A Standardized Interface for MI of Transformers"** — Dumas (Nov 2025)
    - Unified interface across 50+ model variants spanning 16 architecture families.
    - https://arxiv.org/abs/2511.14465

15. **"NNsight and NDIF: Democratizing Access to Foundation Model Internals"** — Fiotto-Kaufman et al. (ICLR 2025)
    - Remote execution for MI on massive models. Same code for local GPT-2 or remote Llama-405B.
    - https://github.com/ndif-team/nnsight

16. **"Open-Sourcing Circuit Tracing Tools"** — Anthropic (May 2025)
    - Release of the circuit-tracer library and Neuronpedia integration.
    - https://www.anthropic.com/research/open-source-circuit-tracing

### Bonus: Learning Resources

- **Neel Nanda's "How to Become a Mechanistic Interpretability Researcher"** (Sep 2025) — Opinionated, actionable career guide with concrete learning stages. https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/
- **learnmechinterp.com** — Curated topic-by-topic explanations of MI concepts (TransformerLens, SAELens, nnsight, activation patching, logit lens, etc.)
- **Neel Nanda's Comprehensive MI Glossary** — The canonical reference for MI terminology. https://www.neelnanda.io/mechanistic-interpretability/glossary

---

## 6. Hardware Requirements & Practical Advice

### The Good News: You Don't Need a GPU Cluster

The most common misconception about MI is that you need to run massive models locally. In reality:

| Approach | Hardware Needed | Notes |
|---|---|---|
| **SAEs on GPT-2 Small** | CPU only (4-8 GB RAM) | Pre-trained SAEs available. The demo script in this package runs on CPU. |
| **TransformerLens + GPT-2** | CPU or basic GPU | 124M params, fits easily in memory |
| **Logit Lens on any small model** | CPU | Just matrix multiplication |
| **SAELens with pre-trained SAEs** | CPU for GPT-2; GPU for larger models | Loading pre-trained SAEs avoids expensive training |
| **nnsight + NDIF (remote)** | No local GPU needed | Run experiments on Llama-70B+ remotely |
| **Neuronpedia (web)** | Just a browser | Explore SAE features and attribution graphs online |
| **circuit-tracer on Gemma-2-2B** | 1 GPU (8-16 GB VRAM) | Smallest supported model for circuit tracing |

### Recommended Starting Path

1. **Start with the demo script** (included) — runs on CPU, explores SAE features on GPT-2 Small
2. **Explore Neuronpedia** — browse features interactively at https://www.neuronpedia.org
3. **Try TransformerLens tutorials** — logit lens, direct logit attribution, activation caching
4. **Scale up with nnsight + NDIF** — run the same code on 70B+ models remotely

---

## 7. Demo Script Overview

The accompanying `mech_interp_demo.py` script demonstrates three lightweight MI techniques on **GPT-2 Small** (124M params, runs on CPU):

1. **Logit Lens** — See what GPT-2 would predict at each intermediate layer
2. **SAE Feature Analysis** — Load a pre-trained SAE, decompose activations into interpretable features, and explore what concepts the model uses
3. **Activation Steering** — Find a "formal vs. casual" direction and steer the model's generation style

All three run on CPU with ~4 GB RAM. No GPU required.

See `mech_interp_demo.py` for the full, runnable code with detailed comments.

---

*This report was compiled for research purposes. For the latest updates, check the papers and repositories linked above.*
