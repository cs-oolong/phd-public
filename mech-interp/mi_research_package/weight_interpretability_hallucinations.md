# Weight-Based Interpretability & Hallucination Prevention in LLMs

*Research summary — April 2026*

---

## The Core Idea

Most MI techniques analyze **activations** (what happens when the model processes a specific input). But a parallel line of research analyzes the **weights themselves** — the static parameters that encode everything the model learned during training. This is powerful because:

- You don't need to run the model on specific inputs
- You can understand what knowledge is stored *where* in the network
- You can surgically edit weights to fix factual errors (hallucinations) without retraining

---

## 1. Knowledge Neurons & Factual Localization

**The Big Idea:** Factual knowledge (e.g., "The Eiffel Tower is in Paris") is stored in specific neurons within the MLP (feed-forward) layers of transformers. These are called **knowledge neurons**.

**Key finding:** Meng et al. (2022) showed through **causal tracing** that factual associations are concentrated in middle-layer MLPs. When the model recalls "The Eiffel Tower is in [Paris]", specific MLP weights at specific layers are causally responsible. This was the foundation for weight-editing approaches.

**Paper:** "Locating and Editing Factual Associations in GPT" — Meng, Bau, Andonian, Belinkov (NeurIPS 2022)
- https://rome.baulab.info/
- Foundational work; heavily cited across the field

---

## 2. ROME & MEMIT — Editing Weights to Fix Hallucinations

This is likely the work you heard about. **ROME** (Rank-One Model Editing) and **MEMIT** (Mass Editing Memory in a Transformer) directly modify model weights to correct factual errors.

### ROME (Rank-One Model Editing)
- **How:** Identifies which MLP layer stores a specific fact, then performs a rank-one update to the weight matrix to change the stored association
- **Example:** If GPT thinks "The president of France is Macron" but you want to update it to a new president, ROME modifies the specific weights responsible
- **Result:** Changes the model's factual output without affecting unrelated knowledge
- **Limitation:** One fact at a time

### MEMIT (Mass Editing Memory in a Transformer)
- **How:** Extends ROME to edit thousands of facts simultaneously by distributing updates across multiple layers
- **Result:** Can update 10,000+ facts at once while maintaining model quality
- **Paper:** https://memit.baulab.info/

### Can Knowledge Editing Really Correct Hallucinations? (ICLR 2025)
- **Key finding:** Huang et al. built **HalluEditBench** — a benchmark of 6,000+ real hallucinations across 9 domains
- **Result:** Knowledge editing methods (ROME, MEMIT, etc.) can partially correct hallucinations, but face limitations in generalization and robustness
- **Paper:** https://arxiv.org/abs/2410.16251

---

## 3. Watch the Weights (ICLR 2026) — Weight Analysis for Safety

This is a very recent and exciting paper that directly analyzes weights for interpretability:

**"Watch the Weights: Unsupervised Monitoring and Control of Fine-tuned LLMs"** — Zhong & Raghunathan, CMU (ICLR 2026)

**Core idea:** Instead of looking at activations, analyze the **top singular vectors of the weight difference** between a fine-tuned model and its base model. These singular vectors correspond to newly acquired behaviors.

**Key results:**
- Detects backdoor attacks with up to 100% success rate (FPR < 1%)
- Detects inference on "unlearned" topics with 95.42% accuracy
- Can even **recover** information that was supposedly "unlearned"
- Applied to commercial models (OLMo, Llama, Qwen) — uncovered model-specific fine-tuning focuses like marketing strategies and Midjourney prompt generation

**Why this matters for hallucinations:** If you can identify what behaviors were introduced during fine-tuning by looking at weight differences, you can potentially identify and remove hallucination-prone patterns.

**Code:** https://github.com/fjzzq2002/WeightWatch

---

## 4. Neuron-Level Knowledge Editing (2025)

### FiNE — Fine-grained Neuron-level Knowledge Editing
- **How:** Goes beyond layer-level localization to identify **specific neurons** within MLP layers that store specific facts
- **Result:** More precise editing = better locality (fewer side effects when correcting a hallucination)
- **Paper:** "Precise Localization of Memories" — Pan et al. (Mar 2025) — https://arxiv.org/abs/2503.01090

### MicroEdit — SAEs for Knowledge Disentanglement (EMNLP 2025)
- **How:** Uses Sparse Autoencoders to disentangle polysemantic neurons before editing, so edits don't spill over to unrelated knowledge
- **Directly combines SAEs + weight editing** — very relevant to your research interests
- **Paper:** "MicroEdit: Neuron-level Knowledge Disentanglement and Localization in Lifelong Model Editing" — Wang et al. (EMNLP 2025)

### CKU — Constrained Knowledge Unlearning
- **How:** Scores neurons in MLP layers to identify knowledge-storing neurons, then selectively prunes gradients during unlearning to preserve useful knowledge while removing harmful content
- **Application:** Safety alignment — removing harmful knowledge while keeping the model functional
- **Paper:** "Safety Alignment via Constrained Knowledge Unlearning" — Shi et al. (May 2025) — https://arxiv.org/abs/2505.18588

---

## 5. Hallucination Detection via Internal Representations

### Contrastive Neuron Steering (CNS) — SAEs for Hallucination Mitigation
- **How:** Uses SAEs to decompose visual embeddings into interpretable neurons. Identifies "always-on" neurons (stable) vs "image-specific" neurons (variable). Hallucinations result from spurious activations of image-specific neurons.
- **Intervention:** Amplify informative neurons, suppress perturbation-induced activations
- **Paper:** "Towards Interpretable Hallucination Analysis and Mitigation in LVLMs via Contrastive Neuron Steering" — Lyu et al. (2025) — https://arxiv.org/abs/2602.00621

### CLAP — Cross-Layer Attention Probing (ECAI 2025)
- **How:** Processes LLM activations across the entire residual stream as a joint sequence to detect hallucinations
- **Result:** Enables a "detect-then-mitigate" strategy — first detect hallucinated tokens, then intervene
- **Paper:** "Cross-Layer Attention Probing for Fine-Grained Hallucination Detection" — Suresh et al. (ECAI 2025) — https://arxiv.org/abs/2509.09700

### LLM-CAS — Dynamic Neuron Perturbation (AAAI 2025)
- **How:** Uses hierarchical reinforcement learning to train an agent that dynamically selects optimal neuron perturbations during inference to correct hallucinations in real-time
- **Result:** +10.98pp accuracy on StoryCloze, +2.71pp on TriviaQA
- **Paper:** "LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction"

---

## 6. Query Localization — Beyond Knowledge Neurons

**"Knowledge Localization: Mission Not Accomplished? Enter Query Localization!"** — Chen et al. (2024, updated 2025)

**Key critique:** The original "Knowledge Neuron" thesis is too simplistic. Knowledge isn't stored in isolated neurons — it depends on the **query** (how you ask). The same fact may be stored differently depending on context.

**Proposes:** Query Localization (QL) — a more nuanced view where knowledge storage and expression depend on both the MLP weights AND the attention module.

**Paper:** https://arxiv.org/abs/2405.14117

---

## Summary: The Weight-Based Interpretability Landscape

| Approach | What It Does | Hallucination Connection |
|---|---|---|
| **Knowledge Neurons** | Locate where facts are stored in MLP weights | Identify which weights cause wrong facts |
| **ROME/MEMIT** | Surgically edit weights to change stored facts | Directly fix hallucinated facts |
| **Watch the Weights** | Analyze weight diffs via SVD to find new behaviors | Detect/remove hallucination-prone patterns |
| **FiNE/MicroEdit** | Neuron-level precision editing with SAEs | Fix hallucinations with minimal side effects |
| **CNS** | SAE-based neuron steering | Suppress hallucination-causing neurons |
| **LLM-CAS** | RL-based dynamic neuron perturbation | Real-time hallucination correction |

---

## Practical Relevance for Your Research

The connection between **your character comparison work** and these weight-analysis methods is direct:

1. You could use **causal tracing** (ROME-style) to find which MLP layers store "Harry Potter" vs "Voldemort" knowledge
2. You could use **knowledge editing** to see what happens when you edit character associations in the weights
3. You could combine **SAE features** (from your demo) with **weight analysis** to understand whether character representations are stored in weights vs computed dynamically
4. The **Watch the Weights** approach could reveal how fine-tuning on character-related data changes the weight structure

This is a very active area — the ICML 2026 Mechanistic Interpretability Workshop (Seoul, July 2026) will likely feature many new weight-analysis papers.
