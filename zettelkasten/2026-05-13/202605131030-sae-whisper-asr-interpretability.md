---
date: 2026-05-13
time: 10:30
tags: [sae, sparse-autoencoder, mechanistic-interpretability, whisper, asr, audio, mlp, latent-space]
paper: "Mechanistic Interpretability of ASR models using Sparse Autoencoders"
arxiv_id: "2605.12225"
authors: ["Dan Pluth", "Zachary Nicholas Houghton", "Yu Zhou", "Vijay K. Gurbani"]
---

# SAE on Whisper ASR — Reading Notes

## Paper Summary
First work applying Sparse Autoencoders (SAE) to an ASR model (Whisper). Extracts frame-level embeddings from Whisper encoder, trains high-dimensional sparse latent space. Uncovers monosemantic features across linguistic (phonetic, semantic, morphological, lexical) and non-linguistic boundaries. Demonstrates cross-lingual feature steering.

## Key Concepts Explained

### ASR (Automatic Speech Recognition)
Models that convert spoken audio to text. Example: OpenAI Whisper. Input: audio frames (~20-40ms chunks). Output: transcribed text. Unlike LLMs that process text tokens, ASRs process raw audio waveforms.

### Dense Vector
A regular vector where most entries are non-zero. Contrast with **sparse vector** where most entries are zero. Neural network activations are naturally dense — every neuron fires to some degree for every input. Example: `[0.12, -0.05, 0.88, 0.33, -0.71, ...]` (all slots non-zero).

### Latent Space
The intermediate representation between encoder and decoder in an autoencoder. "Latent" means hidden/underlying — not directly observable. It's a vector space (e.g., R^10000) where each point is a possible representation of some input. We discover what each dimension means by observing which inputs make it fire.

### Overcomplete
The latent dimension is larger than the original input dimension. Example: 768-dim activation → 10,000-dim or 100,000-dim latent space. Provides more "slots" so each can be specialized. Like a dictionary with way more words than needed.

### Why Sparsity + Overcompleteness Works

**Sparsity constraint** (most entries must be zero) forces competition:
- Each dimension must be useful on its own
- Dimensions become **disentangled** — each captures a distinct, independent concept

**Overcompleteness** provides enough "room" for specialization:
- Without it, dimensions would be polysemantic (encode multiple unrelated things)
- With it, slot #4,237 can *just* mean "phoneme /k/"

Together they create **superposition**: the model represents many more concepts than dimensions by using sparse combinations. Anthropic calls this reversing "polysemanticity" — instead of one neuron encoding many things, many neurons each encode one thing, activated sparsely.

**Analogy:** Describing a face with 10 adjectives (dense, limited) vs. choosing from 10,000 words but only picking 5 (sparse, overcomplete). The second forces each word to be precise and interpretable.

### Feature Steering
Manually activating a latent dimension and observing how the model's output changes. Proves causality (not just correlation) of the feature.

## SAE Training Details

### Architecture
```
Input: activation from target model (e.g., 768-dim)
    ↓
Encoder (linear layer): 768 → 10,000
    ↓
ReLU or TopK → enforces sparsity
    ↓
Sparse latent vector (mostly zeros)
    ↓
Decoder (linear layer): 10,000 → 768
    ↓
Output: reconstructed activation
```
Encoder and decoder are typically just linear transformations (matrices).

### Training Data
Activations from the **target model** (frozen). Process:
1. Feed many inputs through target model (e.g., audio samples through Whisper)
2. Extract activation vectors from an intermediate layer (e.g., layer 6 of encoder)
3. These activations become the SAE's training data

The SAE learns to reconstruct what the target model is doing internally.

### Loss Function
Two objectives combined:

1. **Reconstruction loss:** Output should match input activation
   - Usually MSE: `||input - output||²`

2. **Sparsity penalty:** Latent vector should be sparse
   - Usually L1 penalty: `λ * sum(|latent|)`

**Total loss = MSE(input, output) + λ * L1(latent)**

The λ hyperparameter controls the tradeoff:
- High λ → very sparse but worse reconstruction
- Low λ → good reconstruction but less sparse

### Performance Metrics
1. **Reconstruction MSE:** How well SAE recovers original activation (lower is better)
2. **Sparsity (L0 norm):** Fraction of latent dimensions active (target: ~1-5%)
3. **Feature interpretability:** Do dimensions correspond to human-understandable concepts?

### Training Process
Standard gradient descent:
1. Sample batch of activations from target model
2. Forward pass through SAE
3. Compute loss (reconstruction + sparsity)
4. Backpropagate and update encoder/decoder weights
5. Repeat — target model stays frozen

### Dead Neuron Problem
Sparsity constraint can cause many latent dimensions to become "dead" (never activate), wasting capacity. Solutions:
- **TopK activation:** Keep only K largest values (guarantees exact sparsity)
- **Resampling:** Periodically reinitialize dead neurons to match active ones

## Causality via Steering (Ablation)

### The Problem: Correlation ≠ Causation
A latent firing when /k/ is present doesn't mean it *causes* /k/ processing. It could be a byproduct.

### Steering Procedure
**Step 1: Find candidate feature**
- Observe which inputs make latent #4,237 fire
- Notice it consistently activates on frames containing /k/

**Step 2: Run normally**
- Audio → Whisper encoder → ε → SAE → z → decode → ε' → decoder → transcript T

**Step 3: Intervene (steer)**
- Same audio, same ε
- Pass through SAE → get z
- **Manually modify z**: set z[4237] = 0 (ablation) or z[4237] = 10.0 (amplification)
- Decode modified z → ε'' → decoder → modified transcript T'

**Step 4: Compare**
- If T' lacks /k/ where T had it → latent is **causally necessary**
- If amplifying z[4237] hallucinates /k/ → latent is **causally sufficient**

### Why This Is Causal
Direct manipulation of internal representation + predictable behavior change = gold standard for causality (like gene knockout experiments).

### Concrete Example: Cross-lingual Steering
- Take English audio
- Amplify "Spanish" latent
- Decoder outputs Spanish instead of English
→ Proves latent directly controls language choice

## What "A Latent Fires" Means

### Definition
"Latent #4,237 fires" means **z[4237] > 0** — it's one of the active (non-zero) entries in the sparse latent vector.

### How They Know It Fires for /k/
1. Use forced alignment to label each audio frame with its phoneme
2. Across millions of frames, check: "Of all /k/ frames, what fraction have z[4237] > 0?"
3. High fraction = strong correlation

### Visual Example
For a single frame with /k/ sound (TopK=45):
```
Latent vector z (16,000 dims):
z[0]     = 0.0
z[1]     = 0.0
...
z[4237]  = 5.2   ← FIRES! (non-zero, active)
...
z[15000] = 0.0
```
Only 45 entries are non-zero. The rest are exactly 0.

### Activation Strength
The value itself matters: 5.2 = strongly active, 0.1 = weakly active. Used to rank importance.

## Paper Contributions
1. First SAE-based mechanistic interpretability on ASR model
2. Whisper learns rich diverse representations: phonetic, semantic, morphological, positional, lexical
3. SAE is effective interpretability tool for probing Whisper internals

## Links
- arXiv: http://arxiv.org/abs/2605.12225
- PDF saved at: `~/monorepo/phd/papers/2605.12225.pdf`
- Text saved at: `~/monorepo/phd/papers/2605.12225.txt`
