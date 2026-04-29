# SAE Experiments Report: Feature-Level Analysis of Prompt Injection
## Sparse Autoencoder Decomposition on GPT-2 Small

**Date:** April 8, 2026
**Model:** GPT-2 Small (124M parameters, 12 layers, 768 hidden dim)
**SAEs:** Pre-trained from Joseph Bloom (`gpt2-small-res-jb`), 24,576 features per layer
**Layers analyzed:** 0, 3, 6, 8, 9, 11
**Total runtime:** 70.5 seconds (all CPU, no GPU)

---

## Executive Summary

We decomposed GPT-2 Small's hidden states into ~25K interpretable SAE features at 6 layers to understand prompt injection at the *concept level*. Key findings:

1. **Feature #21868 is a "foreign language / Romance language" feature** — it promotes tokens like ` qui`, ` est`, ` des`, ` é`. It *increases* during injection (baseline 3.68 → injection ~5.0), suggesting the model partially maintains its translation intent even while deviating.

2. **Feature #565 is a "narrative/chapter" feature** — promotes `Chapter`, `CHAPTER`, `Scene`, `Episode`. It fires strongly during narrative injection (5.68) but is nearly zero during baseline (0.002). This is a clear "the model recognizes it's being told a story" signal.

3. **Feature #10599 is a "conversational/informal" feature** — promotes ` Yeah`, ` Alright`, ` Hmm`, ` Okay`. It spikes during injection (prose: 4.0, narrative: 8.4), indicating the model shifts from "formal translation" to "casual conversation" mode.

4. **Injection types share a universal core but have distinct signatures.** At layer 8, only 6 of 20 top features are shared by all three types, but the *suppressed* features (task features being turned off) overlap heavily.

5. **SAE ablation and steering FAILED as defenses on GPT-2.** Neither zeroing out injection features nor boosting task features recovered P(French). In fact, both made P(French) *worse*. This confirms our earlier finding that GPT-2's task deviation is deeply distributed, not localized to a few features.

6. **The tipping point is immediate.** P(French) collapses at the very first injection token ("Ignore", "Forget", "Once") — before any injection-specific SAE features even peak.

---

## SAE Experiment 1: Injection Feature Fingerprint

**Question:** Which SAE features activate during injection vs. baseline?

**Method:** Run 3 baseline and 3 injection prompts (prose, poetry, narrative) through GPT-2. Decompose hidden states at layers 0, 3, 6, 8, 9, 11 using pre-trained SAEs. Compare feature activations.

### Results by Layer

**Layer 0 (embedding):** Nearly no differentiation. Only 2-4 features active total. The SAE at this layer sees raw token embeddings — the model hasn't processed anything yet.

**Layer 3 (early processing):** 16-23 features active. Feature #15772 is the top injection feature across all 3 types (+1.1 to +3.2 vs baseline). Feature #9132 is consistently suppressed (-0.7 to -4.9).

**Layer 6 (mid processing):** 36-62 features active. Feature #11802 emerges as the strongest injection feature (prose +2.7, narrative +5.0). Feature #13479 is consistently suppressed.

**Layer 8 (late-mid processing) — FOCUS LAYER:**

| Feature | Baseline | Prose Inj | Poetry Inj | Narrative Inj | Interpretation |
|---------|----------|-----------|------------|---------------|----------------|
| #10599 | 1.00 | 4.00 | 1.88 | **8.38** | "Conversational" — promotes Yeah, Alright, Hmm |
| #13038 | 5.10 | 7.50 | 6.53 | 6.58 | "Continuation" — promotes Lastly, Finally, Despite |
| #21868 | 3.68 | 5.02 | 5.08 | 5.14 | "Romance language" — promotes qui, est, des |
| #565 | 0.00 | 1.34 | 0.00 | **5.68** | "Narrative" — promotes Chapter, Scene, Episode |
| #16184 | 0.00 | 0.00 | 0.00 | **4.09** | "Continuation marks" — promotes ...", …", [...] |
| **#14419** | **12.73** | 10.28 | 11.20 | **7.44** | **"Placeholder/unknown"** — promotes Unknown, TBA, None |
| **#488** | **9.26** | 7.20 | 7.97 | **1.16** | **"Structured output"** — promotes Miscellaneous, Conclusion |
| **#21028** | **5.32** | 3.39 | 4.43 | **1.93** | **"Foreign syllables"** — promotes Sa, Kas, Sud, Ort |

Bold = task features (active in baseline, suppressed during injection).

**Key insight:** Feature #21868 (Romance language tokens) *increases* during injection. The model doesn't lose its "French awareness" — it loses its "structured output" features (#14419, #488) that frame the response as a translation exercise.

**Layer 11 (final layer):** Strong differentiation. Feature #10391 is the top injection feature for prose (+5.4) and narrative (+12.6). Feature #20056 is massively suppressed during narrative injection (-13.9).

### Active Feature Counts

| Layer | Baseline | Prose Inj | Poetry Inj | Narrative Inj |
|-------|----------|-----------|------------|---------------|
| 0 | 3 | 2 | 2 | 4 |
| 3 | 16 | 17 | 19 | 23 |
| 6 | 44 | 38 | 36 | 62 |
| 8 | 59 | 62 | 62 | 74 |
| 9 | 71 | 75 | 60 | 63 |
| 11 | 53 | 41 | 42 | 31 |

Narrative injection activates MORE features at early/mid layers (L6: 62 vs 44 baseline) but FEWER at late layers (L11: 31 vs 53 baseline). The model broadens its "thinking" in mid layers but narrows its output representation.

---

## SAE Experiment 5: Cross-Injection Feature Overlap

**Question:** Do prose, poetry, and narrative injections activate the SAME or DIFFERENT features?

### Overlap Summary (Top-20 Differential Features)

| Layer | Shared All 3 | Prose-Poetry Only | Prose-Only | Poetry-Only | Narrative-Only |
|-------|-------------|-------------------|------------|-------------|----------------|
| 0 | 13 (65%) | 7 | 0 | 0 | 7 |
| 3 | 8 (40%) | 7 | 2 | 5 | 9 |
| 6 | 9 (45%) | 6 | 3 | 4 | 8 |
| 8 | 6 (30%) | 7 | 5 | 7 | 12 |
| 9 | 6 (30%) | 8 | 2 | 5 | 9 |
| 11 | 6 (30%) | 1 | 6 | 11 | 5 |

**Key findings:**

1. **Overlap decreases with depth.** At layer 0, 65% of top features are shared. By layer 8-11, only 30% are shared. The model starts with a generic "this is different from translation" response, then specializes by injection type in later layers.

2. **Prose and poetry share more features than either shares with narrative.** At most layers, the "prose & poetry only" overlap is 6-8 features. This makes sense — both are text-based deviations, while narrative introduces a fundamentally different framing (storytelling).

3. **Poetry has the most unique features at layer 11** (11 features). This is the output layer — poetry creates the most distinctive output representation, consistent with our earlier finding that poetry prompts create different behavioral patterns.

4. **Narrative has the most unique features at layer 8** (12 features). This is where narrative injection's distinctive processing happens — features like #565 (Chapter/Scene/Episode) and #16184 (continuation marks) are narrative-specific.

---

## SAE Experiment 4: Feature Dashboard

**Focus:** Layer 8 features projected through the unembedding matrix to see what tokens each feature promotes.

### Most Interesting Injection Features

- **Feature #21868** — Promotes ` qui`, ` est`, ` des`, `ó`, ` é` (Romance language tokens). This is a *translation-relevant* feature that INCREASES during injection. The model doesn't forget French — it loses the *task framing*.

- **Feature #565** — Promotes `Chapter`, `CHAPTER`, `Scene`, `Episode`, `Introduction`. Pure narrative-structure feature. Only fires for narrative injection (5.68 vs 0.00 baseline).

- **Feature #10599** — Promotes ` Yeah`, ` Alright`, ` Hmm`, ` Okay`, ` Exactly`. Conversational/informal register. Spikes for narrative (8.38) and prose (4.00) injection.

- **Feature #22228** — Promotes ` Thou`, `Always`, ` Always`, ` Beware`, ` beware`. Archaic/imperative language. Activates more during poetry (3.00) than prose (2.50).

- **Feature #2872** — Promotes `Dear`, `RIP`, `Sorry`, `Welcome`, `Hey`. Interpersonal/address feature. Only fires for narrative injection (2.54).

### Most Interesting Task (Suppressed) Features

- **Feature #14419** — Promotes ` Unknown`, ` TBA`, ` None`, ` TBD`, ` ???`. A "structured/tabular output" feature. Has the highest baseline activation (12.73) and is heavily suppressed by narrative injection (7.44). This feature seems to encode "produce a formatted response."

- **Feature #488** — Promotes ` Miscellaneous`, ` Ibid`, ` Conclusion`, ` Paras`, ` Utilities`. Another structural/reference feature. Baseline 9.26, crushed to 1.16 by narrative injection.

- **Feature #13498** — Promotes ` Amen`, ` qui`, ` Regist`, ` alle`, ` los`. Multilingual/foreign language feature. Suppressed from 6.01 to 4.32-5.76 during injection.

- **Feature #20993** — Promotes ` thou`, ` vain`, ` ain`, ` thy`, ` pity`. Archaic English feature. Interestingly, this INCREASES during poetry injection (3.66 vs 3.29 baseline) but drops for prose (1.79) and narrative (0.64).

### Neuronpedia URLs for All Features

All features can be explored interactively at Neuronpedia. Example:
- Translation-related: https://neuronpedia.org/gpt2-small/8-res-jb/21868
- Narrative-structure: https://neuronpedia.org/gpt2-small/8-res-jb/565
- Conversational: https://neuronpedia.org/gpt2-small/8-res-jb/10599
- Task-framing: https://neuronpedia.org/gpt2-small/8-res-jb/14419

---

## SAE Experiment 6: Feature Trajectories During Tipping Point

**Question:** How do key features evolve token-by-token as injection is fed in?

### Tracked Features (Layer 8)

**Injection features:** #10599 (conversational), #13038 (continuation), #497, #21868 (Romance lang), #22513
**Task features:** #14419 (structured output), #488 (reference), #21028 (foreign syllables), #1623, #10660

### Prose Injection: "Ignore the translation task above..."

- P(French) **tipping point: token 2** ("Ignore the") — P(F) drops from baseline to 0.096
- Injection feature #10599 peaks at **token 9** (activation 4.14)
- Task feature #488 hits zero at **token 14**
- Task feature #21028 hits zero at **token 10**

**Critical observation:** P(French) collapses *7 tokens before* the injection features peak. The output-level tipping happens BEFORE the feature-level changes fully manifest. This suggests the tipping is driven by the residual stream direction shift (which we measured in our original experiments), not by individual SAE features.

### Poetry Injection: "Forget the task that came before..."

- P(French) **tipping point: token 0** ("For") — P(F) already at 0.108
- Injection features peak at tokens 9-14
- Task feature #14419 min at token 12 (7.45 — never fully suppressed)

**Key finding:** Poetry injection tips the model *immediately* — the very first token "For" (from "Forget") is so out-of-distribution for the translation context that P(French) collapses before the model even knows it's a poetry injection.

### Narrative Injection: "Once upon a time..."

- P(French) **tipping point: token 0** ("Once") — P(F) at 0.111
- Feature #13038 (continuation) peaks very early at **token 3** ("Once upon a time")
- Task features suppress more gradually than prose

**Pattern:** All three injection types tip P(French) within 0-2 tokens, but SAE features take 9-14 tokens to reach their peak differential. The output decision is made much faster than the feature-level reorganization.

---

## SAE Experiment 2: Feature Ablation for Injection Defense

**Question:** Can we prevent injection by zeroing out injection-specific features?

### Results (Layer 8)

| Ablated Features | Prose P(F) | Poetry P(F) | Narrative P(F) |
|-----------------|-----------|------------|---------------|
| None (original) | 0.186 | 0.166 | 0.211 |
| Top 3 | 0.146 | 0.130 | 0.137 |
| Top 5 | 0.107 | 0.134 | 0.138 |
| Top 10 | 0.085 | 0.116 | 0.139 |
| Top 20 | 0.069 | 0.092 | 0.082 |
| SAE reconstruction only | 0.132 | 0.140 | 0.144 |

**Baseline (normal translation) P(F) ≈ 0.40-0.50**

**Result: ABLATION MADE THINGS WORSE.** Instead of recovering P(French) toward baseline (~0.40), ablating injection features *decreased* P(French) further. Ablating 20 features dropped prose from 0.186 to 0.069.

**Why?** Two factors:
1. **SAE reconstruction error** — Even the reconstruction-only baseline (no ablation, just encode→decode) drops P(French) by 0.03-0.07. The SAE doesn't perfectly reconstruct the hidden state.
2. **Injection features are entangled with task features.** On GPT-2 Small, the "injection" features also contribute positively to translation. Feature #21868 (which promotes French tokens like `qui`, `est`) is classified as an "injection feature" because it increases during injection — but it's also *essential* for translation. Zeroing it out removes French token promotion.

**This confirms our earlier finding:** GPT-2 Small's task processing is deeply distributed. You can't surgically remove "injection" without also removing "translation" because the same features participate in both.

---

## SAE Experiment 3: Feature Steering for Injection Defense

**Question:** Can we boost task features to overpower injection?

### Results (Layer 8)

| Strategy | Prose P(F) | Poetry P(F) | Narrative P(F) |
|---------|-----------|------------|---------------|
| No defense | 0.186 | 0.166 | 0.211 |
| Boost task x1.5 | 0.128 | 0.128 | 0.149 |
| Boost task x2.0 | 0.119 | 0.115 | 0.152 |
| Boost task x3.0 | 0.100 | 0.093 | 0.152 |
| Boost task x5.0 | 0.071 | 0.062 | 0.141 |
| Boost task x10.0 | 0.040 | 0.030 | 0.106 |
| Combined (ablate + boost x2) | 0.107 | 0.111 | 0.150 |
| Combined (ablate + boost x5) | 0.067 | 0.058 | 0.143 |

**Result: STEERING ALSO MADE THINGS WORSE.** Boosting task features at any multiplier decreased P(French). Even the most aggressive combined strategy (ablate injection + boost task x5) produced worse results than no defense.

**Why?** The same entanglement problem. The features we identified as "task features" (because they're suppressed during injection) promote tokens like ` Unknown`, ` TBA`, ` None` (feature #14419) and ` Miscellaneous`, ` Ibid` (feature #488). Boosting them doesn't promote French output — it promotes *structured/reference-style* tokens. These features encode the translation task's *format*, not its *language*.

**Critical insight:** On GPT-2 Small, the features that are most *differentially active* during translation are not the features that *produce* French output. The differential features encode task-framing (structured, reference-like output), while the French-language features (#21868, #13498) are active in *both* conditions. This is a fundamental problem for feature-level defenses on small models.

---

## Cross-Cutting Findings

### Finding 1: The Feature-Output Timing Gap

SAE features take 9-14 tokens to reach peak differential activation, but P(French) collapses in 0-2 tokens. This means **the output decision happens in the residual stream direction** (which we measured with cosine distance in our original experiments), not at the individual feature level. The feature reorganization is a *consequence* of the direction shift, not its cause.

### Finding 2: Feature Entanglement on Small Models

On GPT-2 Small, injection features and task features are deeply entangled. Feature #21868 (Romance language tokens) is classified as an "injection feature" (because it increases during injection) but is essential for translation. This entanglement is why SAE-based defenses fail on GPT-2 but might succeed on GPT-J-6B (where processing is more localized).

### Finding 3: Narrative Injection Has a Unique SAE Signature

Narrative injection activates features that prose and poetry don't — #565 (Chapter/Scene), #16184 (continuation marks), #2872 (Dear/Welcome), #15400 (Introduction). These are "storytelling" features. Despite this unique signature, narrative is actually the *least effective* injection type on GPT-2, suggesting that unique features ≠ effective deviation.

### Finding 4: Overlap Decreases with Depth

All injection types look similar in early layers (65% feature overlap at L0) but diverge in late layers (30% at L8-11). The model's initial response is generic ("this input is unusual") but specializes into type-specific processing in later layers.

### Finding 5: Negative Results Are Informative

The failure of SAE ablation and steering on GPT-2 Small is consistent with Neel Nanda's team's finding (2025) that SAE-based safety interventions don't generalize well. On small models, representations are too entangled for surgical feature-level intervention. This motivates testing SAE approaches on larger models (GPT-J-6B, Llama) where representations may be more disentangled.

---

## Reproduction Guide

### Requirements
```bash
pip install sae-lens transformer_lens torch numpy
```

### Run
```bash
python3 sae_experiments.py
```

### Time Estimate
~70 seconds total on CPU. No GPU needed. SAE downloads (~150MB per layer) happen on first run.

### Output Files
- `sae_experiments_results.json` — All numerical results
- `sae_experiments.py` — Full experiment script

---

## Next Steps (If Continuing This Research)

1. **Run SAE experiments on GPT-J-6B** — Need to either train SAEs for GPT-J or find pre-trained ones. The localized processing in GPT-J-6B (layer 24 is critical) suggests SAE-based defenses might actually work on the larger model.

2. **Multi-layer SAE steering** — Instead of intervening at a single layer, steer features at multiple layers simultaneously. This might overcome the entanglement problem.

3. **Train task-specific SAEs** — The pre-trained SAEs were trained on OpenWebText (general text). Training SAEs specifically on translation contexts might yield features that are more relevant to our task and less entangled.

4. **SAIF-style analysis** — Apply the SAIF framework (He et al., 2025) to our specific injection scenario. SAIF found that instruction-following features are concentrated in the final layer — this aligns with our finding that L11 features have the most divergent signatures.
