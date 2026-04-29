# Today

## PhD study session — Geva et al. 2021

**Paper:** "Transformer Feed-Forward Layers Are Key-Value Memories"
**arxiv:** https://arxiv.org/abs/2012.14823

### Why this paper, why now

Your Experiment 2 found that prompt injection is **MLP-mediated, not attention-mediated** —
clamping the top 5 injection-sensitive attention heads had zero effect on output.
This paper is the theoretical backbone for *why* that is true.

### The core idea

Each FFN layer is two linear layers: `FFN(x) = W_V · ReLU(W_K · x)`

Read it as a **key-value memory bank:**
- `W_K` — pattern detectors (keys), ~4× more than the embedding dimension
- `W_V` — what to emit when a key fires (values)
- Knowledge about facts, associations, style lives here — not in attention heads
- Attention routes and aggregates; **MLPs recall**

Creative-writing framings activate stored stylistic associations (poetry → Romance language → French-like outputs)
via MLP key-value pairs. Of course clamping attention heads does nothing.

### Reading order (easy)

1. The paper: focus on §3 and Figure 2 (~8 pages total, short)
2. Neel Nanda's mech-interp 101 video series if you want a walkthrough
3. Follow-up when ready: Meng et al. 2022 "ROME" (uses this exact framing to edit factual memories)

### What you'll walk away with

- Mechanistic vocabulary for writing up *why* injection is MLP-mediated in your thesis
- Intuition for how SAEs (which you already use) decompose this MLP value space
- One clean foundational paper checked off

---

*Created the night before. You said you'd procrastinate otherwise. Don't.*
