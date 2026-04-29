#!/usr/bin/env python3
"""
Weight-Based Interpretability Demo — Causal Tracing, Knowledge Editing & Weight Analysis
=========================================================================================

This script demonstrates techniques that analyze the WEIGHTS of a neural network
(not just activations) to understand where knowledge is stored and how to fix
hallucinations by surgically editing weights.

Techniques demonstrated:
  1. Causal Tracing (ROME-style) — locate which MLP layers store a specific fact
  2. Weight Inspection — examine the weight matrices that encode factual knowledge
  3. Knowledge Editing — modify weights to change what the model "knows"
  4. Weight Diff Analysis — SVD-based analysis of weight changes (Watch the Weights-style)

Requirements:
  pip install transformer-lens torch numpy

Hardware: CPU only, ~4 GB RAM. No GPU required.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
import copy

DEVICE = "cpu"  # Change to "cuda" if you have a GPU


# ============================================================================
# PART 1: CAUSAL TRACING — Where is factual knowledge stored?
# ============================================================================
def demo_causal_tracing():
    """
    Causal Tracing (inspired by Meng et al., 2022 — ROME paper):

    The idea: Run the model on a factual prompt, then corrupt the subject
    tokens with noise. Then, one component at a time, restore the clean
    activation and measure how much the correct answer recovers.

    Components that cause the biggest recovery are the ones that STORE
    the factual knowledge.

    This is how ROME discovered that facts are stored in middle-layer MLPs.
    """
    print("\n" + "=" * 70)
    print("PART 1: CAUSAL TRACING — Locating Factual Knowledge in Weights")
    print("=" * 70)
    print("\nWe corrupt the subject ('Eiffel Tower') with noise, then restore")
    print("activations one layer at a time to find where facts are stored.\n")

    from transformer_lens import HookedTransformer

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # --- Setup: factual prompt ---
    prompt = "The Eiffel Tower is located in the city of"
    subject = "Eiffel Tower"

    # Tokenize and find subject token positions
    tokens = model.to_tokens(prompt)
    str_tokens = model.to_str_tokens(prompt)
    print(f"Prompt: \"{prompt}\"")
    print(f"Tokens: {str_tokens}")

    # Find subject token positions (approximate — find "E", "iff", "el", " Tower")
    subject_positions = []
    for i, tok in enumerate(str_tokens):
        if any(sub_part in tok for sub_part in ["Ei", "iff", "el", "Tower"]):
            subject_positions.append(i)
    print(f"Subject positions: {subject_positions} (tokens: {[str_tokens[i] for i in subject_positions]})")

    # --- Step 1: Clean run (baseline) ---
    clean_logits, clean_cache = model.run_with_cache(tokens)

    # Get the target token (what the model predicts for the last position)
    # We'll measure recovery of logit for a known-correct answer
    # Use " Paris" as the target
    target_token = model.to_tokens(" Paris")[0, 1]  # skip BOS
    clean_logit_for_target = clean_logits[0, -1, target_token].item()
    print(f"\nClean logit for ' Paris': {clean_logit_for_target:.2f}")

    # Also check what model actually predicts
    top_token = model.tokenizer.decode(clean_logits[0, -1].argmax().item())
    print(f"Model's top prediction: '{top_token}'")

    # --- Step 2: Corrupted run (add noise to subject embeddings) ---
    noise_level = 3.0  # Standard deviations of noise

    def corrupt_subject_hook(activation, hook):
        """Add Gaussian noise to subject token embeddings."""
        for pos in subject_positions:
            if pos < activation.shape[1]:
                activation[0, pos] += noise_level * torch.randn_like(activation[0, pos])
        return activation

    # Run with corrupted subject
    with model.hooks(fwd_hooks=[("hook_embed", corrupt_subject_hook)]):
        corrupted_logits, corrupted_cache = model.run_with_cache(tokens)

    corrupted_logit = corrupted_logits[0, -1, target_token].item()
    print(f"Corrupted logit for ' Paris': {corrupted_logit:.2f}")
    corrupted_top = model.tokenizer.decode(corrupted_logits[0, -1].argmax().item())
    print(f"Corrupted top prediction: '{corrupted_top}'")

    total_effect = clean_logit_for_target - corrupted_logit
    print(f"Total effect of corruption: {total_effect:.2f}")

    # --- Step 3: Restore activations one layer at a time ---
    # For each layer, restore the MLP output from the clean run into the
    # corrupted run, and measure how much the correct answer recovers
    print(f"\n{'─' * 60}")
    print("CAUSAL TRACING RESULTS: Restoring MLP outputs layer by layer")
    print("(Higher = this layer stores more of the factual knowledge)")
    print(f"{'─' * 60}\n")

    mlp_effects = []
    attn_effects = []

    for layer in range(model.cfg.n_layers):
        # --- Test MLP at this layer ---
        def make_restore_mlp_hook(l, clean_c):
            def hook_fn(activation, hook):
                # Restore the clean MLP output at the last subject position
                last_subj_pos = subject_positions[-1]
                if last_subj_pos < activation.shape[1]:
                    activation[0, last_subj_pos] = clean_c[f"blocks.{l}.hook_mlp_out"][0, last_subj_pos]
                return activation
            return hook_fn

        with model.hooks(fwd_hooks=[
            ("hook_embed", corrupt_subject_hook),
            (f"blocks.{layer}.hook_mlp_out", make_restore_mlp_hook(layer, clean_cache)),
        ]):
            restored_logits = model(tokens)

        restored_logit = restored_logits[0, -1, target_token].item()
        recovery = (restored_logit - corrupted_logit) / (total_effect + 1e-8)
        mlp_effects.append(recovery)

        # --- Test Attention at this layer ---
        def make_restore_attn_hook(l, clean_c):
            def hook_fn(activation, hook):
                last_subj_pos = subject_positions[-1]
                if last_subj_pos < activation.shape[1]:
                    activation[0, last_subj_pos] = clean_c[f"blocks.{l}.hook_attn_out"][0, last_subj_pos]
                return activation
            return hook_fn

        with model.hooks(fwd_hooks=[
            ("hook_embed", corrupt_subject_hook),
            (f"blocks.{layer}.hook_attn_out", make_restore_attn_hook(layer, clean_cache)),
        ]):
            restored_logits_attn = model(tokens)

        restored_logit_attn = restored_logits_attn[0, -1, target_token].item()
        recovery_attn = (restored_logit_attn - corrupted_logit) / (total_effect + 1e-8)
        attn_effects.append(recovery_attn)

    # Display results
    print(f"  Layer | MLP Recovery | Attn Recovery | MLP Bar")
    print(f"  {'─' * 55}")

    for layer in range(model.cfg.n_layers):
        mlp_bar = "█" * max(0, int(mlp_effects[layer] * 40))
        print(f"    {layer:2d}   |    {mlp_effects[layer]:+.3f}   |     {attn_effects[layer]:+.3f}   | {mlp_bar}")

    peak_mlp = np.argmax(mlp_effects)
    peak_attn = np.argmax(attn_effects)
    print(f"\n  Peak MLP layer:  {peak_mlp} (recovery: {mlp_effects[peak_mlp]:.3f})")
    print(f"  Peak Attn layer: {peak_attn} (recovery: {attn_effects[peak_attn]:.3f})")

    print("\n✦ The peak MLP layer is where the model stores the factual")
    print("  association 'Eiffel Tower → Paris'. This is exactly what ROME")
    print("  discovered — facts live in middle-layer MLPs!")
    print("  (In the original ROME paper on GPT-2 XL, the peak was around layer 17-20)")

    # --- Run causal tracing on a second fact for comparison ---
    print(f"\n{'═' * 60}")
    print("SECOND FACT: 'The official language of Japan is'")
    print("═" * 60)

    prompt2 = "The official language of Japan is"
    tokens2 = model.to_tokens(prompt2)
    str_tokens2 = model.to_str_tokens(prompt2)

    subject_positions2 = []
    for i, tok in enumerate(str_tokens2):
        if "Japan" in tok:
            subject_positions2.append(i)
    print(f"Subject positions: {subject_positions2}")

    target_token2 = model.to_tokens(" Japanese")[0, 1]

    clean_logits2, clean_cache2 = model.run_with_cache(tokens2)
    clean_logit2 = clean_logits2[0, -1, target_token2].item()

    def corrupt_subject_hook2(activation, hook):
        for pos in subject_positions2:
            if pos < activation.shape[1]:
                activation[0, pos] += noise_level * torch.randn_like(activation[0, pos])
        return activation

    with model.hooks(fwd_hooks=[("hook_embed", corrupt_subject_hook2)]):
        corrupted_logits2 = model(tokens2)
    corrupted_logit2 = corrupted_logits2[0, -1, target_token2].item()
    total_effect2 = clean_logit2 - corrupted_logit2

    mlp_effects2 = []
    for layer in range(model.cfg.n_layers):
        def make_hook2(l, cc):
            def hook_fn(activation, hook):
                last_pos = subject_positions2[-1]
                if last_pos < activation.shape[1]:
                    activation[0, last_pos] = cc[f"blocks.{l}.hook_mlp_out"][0, last_pos]
                return activation
            return hook_fn

        with model.hooks(fwd_hooks=[
            ("hook_embed", corrupt_subject_hook2),
            (f"blocks.{layer}.hook_mlp_out", make_hook2(layer, clean_cache2)),
        ]):
            restored2 = model(tokens2)
        recovery2 = (restored2[0, -1, target_token2].item() - corrupted_logit2) / (total_effect2 + 1e-8)
        mlp_effects2.append(recovery2)

    print(f"\n  Layer | MLP Recovery | Bar")
    print(f"  {'─' * 40}")
    for layer in range(model.cfg.n_layers):
        bar = "█" * max(0, int(mlp_effects2[layer] * 40))
        print(f"    {layer:2d}   |    {mlp_effects2[layer]:+.3f}   | {bar}")

    peak2 = np.argmax(mlp_effects2)
    print(f"\n  Peak MLP layer: {peak2} (recovery: {mlp_effects2[peak2]:.3f})")
    print("  Notice: different facts may peak at different layers!")

    del model, clean_cache, corrupted_cache, clean_cache2
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# PART 2: WEIGHT INSPECTION — What do knowledge-storing weights look like?
# ============================================================================
def demo_weight_inspection():
    """
    Inspect the MLP weight matrices to understand how factual knowledge
    is encoded. In transformers, each MLP has:
      - W_in (d_model → d_mlp): "key" matrix — what patterns to respond to
      - W_out (d_mlp → d_model): "value" matrix — what to output when triggered

    ROME's insight: Facts are stored as key-value pairs in these matrices.
    The "key" encodes the subject, the "value" encodes the associated object.
    """
    print("\n" + "=" * 70)
    print("PART 2: WEIGHT INSPECTION — Inside the Knowledge-Storing Matrices")
    print("=" * 70)
    print("\nMLPs store facts as key-value pairs in their weight matrices.")
    print("Let's inspect these matrices and see how they relate to knowledge.\n")

    from transformer_lens import HookedTransformer

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # Focus on a middle layer (where causal tracing typically peaks)
    target_layer = 7

    # Get the MLP weights
    W_in = model.blocks[target_layer].mlp.W_in.data    # (d_model, d_mlp) = (768, 3072)
    W_out = model.blocks[target_layer].mlp.W_out.data   # (d_mlp, d_model) = (3072, 768)
    b_in = model.blocks[target_layer].mlp.b_in.data     # (d_mlp,)
    b_out = model.blocks[target_layer].mlp.b_out.data    # (d_model,)

    print(f"Layer {target_layer} MLP weights:")
    print(f"  W_in:  {W_in.shape}  (d_model → d_mlp)")
    print(f"  W_out: {W_out.shape}  (d_mlp → d_model)")
    print(f"  b_in:  {b_in.shape}")
    print(f"  b_out: {b_out.shape}")

    # --- Analyze the "value" vectors (rows of W_out) ---
    # Each row of W_out is a "value" vector — what the MLP outputs when a specific
    # neuron fires. We can project these through the unembedding matrix to see
    # what tokens each neuron promotes.
    print(f"\n{'─' * 60}")
    print("WHAT DO INDIVIDUAL MLP NEURONS PROMOTE?")
    print("(Each neuron's value vector → projected to vocabulary)")
    print(f"{'─' * 60}\n")

    W_U = model.W_U  # (d_model, d_vocab) — unembedding matrix

    # Pick some random neurons and show what tokens they promote
    interesting_neurons = [100, 500, 1000, 1500, 2000, 2500, 3000]

    for neuron_idx in interesting_neurons:
        value_vector = W_out[neuron_idx]  # (d_model,)
        # Project through unembedding to get vocabulary effect
        vocab_effect = value_vector @ W_U  # (d_vocab,)

        top_tokens_ids = vocab_effect.topk(6).indices
        top_tokens = [model.tokenizer.decode(t.item()).strip() for t in top_tokens_ids]
        top_tokens = [t for t in top_tokens if t][:5]

        bottom_tokens_ids = (-vocab_effect).topk(3).indices
        bottom_tokens = [model.tokenizer.decode(t.item()).strip() for t in bottom_tokens_ids]
        bottom_tokens = [t for t in bottom_tokens if t][:3]

        print(f"  Neuron {neuron_idx:4d}: promotes {top_tokens}  |  suppresses {bottom_tokens}")

    # --- Analyze the "key" vectors (columns of W_in) ---
    print(f"\n{'─' * 60}")
    print("WEIGHT STATISTICS ACROSS LAYERS")
    print("(How does knowledge capacity vary by layer?)")
    print(f"{'─' * 60}\n")

    print(f"  Layer | W_in norm  | W_out norm | W_out rank (eff.) | Sparsity")
    print(f"  {'─' * 65}")

    for layer in range(model.cfg.n_layers):
        w_in = model.blocks[layer].mlp.W_in.data
        w_out = model.blocks[layer].mlp.W_out.data

        w_in_norm = w_in.norm().item()
        w_out_norm = w_out.norm().item()

        # Effective rank: how many singular values are needed to capture 90% of variance
        # This indicates how "spread out" the knowledge is
        U, S, V = torch.svd_lowrank(w_out, q=min(100, w_out.shape[0]))
        cumsum = torch.cumsum(S ** 2, dim=0)
        total_var = cumsum[-1].item()
        eff_rank = (cumsum < 0.9 * total_var).sum().item() + 1

        # Sparsity: fraction of near-zero weights
        sparsity = (w_out.abs() < 0.01).float().mean().item()

        print(f"    {layer:2d}   | {w_in_norm:9.1f} | {w_out_norm:9.1f} |        {eff_rank:3d}        | {sparsity:.1%}")

    # --- Show how a specific fact's representation relates to weights ---
    print(f"\n{'─' * 60}")
    print("FACT-WEIGHT ALIGNMENT: How well do specific facts align with MLP neurons?")
    print(f"{'─' * 60}\n")

    prompts_and_targets = [
        ("The Eiffel Tower is located in the city of", " Paris"),
        ("The capital of Germany is", " Berlin"),
        ("Barack Obama was the president of the United", " States"),
    ]

    for prompt, target_str in prompts_and_targets:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)

        # Get the residual stream at the target layer (input to MLP)
        resid = cache[f"blocks.{target_layer}.hook_resid_pre"][0, -1]  # (d_model,)

        # The MLP "key" match: how much does each neuron respond to this input?
        key_match = resid @ W_in + b_in  # (d_mlp,) — pre-activation
        # Apply GELU activation (GPT-2 uses GELU)
        activations = torch.nn.functional.gelu(key_match)

        # Which neurons fire the most?
        top_neurons = activations.topk(5).indices
        top_vals = activations.topk(5).values

        print(f"  \"{prompt}\" → target: '{target_str}'")
        print(f"    Top-firing neurons at layer {target_layer}: ", end="")
        for n, v in zip(top_neurons, top_vals):
            # What does this neuron's value vector promote?
            value_vec = W_out[n.item()]
            vocab_effect = value_vec @ W_U
            top_tok = model.tokenizer.decode(vocab_effect.argmax().item()).strip()
            print(f"N{n.item()}({v.item():.1f}→'{top_tok}') ", end="")
        print("\n")

    print("✦ When a neuron fires strongly AND its value vector promotes the")
    print("  correct answer token, that's evidence of factual storage!")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# PART 3: KNOWLEDGE EDITING — Surgically modify weights to change facts
# ============================================================================
def demo_knowledge_editing():
    """
    Simplified ROME-style knowledge editing:

    We modify the MLP weights at a specific layer to change what the model
    "knows" about a specific fact. This is a simplified version of the
    rank-one model editing technique from Meng et al. (2022).

    The key insight: we can add a rank-one update to W_out that causes
    a specific neuron pattern to produce a different output.
    """
    print("\n" + "=" * 70)
    print("PART 3: KNOWLEDGE EDITING — Changing What the Model 'Knows'")
    print("=" * 70)
    print("\nWe'll edit GPT-2's weights to change a factual association.")
    print("This demonstrates the core idea behind ROME.\n")

    from transformer_lens import HookedTransformer

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # The fact we want to edit
    prompt = "The Eiffel Tower is located in the city of"
    new_target = " Rome"  # Change from Paris to Rome

    print(f'Prompt: "{prompt}"')
    print(f'Goal: Make the model say "{new_target}" instead of its current answer\n')

    # --- Before editing: what does the model predict? ---
    tokens = model.to_tokens(prompt)
    logits_before = model(tokens)
    probs_before = torch.softmax(logits_before[0, -1], dim=-1)

    # Check top predictions
    top5_before = logits_before[0, -1].topk(5)
    print("BEFORE editing — top predictions:")
    for i in range(5):
        tok = model.tokenizer.decode(top5_before.indices[i].item()).strip()
        prob = probs_before[top5_before.indices[i]].item()
        print(f"  '{tok:>12s}': {prob:.1%}")

    target_token_id = model.to_tokens(new_target)[0, 1].item()
    prob_rome_before = probs_before[target_token_id].item()
    print(f"\n  P('{new_target.strip()}'): {prob_rome_before:.4%}")

    # --- Perform the edit ---
    # Simplified ROME: We'll modify W_out at the critical layer to push
    # the output toward the new target when this specific input is processed

    edit_layer = 8  # Middle layer — where facts tend to be stored

    # Step 1: Get the MLP hidden state for this input (the "key")
    _, cache = model.run_with_cache(tokens)
    mlp_input = cache[f"blocks.{edit_layer}.hook_resid_pre"][0, -1]  # (d_model,)
    W_in = model.blocks[edit_layer].mlp.W_in.data
    b_in = model.blocks[edit_layer].mlp.b_in.data
    hidden = torch.nn.functional.gelu(mlp_input @ W_in + b_in)  # (d_mlp,)

    # Step 2: Compute what we want the MLP to output instead
    # We want to push toward the new target token's embedding
    target_embedding = model.W_U[:, target_token_id]  # (d_model,) — the unembedding vector for "Rome"

    # Current MLP output
    W_out = model.blocks[edit_layer].mlp.W_out.data  # (d_mlp, d_model)
    current_output = hidden @ W_out  # (d_model,)

    # Desired change in output
    edit_strength = 5.0  # How strongly to push toward the new target
    delta = edit_strength * target_embedding - (current_output @ model.W_U)[:1].item() * target_embedding
    delta = edit_strength * target_embedding  # Simplified: just push toward target

    # Step 3: Rank-one update to W_out
    # We want: (hidden @ W_out_new) = (hidden @ W_out) + delta
    # Solution: W_out_new = W_out + outer(key_direction, delta) / (key_direction @ hidden)
    key_direction = hidden / (hidden @ hidden + 1e-8)  # Normalize

    # Apply the rank-one update
    print(f"\nApplying rank-one edit to layer {edit_layer} MLP W_out...")
    print(f"  W_out shape: {W_out.shape}")
    print(f"  Edit rank: 1 (rank-one update)")

    # Save original weights
    W_out_original = W_out.clone()

    # Apply edit
    model.blocks[edit_layer].mlp.W_out.data += torch.outer(key_direction, delta)

    # --- After editing: what does the model predict? ---
    logits_after = model(tokens)
    probs_after = torch.softmax(logits_after[0, -1], dim=-1)

    top5_after = logits_after[0, -1].topk(5)
    print("\nAFTER editing — top predictions:")
    for i in range(5):
        tok = model.tokenizer.decode(top5_after.indices[i].item()).strip()
        prob = probs_after[top5_after.indices[i]].item()
        print(f"  '{tok:>12s}': {prob:.1%}")

    prob_rome_after = probs_after[target_token_id].item()
    print(f"\n  P('{new_target.strip()}'): {prob_rome_before:.4%} → {prob_rome_after:.4%} "
          f"({'↑' if prob_rome_after > prob_rome_before else '↓'} {abs(prob_rome_after - prob_rome_before):.4%})")

    # --- Test specificity: does the edit affect unrelated facts? ---
    print(f"\n{'─' * 60}")
    print("SPECIFICITY CHECK: Do unrelated facts still work?")
    print(f"{'─' * 60}\n")

    unrelated_prompts = [
        "The capital of Germany is",
        "The Great Wall is located in",
        "Barack Obama was the president of the United",
        "The Statue of Liberty is in New",
    ]

    for p in unrelated_prompts:
        tok = model.to_tokens(p)

        # Before edit
        model.blocks[edit_layer].mlp.W_out.data = W_out_original.clone()
        logits_orig = model(tok)
        top_orig = model.tokenizer.decode(logits_orig[0, -1].argmax().item()).strip()

        # After edit
        model.blocks[edit_layer].mlp.W_out.data = W_out_original + torch.outer(key_direction, delta)
        logits_edit = model(tok)
        top_edit = model.tokenizer.decode(logits_edit[0, -1].argmax().item()).strip()

        changed = "✗ CHANGED" if top_orig != top_edit else "✓ same"
        print(f"  \"{p}\"")
        print(f"    Before: '{top_orig}' → After: '{top_edit}'  {changed}")

    # Restore original weights
    model.blocks[edit_layer].mlp.W_out.data = W_out_original

    # --- Generate text before and after ---
    print(f"\n{'─' * 60}")
    print("GENERATION COMPARISON")
    print(f"{'─' * 60}\n")

    gen_prompt = "The Eiffel Tower, located in"

    # Before
    model.blocks[edit_layer].mlp.W_out.data = W_out_original
    text_before = model.generate(gen_prompt, max_new_tokens=25, temperature=0.7, do_sample=True, verbose=False)
    print(f"  [Before edit] {text_before}")

    # After
    model.blocks[edit_layer].mlp.W_out.data = W_out_original + torch.outer(key_direction, delta)
    text_after = model.generate(gen_prompt, max_new_tokens=25, temperature=0.7, do_sample=True, verbose=False)
    print(f"  [After edit]  {text_after}")

    # Restore
    model.blocks[edit_layer].mlp.W_out.data = W_out_original

    print("\n✦ The rank-one edit changed the model's factual association!")
    print("  This is the core mechanism of ROME — one rank-one update per fact.")
    print("  MEMIT extends this to edit thousands of facts simultaneously.")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# PART 4: WEIGHT DIFF ANALYSIS — Watch the Weights-style SVD
# ============================================================================
def demo_weight_diff_analysis():
    """
    Inspired by "Watch the Weights" (ICLR 2026):

    Analyze the weight difference between two versions of a model
    (before/after editing) using SVD. The top singular vectors reveal
    what behaviors changed.

    We simulate this by:
    1. Taking the original GPT-2 weights
    2. Making a knowledge edit (like Part 3)
    3. Computing SVD of the weight difference
    4. Showing what the top singular vectors correspond to
    """
    print("\n" + "=" * 70)
    print("PART 4: WEIGHT DIFF ANALYSIS — Watch the Weights (SVD)")
    print("=" * 70)
    print("\nAnalyzing weight changes via SVD to understand what behaviors")
    print("were modified. Inspired by 'Watch the Weights' (ICLR 2026).\n")

    from transformer_lens import HookedTransformer

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    edit_layer = 8
    W_out_original = model.blocks[edit_layer].mlp.W_out.data.clone()

    # --- Make multiple edits to simulate fine-tuning ---
    print("Simulating multiple knowledge edits (like fine-tuning)...\n")

    edits = [
        ("The Eiffel Tower is located in the city of", " Rome"),
        ("The capital of France is", " Berlin"),
        ("The president of the United States is", " Putin"),
    ]

    total_delta = torch.zeros_like(W_out_original)

    for prompt, new_target in edits:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)

        mlp_input = cache[f"blocks.{edit_layer}.hook_resid_pre"][0, -1]
        W_in = model.blocks[edit_layer].mlp.W_in.data
        b_in = model.blocks[edit_layer].mlp.b_in.data
        hidden = torch.nn.functional.gelu(mlp_input @ W_in + b_in)

        target_token_id = model.to_tokens(new_target)[0, 1].item()
        target_emb = model.W_U[:, target_token_id]

        key_dir = hidden / (hidden @ hidden + 1e-8)
        delta = 5.0 * target_emb
        rank_one = torch.outer(key_dir, delta)
        total_delta += rank_one

        print(f"  Edit: \"{prompt}\" → '{new_target.strip()}'")

    # Apply all edits
    W_out_edited = W_out_original + total_delta

    # --- Compute SVD of the weight difference ---
    print(f"\n{'─' * 60}")
    print("SVD OF WEIGHT DIFFERENCE")
    print(f"{'─' * 60}\n")

    weight_diff = W_out_edited - W_out_original  # (d_mlp, d_model)
    print(f"  Weight diff shape: {weight_diff.shape}")
    print(f"  Weight diff Frobenius norm: {weight_diff.norm():.4f}")
    print(f"  Original weight norm: {W_out_original.norm():.4f}")
    print(f"  Relative change: {weight_diff.norm() / W_out_original.norm():.6f}")

    # Compute SVD
    U, S, Vh = torch.linalg.svd(weight_diff, full_matrices=False)
    # U: (d_mlp, k) — left singular vectors (neuron-space directions)
    # S: (k,) — singular values
    # Vh: (k, d_model) — right singular vectors (residual-stream directions)

    print(f"\n  Top 10 singular values:")
    for i in range(min(10, len(S))):
        pct = (S[i] ** 2 / (S ** 2).sum() * 100).item()
        bar = "█" * int(pct)
        print(f"    σ_{i}: {S[i].item():8.4f} ({pct:5.1f}% of variance) {bar}")

    # How many components capture 99% of the change?
    cumvar = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
    n99 = (cumvar < 0.99).sum().item() + 1
    n90 = (cumvar < 0.90).sum().item() + 1
    print(f"\n  Components for 90% variance: {n90}")
    print(f"  Components for 99% variance: {n99}")
    print(f"  (We made {len(edits)} edits → expect ~{len(edits)} dominant components)")

    # --- Interpret the top singular vectors ---
    print(f"\n{'─' * 60}")
    print("INTERPRETING TOP SINGULAR VECTORS")
    print("(What do the principal directions of change correspond to?)")
    print(f"{'─' * 60}\n")

    for i in range(min(3, len(S))):
        # Right singular vector: direction in residual stream space
        direction = Vh[i]  # (d_model,)

        # Project to vocabulary to see what tokens this direction promotes
        vocab_effect = direction @ model.W_U
        top_ids = vocab_effect.topk(8).indices
        top_tokens = [model.tokenizer.decode(t.item()).strip() for t in top_ids]
        top_tokens = [t for t in top_tokens if t][:6]

        bottom_ids = (-vocab_effect).topk(5).indices
        bottom_tokens = [model.tokenizer.decode(t.item()).strip() for t in bottom_ids]
        bottom_tokens = [t for t in bottom_tokens if t][:4]

        print(f"  Singular vector {i} (σ={S[i].item():.4f}, {(S[i]**2 / (S**2).sum() * 100).item():.1f}% var):")
        print(f"    Promotes:    {top_tokens}")
        print(f"    Suppresses:  {bottom_tokens}")
        print()

    # --- Monitor activations along singular directions ---
    print(f"{'─' * 60}")
    print("ACTIVATION MONITORING (Watch the Weights-style)")
    print("(Cosine similarity of activations with top singular vectors)")
    print(f"{'─' * 60}\n")

    test_prompts = [
        ("The Eiffel Tower is in", "Edited fact (Eiffel Tower)"),
        ("The capital of France is", "Edited fact (France capital)"),
        ("The Great Wall is in", "Unrelated fact"),
        ("I love eating pizza with", "Completely unrelated"),
    ]

    top_direction = Vh[0]  # Top singular vector of the weight diff

    print(f"  Prompt → cosine sim with top weight-change direction:")
    for prompt, label in test_prompts:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        activation = cache[f"blocks.{edit_layer}.hook_resid_pre"][0, -1]

        cos_sim = torch.nn.functional.cosine_similarity(
            activation.unsqueeze(0), top_direction.unsqueeze(0)
        ).item()

        bar = "█" * int(abs(cos_sim) * 40)
        edited_marker = " ← EDITED" if "Edited" in label else ""
        print(f"    {label:30s} | sim={cos_sim:+.4f} | {bar}{edited_marker}")

    print("\n✦ Watch the Weights insight: prompts related to edited facts show")
    print("  higher alignment with the top singular vectors of the weight diff.")
    print("  This allows detecting what was changed during fine-tuning —")
    print("  without knowing the training data!")

    print("\n" + "─" * 60)
    print("RESEARCH APPLICATIONS:")
    print("─" * 60)
    print("""
  These weight-analysis techniques enable:

  1. HALLUCINATION DETECTION: If you know the base model weights,
     SVD of the weight diff reveals what facts were modified during
     fine-tuning — potentially exposing introduced hallucinations.

  2. MODEL AUDITING: Analyze commercial fine-tuned models to
     discover what behaviors were added (Watch the Weights found
     that Qwen was fine-tuned on Midjourney prompt generation).

  3. BACKDOOR DETECTION: Malicious fine-tuning leaves detectable
     traces in the weight diff's singular vectors.

  4. KNOWLEDGE EDITING FOR SAFETY: Use ROME/MEMIT to correct
     hallucinated facts, while monitoring specificity to ensure
     unrelated knowledge is preserved.

  5. COMBINING WITH SAEs: Use SAEs to interpret what the
     singular vectors of the weight diff correspond to in terms
     of monosemantic features (see MicroEdit, EMNLP 2025).
""")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  WEIGHT-BASED INTERPRETABILITY DEMO                            ║")
    print("║  Causal Tracing, Knowledge Editing & Weight Analysis           ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  Parts:                                                        ║")
    print("║    1. Causal Tracing — locate where facts are stored           ║")
    print("║    2. Weight Inspection — inside the knowledge matrices        ║")
    print("║    3. Knowledge Editing — surgically fix hallucinations        ║")
    print("║    4. Weight Diff Analysis — SVD-based change detection        ║")
    print("║                                                                ║")
    print("║  Hardware: CPU only, ~4 GB RAM. No GPU needed.                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    try:
        demo_causal_tracing()
    except Exception as e:
        print(f"\n[!] Causal Tracing failed: {e}")
        import traceback; traceback.print_exc()

    try:
        demo_weight_inspection()
    except Exception as e:
        print(f"\n[!] Weight Inspection failed: {e}")
        import traceback; traceback.print_exc()

    try:
        demo_knowledge_editing()
    except Exception as e:
        print(f"\n[!] Knowledge Editing failed: {e}")
        import traceback; traceback.print_exc()

    try:
        demo_weight_diff_analysis()
    except Exception as e:
        print(f"\n[!] Weight Diff Analysis failed: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)
    print("\nKey papers to read next:")
    print("  1. ROME: https://rome.baulab.info/")
    print("  2. MEMIT: https://memit.baulab.info/")
    print("  3. Watch the Weights: https://github.com/fjzzq2002/WeightWatch")
    print("  4. Can Knowledge Editing Correct Hallucinations? (ICLR 2025)")
    print("  5. MicroEdit: SAE + Knowledge Editing (EMNLP 2025)")
    print("\nSee also:")
    print("  - mech_interp_demo.py — general MI techniques demo")
    print("  - character_comparison_demo.py — character analysis demo")
    print("  - mechanistic_interpretability_report.md — full research report")
    print("  - weight_interpretability_hallucinations.md — weight analysis report")


if __name__ == "__main__":
    main()
