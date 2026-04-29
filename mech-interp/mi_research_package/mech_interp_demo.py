#!/usr/bin/env python3
"""
Mechanistic Interpretability Demo — Lightweight, CPU-friendly
=============================================================

This script demonstrates three key MI techniques on GPT-2 Small (124M params).
Everything runs on CPU with ~4 GB RAM. No GPU required.

Techniques demonstrated:
  1. Logit Lens — see what the model "thinks" at each layer
  2. SAE Feature Analysis — decompose activations into interpretable features
  3. Activation Steering — shift model behavior by adding a direction vector

Requirements:
  pip install transformer-lens sae-lens torch numpy

Author: Research demo for Mechanistic Interpretability study
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np

DEVICE = "cpu"  # Change to "cuda" if you have a GPU

# ============================================================================
# PART 1: LOGIT LENS
# ============================================================================
def demo_logit_lens():
    """
    Logit Lens: Apply the unembedding matrix to intermediate layers to see
    what the model would predict if it stopped processing at each layer.
    
    This reveals how the model's "belief" about the next token evolves
    layer by layer through the network.
    """
    print("\n" + "=" * 70)
    print("PART 1: LOGIT LENS")
    print("=" * 70)
    print("\nThe Logit Lens shows what GPT-2 would predict at each layer.")
    print("We watch the model's 'belief' evolve from input to output.\n")

    from transformer_lens import HookedTransformer

    # Load GPT-2 Small (124M params — fits easily on CPU)
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model}d model\n")

    # Choose a prompt where we know the expected completion
    prompt = "The Eiffel Tower is located in the city of"
    print(f'Prompt: "{prompt}"')

    # Run the model and cache all intermediate activations
    logits, cache = model.run_with_cache(prompt)

    # Get the final prediction for reference
    final_token = model.tokenizer.decode(logits[0, -1].argmax().item())
    final_prob = torch.softmax(logits[0, -1], dim=-1).max().item()
    print(f"Final prediction: '{final_token}' (prob: {final_prob:.1%})\n")

    # Apply the logit lens at each layer
    print("Layer-by-layer predictions (logit lens):")
    print("-" * 50)

    for layer in range(model.cfg.n_layers):
        # Get the residual stream at this layer (after the layer's computation)
        residual = cache[f"blocks.{layer}.hook_resid_post"][0, -1]  # last token position

        # Apply layer norm + unembedding (the "logit lens")
        normalized = model.ln_final(residual)
        layer_logits = normalized @ model.W_U

        # Get the top prediction
        probs = torch.softmax(layer_logits, dim=-1)
        top_token_id = layer_logits.argmax().item()
        top_token = model.tokenizer.decode(top_token_id)
        top_prob = probs[top_token_id].item()

        # Also get 2nd and 3rd predictions for context
        top3_ids = layer_logits.topk(3).indices
        top3_tokens = [model.tokenizer.decode(t.item()) for t in top3_ids]

        bar = "█" * int(top_prob * 30)
        print(f"  Layer {layer:2d}: '{top_token:>12s}' ({top_prob:5.1%}) {bar}")
        if layer == 0 or layer == model.cfg.n_layers - 1:
            print(f"           Top 3: {top3_tokens}")

    print("\n✦ Notice how early layers are uncertain/wrong, and the correct")
    print("  answer emerges in the middle-to-late layers.")

    del model, cache
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# PART 2: SAE FEATURE ANALYSIS
# ============================================================================
def demo_sae_features():
    """
    Sparse Autoencoder (SAE) Feature Analysis:
    
    Load a pre-trained SAE for GPT-2 Small and decompose the model's
    activations into interpretable features. Each feature ideally
    represents one human-understandable concept.
    
    This demonstrates:
    - Loading a pre-trained SAE from SAELens
    - Running the SAE on model activations
    - Identifying which features activate for different prompts
    - Examining the top vocabulary items associated with each feature
    """
    print("\n" + "=" * 70)
    print("PART 2: SAE FEATURE ANALYSIS")
    print("=" * 70)
    print("\nSparse Autoencoders decompose polysemantic activations into")
    print("monosemantic features. Each feature = one interpretable concept.\n")

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    # Load model
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # Load a pre-trained SAE for layer 8 residual stream
    # This SAE was trained by Joseph Bloom and is freely available
    print("Loading pre-trained SAE for layer 8...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=DEVICE,
    )
    print(f"SAE loaded: {sae.cfg.d_in}d input → {sae.cfg.d_sae}d features\n")

    # --- Analyze different prompts ---
    prompts = [
        "The capital of France is Paris, a beautiful city",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1)",
        "The patient was diagnosed with acute myocardial infarction",
        "Once upon a time, in a land far far away, there lived",
    ]

    for prompt in prompts:
        print(f'Prompt: "{prompt[:60]}..."' if len(prompt) > 60 else f'Prompt: "{prompt}"')

        # Run model and get activations at layer 8
        _, cache = model.run_with_cache(prompt, prepend_bos=True)
        activations = cache["blocks.8.hook_resid_pre"]

        # Run activations through the SAE
        feature_acts = sae.encode(activations)

        # Analyze the last token position (most relevant for next-token prediction)
        last_pos_features = feature_acts[0, -1]  # shape: (d_sae,)

        # Find active features (non-zero activations)
        active_mask = last_pos_features > 0
        active_indices = active_mask.nonzero().squeeze(-1)
        active_values = last_pos_features[active_mask]

        # Sort by activation strength
        sorted_order = active_values.argsort(descending=True)
        active_indices = active_indices[sorted_order]
        active_values = active_values[sorted_order]

        n_active = len(active_indices)
        print(f"  Active features (L0): {n_active} out of {sae.cfg.d_sae}")

        # Show top 5 most active features with their logit effects
        print(f"  Top 5 features by activation strength:")
        W_dec = sae.W_dec.data  # decoder weights: (d_sae, d_model)

        for i in range(min(5, n_active)):
            feat_idx = active_indices[i].item()
            feat_val = active_values[i].item()

            # What tokens does this feature promote? (via the decoder → unembedding)
            feat_dir = W_dec[feat_idx]  # direction in residual stream
            logit_effect = feat_dir @ model.W_U  # project to vocabulary
            top_tokens = logit_effect.topk(8).indices
            token_strs = [model.tokenizer.decode(t.item()).strip() for t in top_tokens]
            # Filter out empty strings
            token_strs = [t for t in token_strs if t][:5]

            print(f"    Feature {feat_idx:5d} (act={feat_val:.2f}): "
                  f"promotes tokens → {token_strs}")

        print()

    # --- Compare features across related prompts ---
    print("=" * 50)
    print("FEATURE COMPARISON: Same concept, different context")
    print("=" * 50)

    pairs = [
        ("The weather in Paris is", "The weather in London is"),
        ("She wrote a Python script", "He wrote a Python script"),
    ]

    for prompt_a, prompt_b in pairs:
        print(f'\n  A: "{prompt_a}"')
        print(f'  B: "{prompt_b}"')

        _, cache_a = model.run_with_cache(prompt_a, prepend_bos=True)
        _, cache_b = model.run_with_cache(prompt_b, prepend_bos=True)

        feats_a = sae.encode(cache_a["blocks.8.hook_resid_pre"])[0, -1]
        feats_b = sae.encode(cache_b["blocks.8.hook_resid_pre"])[0, -1]

        # Find shared active features
        active_a = set((feats_a > 0).nonzero().squeeze(-1).tolist())
        active_b = set((feats_b > 0).nonzero().squeeze(-1).tolist())
        shared = active_a & active_b
        only_a = active_a - active_b
        only_b = active_b - active_a

        print(f"  Shared features: {len(shared)} | Only in A: {len(only_a)} | Only in B: {len(only_b)}")

        if shared:
            # Show top shared features
            shared_list = sorted(shared, key=lambda f: feats_a[f].item(), reverse=True)[:3]
            print(f"  Top shared features: {shared_list}")

    print("\n✦ Shared features between related prompts reveal common concepts")
    print("  the model is tracking (e.g., 'weather', 'city', 'programming').")

    del model, sae, cache
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# PART 3: ACTIVATION STEERING (Representation Engineering)
# ============================================================================
def demo_activation_steering():
    """
    Activation Steering: Find a direction in activation space that
    corresponds to a concept, then add it during generation to steer
    the model's behavior.
    
    We'll find a "formal writing" direction by contrasting formal
    and casual text, then use it to make GPT-2 write more formally.
    """
    print("\n" + "=" * 70)
    print("PART 3: ACTIVATION STEERING")
    print("=" * 70)
    print("\nActivation steering finds a 'concept direction' in the model's")
    print("internal space and adds it during generation to shift behavior.\n")

    from transformer_lens import HookedTransformer

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # --- Step 1: Find the "formal" direction ---
    # We use contrastive pairs: formal vs. casual sentences
    formal_prompts = [
        "I would like to formally request your assistance with this matter.",
        "We are writing to inform you of the following policy changes.",
        "The committee has reached a unanimous decision regarding the proposal.",
        "It is with great pleasure that we announce the following results.",
        "Pursuant to our previous correspondence, we wish to clarify.",
        "The aforementioned regulations shall be enacted forthwith.",
        "We respectfully submit this report for your consideration.",
        "The board of directors convened to discuss quarterly performance.",
    ]

    casual_prompts = [
        "hey can you help me out with this thing real quick?",
        "so basically what happened was the rules changed lol",
        "everyone agreed the idea was pretty cool honestly",
        "omg we're so happy to tell you what happened next!",
        "about that email from before, just wanted to clear stuff up",
        "the new rules are gonna start like right away",
        "here's what we found, take a look when you get a chance",
        "the bosses got together to talk about how things are going",
    ]

    print("Computing 'formal writing' direction from contrastive pairs...")

    # Collect activations for formal and casual prompts
    # We use the residual stream at an intermediate layer (layer 6)
    steering_layer = 6

    formal_acts = []
    casual_acts = []

    for prompt in formal_prompts:
        _, cache = model.run_with_cache(prompt, prepend_bos=True)
        # Average over all token positions
        act = cache[f"blocks.{steering_layer}.hook_resid_post"][0].mean(dim=0)
        formal_acts.append(act)

    for prompt in casual_prompts:
        _, cache = model.run_with_cache(prompt, prepend_bos=True)
        act = cache[f"blocks.{steering_layer}.hook_resid_post"][0].mean(dim=0)
        casual_acts.append(act)

    # The "formal direction" = mean(formal) - mean(casual)
    formal_mean = torch.stack(formal_acts).mean(dim=0)
    casual_mean = torch.stack(casual_acts).mean(dim=0)
    formal_direction = formal_mean - casual_mean
    formal_direction = formal_direction / formal_direction.norm()  # normalize

    print(f"Direction computed (layer {steering_layer}, norm=1.0)")

    # --- Step 2: Generate text with and without steering ---
    test_prompts = [
        "The company announced that",
        "Dear Sir, I am writing to",
        "The results of the experiment showed",
    ]

    for test_prompt in test_prompts:
        print(f'\n{"─" * 60}')
        print(f'Prompt: "{test_prompt}"')

        # Normal generation (no steering)
        normal_text = model.generate(
            test_prompt,
            max_new_tokens=30,
            temperature=0.7,
            do_sample=True,
            verbose=False,
        )
        print(f"\n  [Normal]  {normal_text}")

        # Steered generation (add formal direction)
        steering_strength = 15.0  # how much to push toward formal

        def steering_hook(activation, hook):
            # Add the formal direction to the residual stream
            activation[:, :, :] += steering_strength * formal_direction
            return activation

        # Register the hook and generate
        hook_name = f"blocks.{steering_layer}.hook_resid_post"
        with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
            steered_text = model.generate(
                test_prompt,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                verbose=False,
            )
        print(f"  [Steered] {steered_text}")

    print(f'\n{"─" * 60}')
    print("\n✦ The steered text should sound more formal/professional.")
    print("  This works because we're pushing the model's internal state")
    print("  toward the 'formal' region of its representation space.")
    print("\n  You can experiment with:")
    print("  - Different steering strengths (try 5, 10, 20, 30)")
    print("  - Different layers (earlier = more global, later = more specific)")
    print("  - Different concept directions (positive/negative sentiment, etc.)")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   MECHANISTIC INTERPRETABILITY DEMO — GPT-2 Small (CPU-OK)     ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  This demo shows 3 key MI techniques:                          ║")
    print("║    1. Logit Lens — layer-by-layer predictions                  ║")
    print("║    2. SAE Features — decomposing activations into concepts     ║")
    print("║    3. Activation Steering — pushing model behavior             ║")
    print("║                                                                ║")
    print("║  Hardware: CPU only, ~4 GB RAM. No GPU needed.                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # You can run individual parts by commenting out the others
    print("\nNote: First run will download GPT-2 Small (~500 MB) and")
    print("a pre-trained SAE (~200 MB). Subsequent runs use cached files.\n")

    try:
        demo_logit_lens()
    except Exception as e:
        print(f"\n[!] Logit Lens demo failed: {e}")

    try:
        demo_sae_features()
    except Exception as e:
        print(f"\n[!] SAE Features demo failed: {e}")

    try:
        demo_activation_steering()
    except Exception as e:
        print(f"\n[!] Activation Steering demo failed: {e}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Explore features interactively: https://www.neuronpedia.org")
    print("  2. Try TransformerLens tutorials: https://transformerlensorg.github.io/TransformerLens/")
    print("  3. Scale up with nnsight (remote execution on 70B+ models): https://nnsight.net")
    print("  4. Try circuit tracing: https://github.com/decoderesearch/circuit-tracer")
    print("  5. Read the companion report: mechanistic_interpretability_report.md")


if __name__ == "__main__":
    main()
