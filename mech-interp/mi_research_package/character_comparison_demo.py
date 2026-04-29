#!/usr/bin/env python3
"""
Character Comparison Demo — Mechanistic Interpretability
=========================================================

This script compares how GPT-2 Small internally represents different
characters/personas (Harry Potter vs Voldemort) using MI techniques.

Research question: How does a model's internal state differ when it
"pretends" to be different characters? Can we identify which internal
features correspond to heroic vs villainous traits?

Techniques demonstrated:
  1. Logit Lens Comparison — layer-by-layer prediction differences
  2. SAE Feature Contrast — which interpretable features differ?
  3. Activation Distance Analysis — how far apart are the representations?
  4. Character Steering — find the Harry↔Voldemort direction and steer

Requirements:
  pip install transformer-lens sae-lens torch numpy

Hardware: CPU only, ~4 GB RAM. No GPU required.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np

DEVICE = "cpu"  # Change to "cuda" if you have a GPU


# ============================================================================
# PART 1: LOGIT LENS COMPARISON
# ============================================================================
def demo_logit_lens_comparison():
    """
    Compare what the model predicts layer-by-layer for Harry Potter
    vs Voldemort prompts. This reveals at which layer the model
    "decides" on a character-appropriate response.
    """
    print("\n" + "=" * 70)
    print("PART 1: LOGIT LENS — CHARACTER COMPARISON")
    print("=" * 70)
    print("\nComparing layer-by-layer predictions for Harry vs Voldemort.")
    print("We'll see how the model's next-token prediction differs.\n")

    from transformer_lens import HookedTransformer

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # Character prompts — both ask the model to speak as the character
    prompts = {
        "Harry Potter": 'Pretend you are Harry Potter and tell me something Harry would say: "',
        "Voldemort": 'Pretend you are Voldemort and tell me something Voldemort would say: "',
    }

    # Additional character prompts for richer analysis
    extra_prompts = {
        "Harry context": "Harry Potter bravely faced the danger and said",
        "Voldemort context": "Lord Voldemort raised his wand menacingly and said",
        "Harry values": "The things Harry Potter cares about most are",
        "Voldemort values": "The things Lord Voldemort desires most are",
    }

    all_prompts = {**prompts, **extra_prompts}

    for name, prompt in all_prompts.items():
        print(f"\n{'─' * 60}")
        print(f"[{name}]")
        print(f'  Prompt: "{prompt}"')

        logits, cache = model.run_with_cache(prompt)

        # Final prediction
        final_probs = torch.softmax(logits[0, -1], dim=-1)
        top5_ids = logits[0, -1].topk(5).indices
        top5_tokens = [model.tokenizer.decode(t.item()).strip() for t in top5_ids]
        top5_probs = [final_probs[t.item()].item() for t in top5_ids]

        print(f"  Final top-5 predictions:")
        for tok, prob in zip(top5_tokens, top5_probs):
            bar = "█" * int(prob * 40)
            print(f"    '{tok:>15s}' ({prob:5.1%}) {bar}")

        # Show logit lens at key layers (early, middle, late)
        key_layers = [0, 3, 6, 9, 11]
        print(f"\n  Logit lens at key layers:")
        for layer in key_layers:
            residual = cache[f"blocks.{layer}.hook_resid_post"][0, -1]
            normalized = model.ln_final(residual)
            layer_logits = normalized @ model.W_U
            top_id = layer_logits.argmax().item()
            top_token = model.tokenizer.decode(top_id).strip()
            top_prob = torch.softmax(layer_logits, dim=-1)[top_id].item()
            print(f"    Layer {layer:2d}: '{top_token:>15s}' ({top_prob:5.1%})")

    # --- Direct comparison: where do predictions diverge? ---
    print(f"\n{'=' * 60}")
    print("DIVERGENCE ANALYSIS: At which layer do Harry & Voldemort diverge?")
    print("=" * 60)

    _, cache_h = model.run_with_cache(prompts["Harry Potter"])
    _, cache_v = model.run_with_cache(prompts["Voldemort"])

    print(f"\n  Layer | Harry top pred    | Voldemort top pred | Same?")
    print(f"  {'─' * 55}")

    for layer in range(model.cfg.n_layers):
        res_h = cache_h[f"blocks.{layer}.hook_resid_post"][0, -1]
        res_v = cache_v[f"blocks.{layer}.hook_resid_post"][0, -1]

        logits_h = model.ln_final(res_h) @ model.W_U
        logits_v = model.ln_final(res_v) @ model.W_U

        tok_h = model.tokenizer.decode(logits_h.argmax().item()).strip()
        tok_v = model.tokenizer.decode(logits_v.argmax().item()).strip()
        same = "✓" if tok_h == tok_v else "✗ DIFFERENT"

        print(f"    {layer:2d}   | {tok_h:>17s} | {tok_v:>18s} | {same}")

    print("\n✦ When predictions diverge, the model has 'decided' on")
    print("  character-specific content at that layer.")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# PART 2: SAE FEATURE CONTRAST
# ============================================================================
def demo_sae_feature_contrast():
    """
    Use a pre-trained SAE to decompose activations for Harry vs Voldemort
    prompts, then identify which features are unique to each character
    and which are shared. This reveals what "concepts" the model associates
    with each character.
    """
    print("\n" + "=" * 70)
    print("PART 2: SAE FEATURE CONTRAST — HARRY vs VOLDEMORT")
    print("=" * 70)
    print("\nDecomposing activations into interpretable features to see")
    print("which concepts the model associates with each character.\n")

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print("Loading GPT-2 Small and pre-trained SAE (layer 8)...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=DEVICE,
    )
    print(f"SAE: {sae.cfg.d_in}d → {sae.cfg.d_sae} features\n")

    # Multiple prompts per character to get robust feature profiles
    harry_prompts = [
        'Pretend you are Harry Potter and tell me something Harry would say: "',
        "Harry Potter bravely faced the danger and said",
        "The things Harry Potter cares about most are",
        "Harry Potter picked up his wand and",
        "As a Gryffindor, Harry Potter believed in",
    ]

    voldemort_prompts = [
        'Pretend you are Voldemort and tell me something Voldemort would say: "',
        "Lord Voldemort raised his wand menacingly and said",
        "The things Lord Voldemort desires most are",
        "Voldemort pointed his wand and",
        "The Dark Lord Voldemort believed that power",
    ]

    W_dec = sae.W_dec.data  # decoder weights for interpreting features

    def get_feature_profile(prompts, label):
        """Get aggregated feature activations across multiple prompts."""
        all_features = []
        for prompt in prompts:
            _, cache = model.run_with_cache(prompt, prepend_bos=True)
            acts = cache["blocks.8.hook_resid_pre"]
            feats = sae.encode(acts)[0, -1]  # last token position
            all_features.append(feats)
        # Average feature activations across prompts
        avg_features = torch.stack(all_features).mean(dim=0)
        return avg_features

    print("Computing feature profiles...")
    harry_feats = get_feature_profile(harry_prompts, "Harry")
    volde_feats = get_feature_profile(voldemort_prompts, "Voldemort")

    # --- Find features unique to each character ---
    harry_active = set((harry_feats > 0.5).nonzero().squeeze(-1).tolist())
    volde_active = set((volde_feats > 0.5).nonzero().squeeze(-1).tolist())
    shared = harry_active & volde_active
    only_harry = harry_active - volde_active
    only_volde = volde_active - harry_active

    print(f"\n  Feature Summary (threshold > 0.5):")
    print(f"    Harry-active features:     {len(harry_active)}")
    print(f"    Voldemort-active features:  {len(volde_active)}")
    print(f"    Shared features:            {len(shared)}")
    print(f"    Only Harry:                 {len(only_harry)}")
    print(f"    Only Voldemort:             {len(only_volde)}")

    def describe_feature(feat_idx, feat_val):
        """Get the top tokens promoted by a feature."""
        feat_dir = W_dec[feat_idx]
        logit_effect = feat_dir @ model.W_U
        top_ids = logit_effect.topk(8).indices
        tokens = [model.tokenizer.decode(t.item()).strip() for t in top_ids]
        tokens = [t for t in tokens if t][:6]
        return tokens

    # --- Top features unique to Harry ---
    print(f"\n{'─' * 60}")
    print("TOP FEATURES UNIQUE TO HARRY POTTER:")
    print("(Features active for Harry but not Voldemort)")
    print("─" * 60)

    harry_only_sorted = sorted(only_harry, key=lambda f: harry_feats[f].item(), reverse=True)
    for feat_idx in harry_only_sorted[:10]:
        val = harry_feats[feat_idx].item()
        tokens = describe_feature(feat_idx, val)
        print(f"  Feature {feat_idx:5d} (act={val:5.2f}): → {tokens}")

    # --- Top features unique to Voldemort ---
    print(f"\n{'─' * 60}")
    print("TOP FEATURES UNIQUE TO VOLDEMORT:")
    print("(Features active for Voldemort but not Harry)")
    print("─" * 60)

    volde_only_sorted = sorted(only_volde, key=lambda f: volde_feats[f].item(), reverse=True)
    for feat_idx in volde_only_sorted[:10]:
        val = volde_feats[feat_idx].item()
        tokens = describe_feature(feat_idx, val)
        print(f"  Feature {feat_idx:5d} (act={val:5.2f}): → {tokens}")

    # --- Top shared features ---
    print(f"\n{'─' * 60}")
    print("TOP SHARED FEATURES (active for both characters):")
    print("(These represent concepts common to both — e.g., 'wizard', 'magic')")
    print("─" * 60)

    shared_sorted = sorted(shared, key=lambda f: (harry_feats[f] + volde_feats[f]).item(), reverse=True)
    for feat_idx in shared_sorted[:10]:
        h_val = harry_feats[feat_idx].item()
        v_val = volde_feats[feat_idx].item()
        tokens = describe_feature(feat_idx, h_val)
        diff_indicator = ""
        if h_val > v_val * 1.5:
            diff_indicator = " ← stronger for Harry"
        elif v_val > h_val * 1.5:
            diff_indicator = " ← stronger for Voldemort"
        print(f"  Feature {feat_idx:5d} (H={h_val:5.2f}, V={v_val:5.2f}): → {tokens}{diff_indicator}")

    # --- Feature difference score ---
    print(f"\n{'─' * 60}")
    print("MOST DIFFERENTIATING FEATURES:")
    print("(Largest activation difference between Harry and Voldemort)")
    print("─" * 60)

    diff = harry_feats - volde_feats
    # Top features where Harry > Voldemort
    harry_dominant = diff.topk(8).indices
    print("\n  Harry >> Voldemort:")
    for idx in harry_dominant:
        feat_idx = idx.item()
        h_val = harry_feats[feat_idx].item()
        v_val = volde_feats[feat_idx].item()
        tokens = describe_feature(feat_idx, h_val)
        print(f"    Feature {feat_idx:5d} (H={h_val:5.2f}, V={v_val:5.2f}, diff={h_val-v_val:+.2f}): → {tokens}")

    # Top features where Voldemort > Harry
    volde_dominant = (-diff).topk(8).indices
    print("\n  Voldemort >> Harry:")
    for idx in volde_dominant:
        feat_idx = idx.item()
        h_val = harry_feats[feat_idx].item()
        v_val = volde_feats[feat_idx].item()
        tokens = describe_feature(feat_idx, h_val)
        print(f"    Feature {feat_idx:5d} (H={h_val:5.2f}, V={v_val:5.2f}, diff={h_val-v_val:+.2f}): → {tokens}")

    print("\n✦ Unique features reveal what concepts the model uniquely")
    print("  associates with each character. Shared features show common")
    print("  context (e.g., magic, fantasy). Differentiating features show")
    print("  the model's internal 'personality profile' for each character.")

    del model, sae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# PART 3: ACTIVATION DISTANCE ANALYSIS
# ============================================================================
def demo_activation_distance():
    """
    Measure how far apart the model's internal representations are
    for different characters at each layer. This reveals at which
    depth the model starts differentiating between characters.
    """
    print("\n" + "=" * 70)
    print("PART 3: ACTIVATION DISTANCE — WHERE CHARACTERS DIVERGE")
    print("=" * 70)
    print("\nMeasuring representation distance between characters at each layer.")
    print("This reveals WHERE in the network character differentiation happens.\n")

    from transformer_lens import HookedTransformer

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # Character pairs to compare
    character_pairs = [
        (
            "Harry Potter",
            'Pretend you are Harry Potter and say something Harry would say: "',
            "Voldemort",
            'Pretend you are Voldemort and say something Voldemort would say: "',
        ),
        (
            "Dumbledore",
            'Pretend you are Dumbledore and say something wise: "',
            "Voldemort",
            'Pretend you are Voldemort and say something Voldemort would say: "',
        ),
        (
            "Harry Potter",
            'Pretend you are Harry Potter and say something Harry would say: "',
            "Hermione",
            'Pretend you are Hermione Granger and say something Hermione would say: "',
        ),
    ]

    for name_a, prompt_a, name_b, prompt_b in character_pairs:
        print(f"\n{'─' * 60}")
        print(f"Comparing: {name_a} vs {name_b}")
        print(f"─" * 60)

        _, cache_a = model.run_with_cache(prompt_a)
        _, cache_b = model.run_with_cache(prompt_b)

        print(f"\n  Layer | Cosine Sim | L2 Distance | Visualization")
        print(f"  {'─' * 55}")

        cosine_sims = []
        l2_dists = []

        for layer in range(model.cfg.n_layers):
            # Get last-token activations at this layer
            act_a = cache_a[f"blocks.{layer}.hook_resid_post"][0, -1]
            act_b = cache_b[f"blocks.{layer}.hook_resid_post"][0, -1]

            # Cosine similarity (1 = identical direction, 0 = orthogonal)
            cos_sim = torch.nn.functional.cosine_similarity(
                act_a.unsqueeze(0), act_b.unsqueeze(0)
            ).item()

            # L2 distance (normalized by vector magnitude)
            l2 = (act_a - act_b).norm().item()
            avg_norm = (act_a.norm().item() + act_b.norm().item()) / 2
            rel_l2 = l2 / avg_norm if avg_norm > 0 else 0

            cosine_sims.append(cos_sim)
            l2_dists.append(rel_l2)

            # Visual bar: longer = more different
            diff_bar = "█" * int((1 - cos_sim) * 80)
            print(f"    {layer:2d}   |   {cos_sim:.4f}  |    {rel_l2:.4f}    | {diff_bar}")

        # Summary
        min_sim_layer = np.argmin(cosine_sims)
        max_diff_layer = np.argmax(l2_dists)
        print(f"\n  Most different layer (cosine): {min_sim_layer} (sim={cosine_sims[min_sim_layer]:.4f})")
        print(f"  Most different layer (L2):     {max_diff_layer} (rel_dist={l2_dists[max_diff_layer]:.4f})")

    # --- Compare character pairs: which are most/least similar? ---
    print(f"\n{'=' * 60}")
    print("CROSS-CHARACTER SIMILARITY MATRIX (last layer, cosine sim)")
    print("=" * 60)

    characters = {
        "Harry": 'Pretend you are Harry Potter and say something: "',
        "Voldemort": 'Pretend you are Voldemort and say something: "',
        "Dumbledore": 'Pretend you are Dumbledore and say something: "',
        "Hermione": 'Pretend you are Hermione and say something: "',
        "Snape": 'Pretend you are Severus Snape and say something: "',
    }

    # Collect last-layer activations
    char_acts = {}
    for name, prompt in characters.items():
        _, cache = model.run_with_cache(prompt)
        char_acts[name] = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"][0, -1]

    names = list(characters.keys())
    print(f"\n  {'':>12s}", end="")
    for n in names:
        print(f" {n:>10s}", end="")
    print()

    for i, n1 in enumerate(names):
        print(f"  {n1:>12s}", end="")
        for j, n2 in enumerate(names):
            sim = torch.nn.functional.cosine_similarity(
                char_acts[n1].unsqueeze(0), char_acts[n2].unsqueeze(0)
            ).item()
            if i == j:
                print(f"     {'1.000':>5s}", end="")
            else:
                print(f"     {sim:.3f}", end="")
        print()

    print("\n✦ Higher cosine similarity = more similar internal representation.")
    print("  Characters with aligned values (Harry/Hermione/Dumbledore) should")
    print("  cluster together, while antagonists (Voldemort) should be further.")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# PART 4: CHARACTER STEERING
# ============================================================================
def demo_character_steering():
    """
    Find the "Harry ↔ Voldemort direction" in activation space,
    then use it to steer the model: make Harry sound more like
    Voldemort and vice versa.
    
    This demonstrates that character traits are linearly encoded
    in the model's representation space.
    """
    print("\n" + "=" * 70)
    print("PART 4: CHARACTER STEERING — HARRY ↔ VOLDEMORT")
    print("=" * 70)
    print("\nFinding the direction that separates Harry from Voldemort")
    print("in the model's internal space, then steering generation.\n")

    from transformer_lens import HookedTransformer

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # --- Step 1: Compute the character direction ---
    # Use multiple prompts to get a robust direction
    harry_prompts = [
        "Harry Potter bravely faced the enemy with courage",
        "Harry Potter cared deeply about his friends and protecting them",
        "Harry Potter chose to do the right thing despite the danger",
        "The Boy Who Lived showed kindness and compassion to everyone",
        "Harry used his wand to defend the innocent and helpless",
        "Gryffindor courage ran through Harry Potter's veins",
        "Harry Potter would sacrifice himself to save his friends",
        "The chosen one stood up against evil with determination",
    ]

    voldemort_prompts = [
        "Lord Voldemort ruthlessly pursued immortality and power",
        "Voldemort cared only about domination and feared death above all",
        "The Dark Lord chose to destroy anyone who opposed him",
        "He Who Must Not Be Named showed cruelty and contempt",
        "Voldemort used his wand to inflict pain and suffering",
        "Slytherin ambition consumed Lord Voldemort entirely",
        "Voldemort would sacrifice anyone to achieve his goals",
        "The Dark Lord crushed opposition with merciless force",
    ]

    steering_layer = 6  # Middle layer — good for conceptual steering

    print(f"Computing character direction from {len(harry_prompts)} prompt pairs...")

    harry_acts = []
    volde_acts = []

    for prompt in harry_prompts:
        _, cache = model.run_with_cache(prompt, prepend_bos=True)
        act = cache[f"blocks.{steering_layer}.hook_resid_post"][0].mean(dim=0)
        harry_acts.append(act)

    for prompt in voldemort_prompts:
        _, cache = model.run_with_cache(prompt, prepend_bos=True)
        act = cache[f"blocks.{steering_layer}.hook_resid_post"][0].mean(dim=0)
        volde_acts.append(act)

    harry_mean = torch.stack(harry_acts).mean(dim=0)
    volde_mean = torch.stack(volde_acts).mean(dim=0)

    # The "Voldemort direction" = Voldemort - Harry
    voldemort_direction = volde_mean - harry_mean
    voldemort_direction = voldemort_direction / voldemort_direction.norm()

    # The "Harry direction" is the opposite
    harry_direction = -voldemort_direction

    cos_sim = torch.nn.functional.cosine_similarity(
        harry_mean.unsqueeze(0), volde_mean.unsqueeze(0)
    ).item()
    print(f"Direction computed. Base cosine similarity: {cos_sim:.4f}")
    print(f"(Lower = more different representations)\n")

    # --- Step 2: Steer generation ---
    test_scenarios = [
        {
            "prompt": "The wizard raised his wand and said",
            "description": "Neutral wizard prompt",
        },
        {
            "prompt": "Looking at the castle, the wizard thought about",
            "description": "Neutral reflection prompt",
        },
        {
            "prompt": "The most important thing in life is",
            "description": "Values prompt (no character context)",
        },
    ]

    steering_strengths = [0, 10, 20]  # 0 = baseline, positive = more Voldemort

    for scenario in test_scenarios:
        print(f"\n{'━' * 60}")
        print(f"Scenario: {scenario['description']}")
        print(f"Prompt: \"{scenario['prompt']}\"")
        print("━" * 60)

        for strength in steering_strengths:
            if strength == 0:
                label = "Baseline (no steering)"
            elif strength > 0:
                label = f"→ Voldemort direction (strength={strength})"
            else:
                label = f"→ Harry direction (strength={abs(strength)})"

            direction = voldemort_direction

            def make_hook(s, d):
                def hook_fn(activation, hook):
                    activation[:, :, :] += s * d
                    return activation
                return hook_fn

            hook_name = f"blocks.{steering_layer}.hook_resid_post"

            if strength == 0:
                text = model.generate(
                    scenario["prompt"],
                    max_new_tokens=35,
                    temperature=0.7,
                    do_sample=True,
                    verbose=False,
                )
            else:
                with model.hooks(fwd_hooks=[(hook_name, make_hook(strength, direction))]):
                    text = model.generate(
                        scenario["prompt"],
                        max_new_tokens=35,
                        temperature=0.7,
                        do_sample=True,
                        verbose=False,
                    )

            print(f"\n  [{label}]")
            print(f"  {text}")

        # Also try steering toward Harry (negative direction)
        print(f"\n  [→ Harry direction (strength=20)]")
        with model.hooks(fwd_hooks=[(hook_name, make_hook(20, harry_direction))]):
            text = model.generate(
                scenario["prompt"],
                max_new_tokens=35,
                temperature=0.7,
                do_sample=True,
                verbose=False,
            )
        print(f"  {text}")

    # --- Step 3: Verify the direction with a sanity check ---
    print(f"\n{'━' * 60}")
    print("SANITY CHECK: Does the direction encode character identity?")
    print("━" * 60)

    test_chars = [
        ("Harry Potter is a brave wizard who", "Harry Potter"),
        ("Lord Voldemort is a dark wizard who", "Voldemort"),
        ("Albus Dumbledore wisely said that", "Dumbledore"),
        ("Draco Malfoy sneered and said", "Draco Malfoy"),
        ("Hermione Granger studied hard and", "Hermione"),
        ("Severus Snape coldly replied", "Snape"),
    ]

    print(f"\n  Character prompt → projection onto Voldemort direction")
    print(f"  (positive = more Voldemort-like, negative = more Harry-like)")
    print(f"  {'─' * 55}")

    for prompt, name in test_chars:
        _, cache = model.run_with_cache(prompt, prepend_bos=True)
        act = cache[f"blocks.{steering_layer}.hook_resid_post"][0].mean(dim=0)
        projection = (act @ voldemort_direction).item()

        # Normalize relative to the Harry-Voldemort scale
        harry_proj = (harry_mean @ voldemort_direction).item()
        volde_proj = (volde_mean @ voldemort_direction).item()
        scale = volde_proj - harry_proj
        normalized = (projection - harry_proj) / scale if scale != 0 else 0

        bar_pos = int(max(0, normalized) * 20)
        bar_neg = int(max(0, -normalized) * 20)
        bar = "◄" * bar_neg + "│" + "►" * bar_pos
        side = "Voldemort-like" if normalized > 0.3 else "Harry-like" if normalized < -0.3 else "Neutral"
        print(f"    {name:>15s}: {normalized:+.3f}  {bar:>25s}  ({side})")

    print("\n✦ If the direction is meaningful, 'good' characters should project")
    print("  negatively (Harry-like) and 'dark' characters positively (Voldemort-like).")
    print("  This validates that character traits are linearly encoded!")

    print("\n" + "─" * 60)
    print("RESEARCH TIPS FOR YOUR WORK:")
    print("─" * 60)
    print("""
  You can extend this analysis by:

  1. Testing more character pairs (historical figures, fictional chars, etc.)
     - "Pretend you are Gandhi" vs "Pretend you are a dictator"
     - "Pretend you are Shakespeare" vs "Pretend you are a modern rapper"

  2. Using different layers for steering to see layer-specific effects:
     - Early layers (0-3): affect broad context/topic
     - Middle layers (4-7): affect style/tone/personality
     - Late layers (8-11): affect specific word choices

  3. Combining with SAE features to name the concepts in the direction:
     - Project the steering direction through the SAE decoder
     - Find which interpretable features align with the direction

  4. Testing with larger models via nnsight (remote, no GPU needed):
     from nnsight import LanguageModel
     model = LanguageModel("meta-llama/Llama-3.1-8B", device_map="auto")
     # Same analysis, but on a much more capable model

  5. Comparing character directions across different model architectures
     to see if character representations are universal
""")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  CHARACTER COMPARISON DEMO — Mechanistic Interpretability      ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  Research question: How does a model internally represent      ║")
    print("║  different characters/personas (Harry Potter vs Voldemort)?    ║")
    print("║                                                                ║")
    print("║  Parts:                                                        ║")
    print("║    1. Logit Lens — layer-by-layer prediction comparison        ║")
    print("║    2. SAE Features — which concepts activate per character?    ║")
    print("║    3. Activation Distance — where do characters diverge?       ║")
    print("║    4. Character Steering — shift Harry → Voldemort & back      ║")
    print("║                                                                ║")
    print("║  Hardware: CPU only, ~4 GB RAM. No GPU needed.                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    try:
        demo_logit_lens_comparison()
    except Exception as e:
        print(f"\n[!] Logit Lens comparison failed: {e}")
        import traceback; traceback.print_exc()

    try:
        demo_sae_feature_contrast()
    except Exception as e:
        print(f"\n[!] SAE Feature contrast failed: {e}")
        import traceback; traceback.print_exc()

    try:
        demo_activation_distance()
    except Exception as e:
        print(f"\n[!] Activation Distance analysis failed: {e}")
        import traceback; traceback.print_exc()

    try:
        demo_character_steering()
    except Exception as e:
        print(f"\n[!] Character Steering failed: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)
    print("\nSee also:")
    print("  - mech_interp_demo.py — general MI techniques demo")
    print("  - mechanistic_interpretability_report.md — full research report")
    print("  - https://www.neuronpedia.org — explore features interactively")


if __name__ == "__main__":
    main()
