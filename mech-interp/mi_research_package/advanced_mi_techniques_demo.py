#!/usr/bin/env python3
"""
Advanced MI Techniques Demo — Beyond Vectors & Words
=====================================================

This script demonstrates MI techniques BEYOND the activation/word-level
analysis covered in previous demos. These are the other families of
interpretability research.

Techniques demonstrated:
  1. Attention Pattern Analysis — where does the model look?
  2. Circuit Discovery (simplified) — what subnetwork does this task?
  3. Automated Interpretability — labeling what neurons/features detect
  4. Probing Classifiers — does the model represent specific concepts?
  5. Representation Similarity — comparing layers and representations

Requirements:
  pip install transformer-lens torch numpy scikit-learn

Hardware: CPU only, ~4 GB RAM. No GPU required.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np

DEVICE = "cpu"


# ============================================================================
# PART 1: ATTENTION PATTERN ANALYSIS — Where does the model look?
# ============================================================================
def demo_attention_patterns():
    """
    Analyze attention patterns to understand WHERE the model directs
    its attention for different prompts. Some attention heads have
    clear algorithmic roles:
      - "Previous token" heads: always attend to the prior token
      - "Induction heads": copy patterns seen earlier
      - "Name mover heads": route entity names to the prediction position
    """
    print("\n" + "=" * 70)
    print("PART 1: ATTENTION PATTERN ANALYSIS — Where Does the Model Look?")
    print("=" * 70)
    print("\nAttention heads route information between tokens. By analyzing")
    print("attention patterns, we can see WHERE the model looks to make")
    print("predictions — not what it computes, but what it reads.\n")

    from transformer_lens import HookedTransformer

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # --- Analyze attention on a factual prompt ---
    prompt = "When John gave the book to Mary, Mary gave it back to"
    tokens = model.to_tokens(prompt)
    str_tokens = model.to_str_tokens(prompt)
    print(f'Prompt: "{prompt}"')
    print(f"Tokens: {str_tokens}\n")

    _, cache = model.run_with_cache(tokens)

    # The model should predict "John" here. Let's see which heads
    # attend the last position to "John" (token index ~1)
    john_pos = None
    mary_positions = []
    for i, tok in enumerate(str_tokens):
        if "John" in tok:
            john_pos = i
        if "Mary" in tok:
            mary_positions.append(i)

    last_pos = len(str_tokens) - 1
    print(f"  'John' is at position {john_pos}")
    print(f"  'Mary' is at positions {mary_positions}")
    print(f"  Last token (prediction point) is at position {last_pos}")

    # --- Find which heads attend from last position to "John" ---
    print(f"\n{'─' * 60}")
    print("WHICH HEADS ATTEND TO 'JOHN' FROM THE LAST POSITION?")
    print("(These are candidate 'Name Mover' heads)")
    print(f"{'─' * 60}\n")

    head_scores = []
    for layer in range(model.cfg.n_layers):
        attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"]  # (batch, heads, dest, src)
        for head in range(model.cfg.n_heads):
            attn_to_john = attn_pattern[0, head, last_pos, john_pos].item()
            head_scores.append((layer, head, attn_to_john))

    # Sort by attention to John
    head_scores.sort(key=lambda x: -x[2])

    print(f"  Top 10 heads attending to 'John' from last position:")
    print(f"  {'Layer':>5} {'Head':>5} {'Attn to John':>15}  Bar")
    print(f"  {'─' * 45}")
    for layer, head, score in head_scores[:10]:
        bar = "█" * int(score * 40)
        print(f"  {layer:5d} {head:5d} {score:15.4f}  {bar}")

    # --- Classify head types across the model ---
    print(f"\n{'─' * 60}")
    print("ATTENTION HEAD TYPE CLASSIFICATION")
    print("(Detecting algorithmic roles of each head)")
    print(f"{'─' * 60}\n")

    # Test with a longer prompt to detect patterns
    test_prompt = "The cat sat on the mat. The dog sat on the"
    test_tokens = model.to_tokens(test_prompt)
    test_str_tokens = model.to_str_tokens(test_prompt)
    _, test_cache = model.run_with_cache(test_tokens)

    n_tokens = len(test_str_tokens)

    prev_token_heads = []
    diagonal_heads = []
    induction_candidates = []

    for layer in range(model.cfg.n_layers):
        attn = test_cache[f"blocks.{layer}.attn.hook_pattern"][0]  # (heads, dest, src)
        for head in range(model.cfg.n_heads):
            pattern = attn[head]  # (dest, src)

            # Check: "previous token" head — does it mostly attend to position dest-1?
            prev_token_score = 0.0
            count = 0
            for dest in range(1, n_tokens):
                prev_token_score += pattern[dest, dest - 1].item()
                count += 1
            prev_token_score /= max(count, 1)

            if prev_token_score > 0.3:
                prev_token_heads.append((layer, head, prev_token_score))

            # Check: "identity/diagonal" head — does it attend to same position?
            diag_score = 0.0
            for dest in range(n_tokens):
                diag_score += pattern[dest, dest].item()
            diag_score /= n_tokens

            if diag_score > 0.3:
                diagonal_heads.append((layer, head, diag_score))

            # Check: "induction" head — does it attend to token after previous
            # occurrence of current token? (Simplified check)
            # For "The cat sat on the mat. The dog sat on the"
            # At the last "the", an induction head would attend to tokens that
            # followed previous "the" tokens
            # This is a simplified heuristic
            if layer >= 1:  # Induction heads need at least 1 prior layer
                # Check if attention is concentrated on specific earlier positions
                last_attn = pattern[-1]  # attention from last token
                entropy = -(last_attn * (last_attn + 1e-10).log()).sum().item()
                if entropy < 2.0 and prev_token_score < 0.2:
                    induction_candidates.append((layer, head, entropy))

    print("  PREVIOUS TOKEN HEADS (attend to position dest-1):")
    prev_token_heads.sort(key=lambda x: -x[2])
    for layer, head, score in prev_token_heads[:5]:
        print(f"    L{layer}H{head}: avg prev-token attention = {score:.3f}")

    print(f"\n  IDENTITY/DIAGONAL HEADS (attend to same position):")
    diagonal_heads.sort(key=lambda x: -x[2])
    for layer, head, score in diagonal_heads[:5]:
        print(f"    L{layer}H{head}: avg self-attention = {score:.3f}")

    print(f"\n  INDUCTION HEAD CANDIDATES (low entropy, pattern-copying):")
    induction_candidates.sort(key=lambda x: x[2])
    for layer, head, entropy in induction_candidates[:5]:
        print(f"    L{layer}H{head}: attention entropy = {entropy:.3f} (lower = more focused)")

    # --- Attention pattern comparison: factual vs creative prompt ---
    print(f"\n{'─' * 60}")
    print("ATTENTION COMPARISON: Factual vs Creative Prompts")
    print(f"{'─' * 60}\n")

    prompts = {
        "Factual": "The capital of France is",
        "Creative": "Once upon a time in a magical forest there lived",
    }

    for label, p in prompts.items():
        tok = model.to_tokens(p)
        strs = model.to_str_tokens(p)
        _, c = model.run_with_cache(tok)

        # Compute average attention entropy across all heads (last position)
        entropies = []
        for layer in range(model.cfg.n_layers):
            attn = c[f"blocks.{layer}.attn.hook_pattern"][0]
            for head in range(model.cfg.n_heads):
                last_attn = attn[head, -1, :len(strs)]
                entropy = -(last_attn * (last_attn + 1e-10).log()).sum().item()
                entropies.append(entropy)

        avg_entropy = np.mean(entropies)
        max_entropy = np.log(len(strs))
        print(f"  {label:10s}: avg attention entropy = {avg_entropy:.3f} "
              f"(max possible: {max_entropy:.3f})")
        print(f"              → {'diffuse' if avg_entropy > max_entropy * 0.6 else 'focused'} attention")

    print("\n✦ Factual prompts tend to produce more FOCUSED attention")
    print("  (the model knows exactly where to look for the answer),")
    print("  while creative prompts produce more DIFFUSE attention")
    print("  (the model draws from many positions to generate text).")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# PART 2: CIRCUIT DISCOVERY (Simplified) — What subnetwork does this task?
# ============================================================================
def demo_circuit_discovery():
    """
    Simplified circuit discovery: find which components (attention heads
    and MLP layers) are responsible for a specific task by ablating them
    one at a time and measuring the effect on performance.

    This is a simplified version of the ACDC algorithm and manual
    circuit analysis used in landmark papers.
    """
    print("\n" + "=" * 70)
    print("PART 2: CIRCUIT DISCOVERY — What Subnetwork Does This Task?")
    print("=" * 70)
    print("\nWe ablate (zero out) each component one at a time and measure")
    print("how much the model's ability to do a specific task degrades.")
    print("Components with large effects are part of the 'circuit'.\n")

    from transformer_lens import HookedTransformer

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # --- Task: Greater-Than (a well-studied circuit) ---
    # "The war lasted from 1732 to 17" → model should predict a digit > 32
    print("Task: Greater-Than")
    print('Prompt: "The war lasted from 1732 to 17"')
    print("Expected: model should predict digits > 32 (e.g., 33, 34, ...99)\n")

    prompt = "The war lasted from 1732 to 17"
    tokens = model.to_tokens(prompt)

    # Baseline: get logits for all two-digit completions
    baseline_logits = model(tokens)[0, -1]

    # Tokens for digits 0-9
    digit_tokens = [model.to_tokens(str(d))[0, 1].item() for d in range(10)]

    # For "17XX", the first digit after "17" should be > 3
    # (valid continuations: 33-99, so first digit should be 3,4,5,6,7,8,9)
    greater_digits = [3, 4, 5, 6, 7, 8, 9]
    lesser_digits = [0, 1, 2]

    baseline_greater_logit = torch.stack([baseline_logits[digit_tokens[d]] for d in greater_digits]).mean().item()
    baseline_lesser_logit = torch.stack([baseline_logits[digit_tokens[d]] for d in lesser_digits]).mean().item()
    baseline_gap = baseline_greater_logit - baseline_lesser_logit

    print(f"  Baseline avg logit for digits > 2: {baseline_greater_logit:.2f}")
    print(f"  Baseline avg logit for digits ≤ 2: {baseline_lesser_logit:.2f}")
    print(f"  Baseline gap (higher = better):     {baseline_gap:.2f}")

    # --- Ablate each attention head and measure effect ---
    print(f"\n{'─' * 60}")
    print("ABLATING ATTENTION HEADS — Which heads matter for Greater-Than?")
    print(f"{'─' * 60}\n")

    head_effects = []

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            def make_ablation_hook(l, h):
                def hook_fn(activation, hook):
                    # Zero out this head's output
                    activation[0, :, h, :] = 0.0
                    return activation
                return hook_fn

            with model.hooks(fwd_hooks=[
                (f"blocks.{layer}.attn.hook_result", make_ablation_hook(layer, head))
            ]):
                ablated_logits = model(tokens)[0, -1]

            abl_greater = torch.stack([ablated_logits[digit_tokens[d]] for d in greater_digits]).mean().item()
            abl_lesser = torch.stack([ablated_logits[digit_tokens[d]] for d in lesser_digits]).mean().item()
            abl_gap = abl_greater - abl_lesser

            effect = baseline_gap - abl_gap  # Positive = removing this head hurts
            head_effects.append((layer, head, effect))

    # Sort by absolute effect
    head_effects.sort(key=lambda x: -abs(x[2]))

    print(f"  Top 15 most important attention heads for Greater-Than:")
    print(f"  {'Layer':>5} {'Head':>5} {'Effect':>10}  Role")
    print(f"  {'─' * 50}")
    for layer, head, effect in head_effects[:15]:
        direction = "PROMOTES >" if effect > 0 else "promotes ≤"
        bar = "█" * int(abs(effect) * 5)
        print(f"  {layer:5d} {head:5d} {effect:+10.3f}  {direction} {bar}")

    # --- Ablate MLP layers ---
    print(f"\n{'─' * 60}")
    print("ABLATING MLP LAYERS — Which MLPs matter?")
    print(f"{'─' * 60}\n")

    mlp_effects = []
    for layer in range(model.cfg.n_layers):
        def make_mlp_hook(l):
            def hook_fn(activation, hook):
                activation[0, :, :] = 0.0
                return activation
            return hook_fn

        with model.hooks(fwd_hooks=[
            (f"blocks.{layer}.hook_mlp_out", make_mlp_hook(layer))
        ]):
            ablated_logits = model(tokens)[0, -1]

        abl_greater = torch.stack([ablated_logits[digit_tokens[d]] for d in greater_digits]).mean().item()
        abl_lesser = torch.stack([ablated_logits[digit_tokens[d]] for d in lesser_digits]).mean().item()
        abl_gap = abl_greater - abl_lesser
        effect = baseline_gap - abl_gap
        mlp_effects.append((layer, effect))

    print(f"  {'Layer':>5} {'MLP Effect':>12}  Bar")
    print(f"  {'─' * 35}")
    for layer, effect in mlp_effects:
        direction = "►" if effect > 0 else "◄"
        bar = "█" * int(abs(effect) * 5)
        print(f"  {layer:5d} {effect:+12.3f}  {direction} {bar}")

    # --- Summarize the circuit ---
    print(f"\n{'─' * 60}")
    print("DISCOVERED CIRCUIT SUMMARY")
    print(f"{'─' * 60}\n")

    important_heads = [(l, h, e) for l, h, e in head_effects if abs(e) > 0.3]
    important_mlps = [(l, e) for l, e in mlp_effects if abs(e) > 0.3]

    print(f"  Circuit components (effect > 0.3):")
    print(f"  Attention heads: {len(important_heads)}")
    for l, h, e in important_heads:
        role = "promotes greater-than" if e > 0 else "inhibits greater-than"
        print(f"    L{l}H{h} ({role}, effect={e:+.3f})")
    print(f"  MLP layers: {len(important_mlps)}")
    for l, e in important_mlps:
        role = "promotes greater-than" if e > 0 else "inhibits greater-than"
        print(f"    MLP{l} ({role}, effect={e:+.3f})")

    total_components = model.cfg.n_layers * model.cfg.n_heads + model.cfg.n_layers
    circuit_size = len(important_heads) + len(important_mlps)
    print(f"\n  Total model components: {total_components}")
    print(f"  Circuit components: {circuit_size} ({circuit_size/total_components:.1%} of the model)")
    print(f"\n✦ Only a small fraction of the model is needed for this task!")
    print("  The original Greater-Than circuit paper (Hanna et al. 2023)")
    print("  found a similar small circuit in GPT-2 Small.")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# PART 3: AUTOMATED INTERPRETABILITY — Labeling what neurons detect
# ============================================================================
def demo_automated_interpretability():
    """
    Automated interpretability: instead of using an external LLM to label
    features (which requires API access), we demonstrate the PRINCIPLE
    by analyzing what maximally activates each neuron/feature and
    generating structured labels from the activation patterns.

    This mirrors what OpenAI did with GPT-4 labeling GPT-2 neurons,
    and what Neuronpedia does for SAE features.
    """
    print("\n" + "=" * 70)
    print("PART 3: AUTOMATED INTERPRETABILITY — Labeling What Neurons Detect")
    print("=" * 70)
    print("\nWe find what inputs maximally activate specific neurons, then")
    print("use the neuron's output weights to infer what it 'means'.")
    print("This is the same principle behind Neuronpedia and OpenAI's")
    print("automated neuron labeling.\n")

    from transformer_lens import HookedTransformer

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # --- Corpus of diverse prompts to find what activates neurons ---
    corpus = [
        "The president of the United States visited France",
        "She was feeling very happy and excited about the news",
        "The chemical formula for water is H2O",
        "In 1969, humans first landed on the moon",
        "The cat sat quietly on the warm windowsill",
        "Stock prices crashed dramatically on Wall Street",
        "The algorithm runs in O(n log n) time complexity",
        "Harry Potter waved his wand and cast a spell",
        "The patient was diagnosed with acute myocardial infarction",
        "She cooked a delicious Italian pasta with fresh tomatoes",
        "The quantum entanglement experiment confirmed Bell's theorem",
        "The football match ended with a dramatic penalty shootout",
        "Climate change is causing unprecedented global warming",
        "The ancient Egyptian pyramids were built around 2560 BC",
        "He wrote beautiful poetry about love and loss",
        "The neural network achieved 99% accuracy on the test set",
        "The judge sentenced the defendant to five years in prison",
        "The orchestra performed Beethoven's Symphony No. 9",
        "GDP growth exceeded expectations in the third quarter",
        "The volcano erupted violently, sending ash into the sky",
    ]

    # --- Find maximally activating prompts for specific MLP neurons ---
    target_layer = 6
    n_neurons = model.cfg.d_mlp  # 3072 for GPT-2 Small

    print(f"Analyzing MLP neurons at layer {target_layer}...")
    print(f"Running {len(corpus)} diverse prompts through the model...\n")

    # Collect activations for all prompts
    all_activations = []  # (n_prompts, d_mlp) — max activation per prompt
    for prompt in corpus:
        tok = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tok)
        # Get MLP post-activation (after GELU) — shape (1, seq_len, d_mlp)
        mlp_act = cache[f"blocks.{target_layer}.mlp.hook_post"]
        # Take max activation across positions for each neuron
        max_act = mlp_act[0].max(dim=0).values  # (d_mlp,)
        all_activations.append(max_act)

    activation_matrix = torch.stack(all_activations)  # (n_prompts, d_mlp)

    # --- For selected neurons, show what they detect ---
    print(f"{'─' * 60}")
    print("NEURON PROFILES: What does each neuron detect?")
    print(f"{'─' * 60}\n")

    # Find neurons with high variance (interesting, not always on/off)
    neuron_variance = activation_matrix.var(dim=0)
    interesting_neurons = neuron_variance.topk(8).indices.tolist()

    W_out = model.blocks[target_layer].mlp.W_out.data  # (d_mlp, d_model)
    W_U = model.W_U  # (d_model, d_vocab)

    for neuron_idx in interesting_neurons:
        activations = activation_matrix[:, neuron_idx]

        # Find top-activating prompts
        top_prompt_indices = activations.topk(3).indices.tolist()
        bottom_prompt_indices = (-activations).topk(2).indices.tolist()

        # What tokens does this neuron promote?
        value_vector = W_out[neuron_idx]  # (d_model,)
        vocab_effect = value_vector @ W_U  # (d_vocab,)
        top_tokens = [model.tokenizer.decode(t.item()).strip()
                      for t in vocab_effect.topk(8).indices]
        top_tokens = [t for t in top_tokens if t][:5]

        # Auto-generate a label based on promoted tokens and top prompts
        print(f"  Neuron {neuron_idx} (variance: {neuron_variance[neuron_idx]:.2f}):")
        print(f"    Promotes tokens: {top_tokens}")
        print(f"    Top-activating prompts:")
        for idx in top_prompt_indices:
            print(f"      [{activations[idx]:.2f}] \"{corpus[idx][:65]}...\"")
        print(f"    Low-activating prompts:")
        for idx in bottom_prompt_indices:
            print(f"      [{activations[idx]:.2f}] \"{corpus[idx][:65]}...\"")
        print()

    # --- "Always-on" vs "context-specific" neurons ---
    print(f"{'─' * 60}")
    print("NEURON TYPES: Always-On vs Context-Specific")
    print(f"{'─' * 60}\n")

    mean_activation = activation_matrix.mean(dim=0)
    neuron_cv = activation_matrix.std(dim=0) / (mean_activation.abs() + 1e-8)  # Coefficient of variation

    # Always-on neurons: high mean, low variance
    always_on_mask = (mean_activation > mean_activation.quantile(0.95)) & (neuron_cv < neuron_cv.quantile(0.3))
    always_on = always_on_mask.nonzero().squeeze(-1).tolist()[:5]

    # Context-specific: moderate mean, high variance
    context_mask = (neuron_variance > neuron_variance.quantile(0.9))
    context_specific = context_mask.nonzero().squeeze(-1).tolist()[:5]

    # Dead neurons: near-zero activation on all prompts
    dead_mask = mean_activation.abs() < 0.01
    n_dead = dead_mask.sum().item()

    print(f"  Always-On neurons (fire on everything): ~{len(always_on)} examples")
    for n in always_on[:3]:
        top_toks = [model.tokenizer.decode(t.item()).strip()
                    for t in (W_out[n] @ W_U).topk(5).indices]
        top_toks = [t for t in top_toks if t][:4]
        print(f"    N{n}: mean act={mean_activation[n]:.2f}, promotes {top_toks}")

    print(f"\n  Context-Specific neurons (fire selectively): ~{len(context_specific)} examples")
    for n in context_specific[:3]:
        top_toks = [model.tokenizer.decode(t.item()).strip()
                    for t in (W_out[n] @ W_U).topk(5).indices]
        top_toks = [t for t in top_toks if t][:4]
        top_prompt = activations.topk(1).indices[0].item()
        print(f"    N{n}: var={neuron_variance[n]:.2f}, promotes {top_toks}")

    print(f"\n  Dead neurons (near-zero activation): {n_dead} / {n_neurons} ({n_dead/n_neurons:.1%})")

    print("\n✦ This is the same analysis pipeline behind Neuronpedia:")
    print("  find what activates each feature, what it promotes, and")
    print("  auto-generate a human-readable label. At scale, an LLM")
    print("  (like GPT-4) reads these profiles and writes descriptions")
    print("  like 'this neuron detects medical terminology'.")
    print("  See: https://www.neuronpedia.org")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# PART 4: PROBING CLASSIFIERS — Does the model represent concept X?
# ============================================================================
def demo_probing_classifiers():
    """
    Train linear classifiers ('probes') on the model's hidden states
    to test whether specific concepts are linearly represented at
    each layer. This tells us WHAT the model knows at each depth.

    We test:
      - Sentiment (positive/negative)
      - Topic (science/non-science)
      - Factual domain detection
    """
    print("\n" + "=" * 70)
    print("PART 4: PROBING CLASSIFIERS — Does the Model Represent Concepts?")
    print("=" * 70)
    print("\nWe train linear classifiers on hidden states to test if specific")
    print("concepts (sentiment, topic, etc.) are linearly represented at")
    print("each layer. If a simple linear probe succeeds, the concept is")
    print("explicitly encoded in the model's representation.\n")

    from transformer_lens import HookedTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # --- Probing dataset: Sentiment ---
    print(f"{'─' * 60}")
    print("PROBE 1: SENTIMENT (positive vs negative)")
    print(f"{'─' * 60}\n")

    sentiment_data = [
        # Positive examples (label=1)
        ("I absolutely love this wonderful day", 1),
        ("This is the best thing that ever happened to me", 1),
        ("She was thrilled and delighted by the surprise", 1),
        ("What a fantastic and beautiful performance", 1),
        ("The children were laughing and playing happily", 1),
        ("I feel so grateful and blessed today", 1),
        ("This movie was amazing and truly inspiring", 1),
        ("He smiled warmly and gave her a big hug", 1),
        ("The garden looked beautiful with blooming flowers", 1),
        ("I am so proud of this incredible achievement", 1),
        ("We had an excellent dinner at the restaurant", 1),
        ("The team celebrated their glorious victory", 1),
        # Negative examples (label=0)
        ("I hate this terrible and awful situation", 0),
        ("This is the worst experience of my life", 0),
        ("She was devastated and heartbroken by the news", 0),
        ("What a horrible and disgusting display", 0),
        ("The people were crying and screaming in fear", 0),
        ("I feel so angry and frustrated today", 0),
        ("This movie was dreadful and utterly boring", 0),
        ("He frowned angrily and slammed the door shut", 0),
        ("The building looked ugly and run down", 0),
        ("I am so disappointed by this pathetic failure", 0),
        ("We had a terrible meal at the restaurant", 0),
        ("The team suffered a humiliating defeat", 0),
    ]

    # Collect hidden states at each layer
    print("  Collecting hidden states for sentiment probing...")
    layer_representations = {l: [] for l in range(model.cfg.n_layers)}
    labels = []

    for text, label in sentiment_data:
        tok = model.to_tokens(text)
        _, cache = model.run_with_cache(tok)
        for layer in range(model.cfg.n_layers):
            # Use mean of all token representations
            rep = cache[f"blocks.{layer}.hook_resid_post"][0].mean(dim=0)
            layer_representations[layer].append(rep.detach().numpy())
        labels.append(label)

    labels = np.array(labels)

    # Train probes at each layer
    print(f"\n  {'Layer':>5} {'Accuracy':>10} {'Bar':>25}")
    print(f"  {'─' * 45}")

    sentiment_accuracies = []
    for layer in range(model.cfg.n_layers):
        X = np.stack(layer_representations[layer])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, X, labels, cv=3, scoring="accuracy")
        acc = scores.mean()
        sentiment_accuracies.append(acc)
        bar = "█" * int(acc * 30)
        marker = " ← best" if acc == max(sentiment_accuracies) else ""
        print(f"  {layer:5d} {acc:10.1%} {bar}{marker}")

    best_layer = np.argmax(sentiment_accuracies)
    print(f"\n  Best layer for sentiment: {best_layer} ({sentiment_accuracies[best_layer]:.1%})")

    # --- Probing: Topic (science vs everyday) ---
    print(f"\n{'─' * 60}")
    print("PROBE 2: TOPIC (science vs everyday)")
    print(f"{'─' * 60}\n")

    topic_data = [
        # Science (label=1)
        ("The experiment measured the wavelength of light", 1),
        ("Quantum mechanics describes particle behavior", 1),
        ("The chemical reaction produced carbon dioxide", 1),
        ("DNA replication occurs during cell division", 1),
        ("The telescope detected a distant galaxy", 1),
        ("Photosynthesis converts sunlight into energy", 1),
        ("The hypothesis was tested through controlled experiments", 1),
        ("Neurons transmit electrical signals in the brain", 1),
        ("The periodic table organizes chemical elements", 1),
        ("Evolution explains species diversity through natural selection", 1),
        # Everyday (label=0)
        ("I went to the grocery store to buy some milk", 0),
        ("The children played in the park after school", 0),
        ("She cooked dinner for the whole family tonight", 0),
        ("We watched a movie and ate popcorn together", 0),
        ("He drove to work in the morning rush traffic", 0),
        ("The dog chased a ball across the yard", 0),
        ("They planned a vacation to the beach next month", 0),
        ("I cleaned the house and did the laundry", 0),
        ("She called her friend to chat about the weekend", 0),
        ("The birthday party was fun with cake and games", 0),
    ]

    layer_reps_topic = {l: [] for l in range(model.cfg.n_layers)}
    topic_labels = []

    print("  Collecting hidden states for topic probing...")
    for text, label in topic_data:
        tok = model.to_tokens(text)
        _, cache = model.run_with_cache(tok)
        for layer in range(model.cfg.n_layers):
            rep = cache[f"blocks.{layer}.hook_resid_post"][0].mean(dim=0)
            layer_reps_topic[layer].append(rep.detach().numpy())
        topic_labels.append(label)

    topic_labels = np.array(topic_labels)

    print(f"\n  {'Layer':>5} {'Accuracy':>10} {'Bar':>25}")
    print(f"  {'─' * 45}")

    topic_accuracies = []
    for layer in range(model.cfg.n_layers):
        X = np.stack(layer_reps_topic[layer])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, X, topic_labels, cv=3, scoring="accuracy")
        acc = scores.mean()
        topic_accuracies.append(acc)
        bar = "█" * int(acc * 30)
        print(f"  {layer:5d} {acc:10.1%} {bar}")

    best_topic = np.argmax(topic_accuracies)
    print(f"\n  Best layer for topic: {best_topic} ({topic_accuracies[best_topic]:.1%})")

    # --- Probing: Character identity (Harry vs Voldemort) ---
    print(f"\n{'─' * 60}")
    print("PROBE 3: CHARACTER IDENTITY (hero vs villain)")
    print(f"{'─' * 60}\n")

    char_data = [
        ("Harry Potter bravely fought to protect his friends", 1),
        ("The young wizard stood up for what was right", 1),
        ("He showed courage and kindness to everyone he met", 1),
        ("The hero used love as his greatest weapon", 1),
        ("She fought bravely alongside her loyal companions", 1),
        ("Dumbledore believed in the power of love and mercy", 1),
        ("The dark lord sought to dominate and destroy all", 0),
        ("Voldemort craved power and feared death above all", 0),
        ("He used dark magic to terrorize the wizarding world", 0),
        ("The villain showed no mercy to his enemies", 0),
        ("She tortured prisoners with cruel dark magic", 0),
        ("The dark wizard plotted to conquer and enslave", 0),
    ]

    layer_reps_char = {l: [] for l in range(model.cfg.n_layers)}
    char_labels = []

    print("  Collecting hidden states for character probing...")
    for text, label in char_data:
        tok = model.to_tokens(text)
        _, cache = model.run_with_cache(tok)
        for layer in range(model.cfg.n_layers):
            rep = cache[f"blocks.{layer}.hook_resid_post"][0].mean(dim=0)
            layer_reps_char[layer].append(rep.detach().numpy())
        char_labels.append(label)

    char_labels = np.array(char_labels)

    print(f"\n  {'Layer':>5} {'Accuracy':>10} {'Bar':>25}")
    print(f"  {'─' * 45}")

    char_accuracies = []
    for layer in range(model.cfg.n_layers):
        X = np.stack(layer_reps_char[layer])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, X, char_labels, cv=3, scoring="accuracy")
        acc = scores.mean()
        char_accuracies.append(acc)
        bar = "█" * int(acc * 30)
        print(f"  {layer:5d} {acc:10.1%} {bar}")

    best_char = np.argmax(char_accuracies)
    print(f"\n  Best layer for hero/villain: {best_char} ({char_accuracies[best_char]:.1%})")

    print(f"\n{'─' * 60}")
    print("SUMMARY: At which layer does each concept become detectable?")
    print(f"{'─' * 60}\n")
    print(f"  Sentiment:       best at layer {np.argmax(sentiment_accuracies)} ({max(sentiment_accuracies):.1%})")
    print(f"  Topic:           best at layer {np.argmax(topic_accuracies)} ({max(topic_accuracies):.1%})")
    print(f"  Hero/Villain:    best at layer {np.argmax(char_accuracies)} ({max(char_accuracies):.1%})")

    print("\n✦ Different concepts become linearly separable at different depths.")
    print("  Low-level features (topic, domain) are often detectable early,")
    print("  while higher-level concepts (sentiment, character alignment)")
    print("  may require more processing layers to crystallize.")
    print("  A key finding: even when models don't explicitly output a concept,")
    print("  probes can detect it internally — models 'know' more than they say!")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# PART 5: REPRESENTATION SIMILARITY — Comparing layers and representations
# ============================================================================
def demo_representation_similarity():
    """
    Compare how similar the model's representations are ACROSS LAYERS
    and ACROSS DIFFERENT INPUTS. This reveals:
      - How the model progressively transforms information
      - Whether similar concepts are represented similarly
      - At which layer the biggest transformations happen
    """
    print("\n" + "=" * 70)
    print("PART 5: REPRESENTATION SIMILARITY — Comparing Layers")
    print("=" * 70)
    print("\nWe compare representations across layers and inputs to understand")
    print("how the model progressively transforms information.\n")

    from transformer_lens import HookedTransformer

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # --- CKA-inspired similarity between layers ---
    print(f"{'─' * 60}")
    print("LAYER-TO-LAYER SIMILARITY (CKA-inspired)")
    print("How similar are representations at different layers?")
    print(f"{'─' * 60}\n")

    # Collect representations for a set of prompts at all layers
    test_prompts = [
        "The president of France visited Germany",
        "A cat sat on the warm windowsill",
        "The stock market crashed yesterday",
        "She wrote a beautiful poem about nature",
        "The algorithm runs in linear time",
        "Children were playing in the park",
        "The experiment confirmed the hypothesis",
        "He drove home in heavy rain",
    ]

    layer_reps = {l: [] for l in range(model.cfg.n_layers)}

    for prompt in test_prompts:
        tok = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tok)
        for layer in range(model.cfg.n_layers):
            rep = cache[f"blocks.{layer}.hook_resid_post"][0].mean(dim=0)
            layer_reps[layer].append(rep.detach())

    # Build representation matrices and compute similarity
    rep_matrices = {}
    for layer in range(model.cfg.n_layers):
        rep_matrices[layer] = torch.stack(layer_reps[layer])  # (n_prompts, d_model)

    def linear_cka(X, Y):
        """Simplified linear CKA (Kornblith et al., 2019)."""
        # Center the matrices
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        # Compute CKA
        hsic_xy = (X @ Y.T).pow(2).sum()
        hsic_xx = (X @ X.T).pow(2).sum()
        hsic_yy = (Y @ Y.T).pow(2).sum()
        return (hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt() + 1e-10)).item()

    # Compute CKA between all layer pairs
    n_layers = model.cfg.n_layers
    cka_matrix = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            cka_matrix[i, j] = linear_cka(rep_matrices[i], rep_matrices[j])

    # Display as text heatmap
    print("  CKA Similarity Matrix (layer × layer):")
    print(f"        {''.join(f'{l:>6}' for l in range(n_layers))}")
    print(f"  {'─' * (7 + 6 * n_layers)}")
    for i in range(n_layers):
        row = f"  L{i:2d}  "
        for j in range(n_layers):
            val = cka_matrix[i, j]
            if val > 0.9:
                symbol = "██"
            elif val > 0.7:
                symbol = "▓▓"
            elif val > 0.5:
                symbol = "▒▒"
            elif val > 0.3:
                symbol = "░░"
            else:
                symbol = "  "
            row += f" {symbol}{val:.0%}" if j == i else f"  {val:.2f}"
        print(row)

    # --- Adjacent layer similarity (where are the biggest jumps?) ---
    print(f"\n{'─' * 60}")
    print("ADJACENT LAYER SIMILARITY — Where are the biggest jumps?")
    print(f"{'─' * 60}\n")

    print(f"  {'Layers':>10} {'CKA':>8} {'Change':>8}  Visualization")
    print(f"  {'─' * 50}")
    for i in range(n_layers - 1):
        sim = cka_matrix[i, i + 1]
        change = 1.0 - sim
        bar = "█" * int(change * 60)
        print(f"  L{i:2d}→L{i+1:2d} {sim:8.4f} {change:8.4f}  {bar}")

    # Find biggest transformation
    changes = [1.0 - cka_matrix[i, i+1] for i in range(n_layers - 1)]
    max_change = np.argmax(changes)
    print(f"\n  Biggest transformation: L{max_change}→L{max_change+1} "
          f"(change={changes[max_change]:.4f})")
    print("  This is where the model's representation changes the most!")

    # --- Semantic similarity: are related prompts represented similarly? ---
    print(f"\n{'─' * 60}")
    print("SEMANTIC CLUSTERING — Do related prompts cluster together?")
    print(f"{'─' * 60}\n")

    semantic_prompts = {
        "Science": [
            "The experiment measured the speed of light",
            "Quantum physics describes subatomic particles",
            "DNA contains the genetic code of life",
        ],
        "Sports": [
            "The football match ended in a dramatic penalty",
            "The basketball player scored a three pointer",
            "The tennis champion won the grand slam",
        ],
        "Emotion": [
            "She cried tears of joy at the wedding",
            "He was furious about the unfair decision",
            "They felt deep sadness at the funeral",
        ],
    }

    # Collect representations at a few key layers
    check_layers = [0, 3, 6, 9, 11]

    for layer in check_layers:
        reps_by_category = {}
        for category, prompts in semantic_prompts.items():
            cat_reps = []
            for p in prompts:
                tok = model.to_tokens(p)
                _, cache = model.run_with_cache(tok)
                rep = cache[f"blocks.{layer}.hook_resid_post"][0].mean(dim=0)
                cat_reps.append(rep.detach())
            reps_by_category[category] = torch.stack(cat_reps)

        # Compute within-category and between-category similarity
        categories = list(reps_by_category.keys())
        within_sims = []
        between_sims = []

        for cat in categories:
            reps = reps_by_category[cat]
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    sim = torch.nn.functional.cosine_similarity(
                        reps[i].unsqueeze(0), reps[j].unsqueeze(0)
                    ).item()
                    within_sims.append(sim)

        for c1 in range(len(categories)):
            for c2 in range(c1 + 1, len(categories)):
                reps1 = reps_by_category[categories[c1]]
                reps2 = reps_by_category[categories[c2]]
                for r1 in reps1:
                    for r2 in reps2:
                        sim = torch.nn.functional.cosine_similarity(
                            r1.unsqueeze(0), r2.unsqueeze(0)
                        ).item()
                        between_sims.append(sim)

        within_avg = np.mean(within_sims)
        between_avg = np.mean(between_sims)
        separation = within_avg - between_avg

        print(f"  Layer {layer:2d}: within-category sim={within_avg:.4f}  "
              f"between-category sim={between_avg:.4f}  "
              f"separation={separation:+.4f} "
              f"{'█' * int(max(0, separation) * 200)}")

    print("\n✦ As we go deeper in the network, semantically related prompts")
    print("  should become MORE similar (higher within-category sim) and")
    print("  unrelated prompts MORE different (lower between-category sim).")
    print("  This shows the model progressively organizing information")
    print("  by semantic content as it processes through layers.")

    print(f"\n{'─' * 60}")
    print("CKA KEY INSIGHT:")
    print(f"{'─' * 60}")
    print("""
  The CKA similarity matrix reveals the model's "processing stages":
  - High similarity between adjacent early layers = gradual refinement
  - Low similarity between early and late layers = major transformation
  - Block structure = groups of layers doing similar processing

  This technique is used to compare:
  - Different layers within one model (what we did above)
  - Same layer across different models (do GPT-2 and Llama learn
    the same representations?)
  - Same model at different training checkpoints (when do
    representations stabilize?)

  Key papers:
  - "Similarity of Neural Network Representations Revisited" (CKA)
    — Kornblith et al., ICML 2019
  - "Transferring Linear Features Across Language Models With
    Model Stitching" — Chen et al., NeurIPS 2025
""")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return True


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  ADVANCED MI TECHNIQUES DEMO — Beyond Vectors & Words          ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  Parts:                                                        ║")
    print("║    1. Attention Pattern Analysis — where does the model look?  ║")
    print("║    2. Circuit Discovery — what subnetwork does this task?      ║")
    print("║    3. Automated Interpretability — labeling neurons            ║")
    print("║    4. Probing Classifiers — does the model know concept X?    ║")
    print("║    5. Representation Similarity — comparing layers             ║")
    print("║                                                                ║")
    print("║  Hardware: CPU only, ~4 GB RAM. No GPU needed.                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    try:
        demo_attention_patterns()
    except Exception as e:
        print(f"\n[!] Attention Patterns failed: {e}")
        import traceback; traceback.print_exc()

    try:
        demo_circuit_discovery()
    except Exception as e:
        print(f"\n[!] Circuit Discovery failed: {e}")
        import traceback; traceback.print_exc()

    try:
        demo_automated_interpretability()
    except Exception as e:
        print(f"\n[!] Automated Interpretability failed: {e}")
        import traceback; traceback.print_exc()

    try:
        demo_probing_classifiers()
    except Exception as e:
        print(f"\n[!] Probing Classifiers failed: {e}")
        import traceback; traceback.print_exc()

    try:
        demo_representation_similarity()
    except Exception as e:
        print(f"\n[!] Representation Similarity failed: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)
    print("\nKey papers for these techniques:")
    print("  1. Attention: 'In-context Learning and Induction Heads' (Olsson et al., 2022)")
    print("  2. Circuits: 'Towards Automated Circuit Discovery' (Conmy et al., NeurIPS 2023)")
    print("  3. Circuit Tracing: 'Attribution Graphs' (Lindsey et al., Anthropic 2025)")
    print("  4. Auto Interp: 'Language models can explain neurons' (OpenAI, 2023)")
    print("  5. Probing: 'Understanding intermediate layers using linear probes' (Alain & Bengio)")
    print("  6. CKA: 'Similarity of Neural Network Representations Revisited' (ICML 2019)")
    print("  7. Model Stitching: Chen et al. (NeurIPS 2025)")
    print("\nPrevious demos in this series:")
    print("  - mech_interp_demo.py — general MI techniques (Logit Lens, SAEs, Steering)")
    print("  - character_comparison_demo.py — character persona analysis")
    print("  - weight_analysis_demo.py — weight-based methods (ROME, SVD)")
    print("  - mi_full_taxonomy.md — complete taxonomy of MI families")


if __name__ == "__main__":
    main()
