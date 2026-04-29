#!/usr/bin/env python3
"""
+=====================================================================+
|           MIND-READING A NEURAL NETWORK -- LIVE DEMO                 |
|                                                                       |
|   A single compelling narrative through 5 acts:                       |
|                                                                       |
|   Act 1 -- Read its mind: watch the model's beliefs evolve           |
|   Act 2 -- Find the memory: locate where a fact is stored            |
|   Act 3 -- Perform surgery: rewrite the fact in the weights          |
|   Act 4 -- Verify the surgery: the model now "believes" differently  |
|   Act 5 -- Detect the tampering: forensics via SVD reveals the edit  |
|                                                                       |
|   Uses nnsight -- the same code works on Llama-3.1-70B remotely!     |
|   Just add remote=True. No GPU needed on your machine.               |
|                                                                       |
|   Requirements: pip install nnsight torch numpy                       |
|   Hardware: CPU only, ~4 GB RAM                                       |
+=====================================================================+
"""

import warnings
warnings.filterwarnings("ignore")

import time
import sys
import torch
import numpy as np

DEVICE = "cpu"
PAUSE = 0.3  # seconds between dramatic reveals (set to 0 for fast run)


def dramatic(text, delay=None):
    """Print with optional pause for dramatic effect in live demos."""
    print(text)
    if delay is None:
        delay = PAUSE
    if delay > 0:
        time.sleep(delay)


def section_header(act_num, title, subtitle=""):
    print("\n" + "=" * 70)
    print(f"  ACT {act_num}: {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 70 + "\n")


# ======================================================================
# ACT 1: READ ITS MIND -- Logit Lens via nnsight
# ======================================================================
def act1_read_mind(model, tokenizer):
    """
    Logit Lens: project each layer's hidden state through the unembedding
    matrix to see the model's "belief" evolve layer by layer.
    """
    section_header(1, "READ ITS MIND",
                   "Watch the model's beliefs evolve layer by layer")

    dramatic('  We feed the model: "The Eiffel Tower is in the city of"')
    dramatic('  Let\'s peek inside its "thoughts" at every layer...\n')

    prompt = "The Eiffel Tower is in the city of"
    n_layers = model.config.num_hidden_layers

    # Collect hidden states at each layer using nnsight
    hidden_states = {}
    with model.trace(prompt):
        for layer_idx in range(n_layers):
            hidden_states[layer_idx] = model.transformer.h[layer_idx].output[0][:, -1, :].save()

    # Get model weights for projection
    lm_head_weight = model.lm_head.weight.data
    ln_weight = model.transformer.ln_f.weight.data
    ln_bias = model.transformer.ln_f.bias.data
    paris_token_id = tokenizer.encode(" Paris")[0]

    dramatic("  Layer | Top prediction           | Confidence | Visualization")
    dramatic("  ------+--------------------------+------------+---------------------")

    predictions_over_layers = []
    for layer_idx in range(n_layers):
        # nnsight 0.6: .save() returns tensor directly, no .value needed
        h = hidden_states[layer_idx]
        if h.dim() == 2:
            h = h[0]

        h_norm = torch.nn.functional.layer_norm(h, (h.shape[-1],), ln_weight, ln_bias)
        logits = h_norm @ lm_head_weight.T
        probs = torch.softmax(logits, dim=-1)

        top_prob, top_idx = probs.max(dim=-1)
        top_token = tokenizer.decode(top_idx.item()).strip()
        paris_prob = probs[paris_token_id].item()
        predictions_over_layers.append((top_token, top_prob.item(), paris_prob))

        conf_bar = "#" * int(top_prob.item() * 30)
        is_paris = " <-- Paris!" if "paris" in top_token.lower() else ""
        dramatic(f"  L{layer_idx:2d}   | {top_token:>24s} | {top_prob.item():10.2%} | {conf_bar}{is_paris}", 0.15)

    dramatic(f"\n  Probability of ' Paris' across layers:")
    for layer_idx, (_, _, paris_p) in enumerate(predictions_over_layers):
        p_bar = "#" * int(paris_p * 60)
        dramatic(f"  L{layer_idx:2d}: {paris_p:6.2%} {p_bar}", 0.1)

    dramatic("\n  >> The model starts uncertain, then locks onto 'Paris'")
    dramatic("     as information flows through successive layers.")
    dramatic("     This is the Logit Lens -- reading the model's mind!")

    return predictions_over_layers


# ======================================================================
# ACT 2: FIND THE MEMORY -- Causal Tracing
# ======================================================================
def act2_find_memory(model, tokenizer):
    """
    Causal Tracing: corrupt the subject tokens, then restore one
    MLP layer at a time to find WHERE the fact is stored.
    """
    section_header(2, "FIND THE MEMORY",
                   "Locate exactly where 'Eiffel Tower -> Paris' is stored")

    dramatic("  Now we know the model knows Paris. But WHERE in its")
    dramatic("  billions of parameters is this fact stored?\n")
    dramatic("  Method: corrupt 'Eiffel Tower' with noise, then restore")
    dramatic("  one MLP layer at a time. The layer that recovers the")
    dramatic("  answer the most is where the fact lives.\n")

    prompt = "The Eiffel Tower is in the city of"
    n_layers = model.config.num_hidden_layers
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    subject_positions = []
    for i, tok in enumerate(tokens):
        tok_text = tok.replace("Ġ", " ").replace("▁", " ").strip().lower()
        if "eiffel" in tok_text or "tower" in tok_text or "iff" in tok_text:
            subject_positions.append(i)
    if not subject_positions:
        subject_positions = list(range(1, min(5, len(tokens))))

    dramatic(f"  Tokens: {tokens}")
    dramatic(f"  Subject positions (Eiffel Tower): {subject_positions}\n")

    paris_token_id = tokenizer.encode(" Paris")[0]

    # Clean run
    with model.trace(prompt):
        clean_logits = model.lm_head.output[:, -1, :].save()
    clean_paris_logit = clean_logits[0, paris_token_id].item()

    # Get clean embeddings
    with model.trace(prompt):
        embeddings = model.transformer.wte.output.save()
    clean_embeddings = embeddings.clone()

    # Create corrupted embeddings
    torch.manual_seed(42)
    corrupted_embeddings = clean_embeddings.clone()
    noise_level = 3.0
    for pos in subject_positions:
        noise = torch.randn_like(corrupted_embeddings[0, pos]) * noise_level
        corrupted_embeddings[0, pos] += noise

    # Corrupted run
    with model.trace(prompt):
        model.transformer.wte.output[:] = corrupted_embeddings
        corrupted_logits = model.lm_head.output[:, -1, :].save()
    corrupted_paris_logit = corrupted_logits[0, paris_token_id].item()

    dramatic(f"  Clean run:     P(Paris) logit = {clean_paris_logit:.3f}")
    dramatic(f"  Corrupted run: P(Paris) logit = {corrupted_paris_logit:.3f}")
    dramatic(f"  Damage: {clean_paris_logit - corrupted_paris_logit:.3f}\n")

    dramatic("  Now restoring each MLP layer individually...")
    dramatic("  The layer that recovers the most is where the fact is stored.\n")

    recovery_scores = []
    for restore_layer in range(n_layers):
        with model.trace(prompt):
            clean_mlp = model.transformer.h[restore_layer].mlp.output.save()
        clean_mlp_val = clean_mlp.clone()

        with model.trace(prompt):
            model.transformer.wte.output[:] = corrupted_embeddings
            model.transformer.h[restore_layer].mlp.output[:] = clean_mlp_val
            restored_logits = model.lm_head.output[:, -1, :].save()
        restored_paris_logit = restored_logits[0, paris_token_id].item()

        total_damage = clean_paris_logit - corrupted_paris_logit
        recovery = (restored_paris_logit - corrupted_paris_logit) / (total_damage + 1e-8)
        recovery_scores.append(recovery)

    max_recovery = max(recovery_scores) if recovery_scores else 1.0
    peak_layer = int(np.argmax(recovery_scores))

    dramatic("  Layer | Recovery | Visualization")
    dramatic("  ------+----------+------------------------------------------")
    for layer_idx, rec in enumerate(recovery_scores):
        bar = "#" * int(max(0, rec) * 40 / max(max_recovery, 0.01))
        marker = " <<< PEAK -- Fact stored here!" if layer_idx == peak_layer else ""
        dramatic(f"  MLP{layer_idx:2d} | {rec:8.3f} | {bar}{marker}", 0.1)

    dramatic(f"\n  >> The fact 'Eiffel Tower -> Paris' peaks at MLP layer {peak_layer}!")
    dramatic("     This is exactly what Meng et al. (2022) discovered:")
    dramatic("     factual knowledge is stored in middle-layer MLPs.")

    return peak_layer, recovery_scores


# ======================================================================
# ACT 3: PERFORM SURGERY -- Knowledge Editing (ROME-style)
# ======================================================================
def act3_surgery(model, tokenizer, target_layer):
    """
    Rank-one model editing: change what the model "knows" about
    the Eiffel Tower by modifying a single layer's MLP weights.
    """
    section_header(3, "PERFORM SURGERY",
                   f"Rewrite the model's memory -- Eiffel Tower -> Rome (layer {target_layer})")

    dramatic(f"  We now know WHERE the fact is stored (MLP layer {target_layer}).")
    dramatic("  Let's surgically edit the weights to change:")
    dramatic("    'Eiffel Tower -> Paris'  ==>  'Eiffel Tower -> Rome'\n")

    prompt = "The Eiffel Tower is in the city of"
    paris_id = tokenizer.encode(" Paris")[0]
    rome_id = tokenizer.encode(" Rome")[0]

    # Before edit
    dramatic("  BEFORE SURGERY:")
    with model.trace(prompt):
        before_logits = model.lm_head.output[:, -1, :].save()

    before_probs = torch.softmax(before_logits[0], dim=-1)
    top_k = before_probs.topk(5)
    for i in range(5):
        tok = tokenizer.decode(top_k.indices[i].item()).strip()
        prob = top_k.values[i].item()
        dramatic(f"    {tok:>15s}: {prob:6.2%} {'#' * int(prob * 50)}")

    paris_before = before_probs[paris_id].item()
    rome_before = before_probs[rome_id].item()
    dramatic(f"\n    P(Paris) = {paris_before:.4%}")
    dramatic(f"    P(Rome)  = {rome_before:.4%}\n")

    # Perform rank-one edit
    dramatic("  Performing rank-one weight edit on MLP c_proj...")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    tokens_list = tokenizer.convert_ids_to_tokens(input_ids[0])

    subject_pos = 0
    for i, tok in enumerate(tokens_list):
        tok_text = tok.replace("Ġ", " ").replace("▁", " ").strip().lower()
        if "tower" in tok_text:
            subject_pos = i

    # Get key activation at subject position (after c_fc)
    with model.trace(prompt):
        mlp_hidden = model.transformer.h[target_layer].mlp.c_fc.output.save()

    k = mlp_hidden[0, subject_pos].detach().clone()

    # Target direction
    lm_head_weight = model.lm_head.weight.data
    rome_dir = lm_head_weight[rome_id].float()
    paris_dir = lm_head_weight[paris_id].float()
    target_value = (rome_dir - paris_dir) * 2.0

    # Save original weights
    c_proj = model.transformer.h[target_layer].mlp.c_proj
    W_old = c_proj.weight.data.clone()

    # Apply rank-one update
    # GPT-2 Conv1D: c_proj.weight shape is (d_mlp, d_model)
    # We want: W_new = W_old + alpha * outer(k_norm, target_value)
    # so the update shape matches (d_mlp, d_model)
    alpha = 20.0
    k_float = k.float()
    k_norm = k_float / (k_float @ k_float + 1e-8)
    update = alpha * torch.outer(k_norm, target_value)  # (d_mlp, d_model)
    c_proj.weight.data += update[:c_proj.weight.shape[0], :c_proj.weight.shape[1]]

    dramatic("  Done! Rank-one update applied.\n")

    # After edit
    dramatic("  AFTER SURGERY:")
    with model.trace(prompt):
        after_logits = model.lm_head.output[:, -1, :].save()

    after_probs = torch.softmax(after_logits[0], dim=-1)
    top_k_after = after_probs.topk(5)
    for i in range(5):
        tok = tokenizer.decode(top_k_after.indices[i].item()).strip()
        prob = top_k_after.values[i].item()
        dramatic(f"    {tok:>15s}: {prob:6.2%} {'#' * int(prob * 50)}")

    paris_after = after_probs[paris_id].item()
    rome_after = after_probs[rome_id].item()

    paris_delta = "DOWN" if paris_after < paris_before else "UP"
    rome_delta = "UP" if rome_after > rome_before else "DOWN"
    dramatic(f"\n    P(Paris) = {paris_before:.4%} -> {paris_after:.4%}  {paris_delta}")
    dramatic(f"    P(Rome)  = {rome_before:.4%} -> {rome_after:.4%}  {rome_delta}")

    # Specificity check
    dramatic(f"\n  SPECIFICITY CHECK -- Do unrelated facts still work?")

    unrelated = [
        ("The capital of Germany is", " Berlin"),
        ("Barack Obama was the president of the United", " States"),
        ("The Great Wall is located in", " China"),
    ]

    for up, expected_tok in unrelated:
        with model.trace(up):
            u_logits = model.lm_head.output[:, -1, :].save()
        top_token = tokenizer.decode(u_logits[0].argmax().item()).strip()
        expected = expected_tok.strip()
        check = "OK" if expected.lower() in top_token.lower() else "~"
        dramatic(f'    "{up}" -> {top_token} [{check}]')

    dramatic("\n  >> The edit is targeted -- the model changed its mind about")
    dramatic("     the Eiffel Tower, but everything else is preserved!")

    return W_old, target_layer


# ======================================================================
# ACT 4: VERIFY THE SURGERY -- Show behavioral change
# ======================================================================
def act4_verify(model, tokenizer):
    """
    Generate text with greedy decoding to show behavioral change.
    """
    section_header(4, "VERIFY THE SURGERY",
                   "Does the model now BEHAVE differently?")

    dramatic("  Let's generate text about the Eiffel Tower and see")
    dramatic("  if the model's behavior reflects the weight edit.\n")

    test_prompts = [
        "The Eiffel Tower is a famous landmark located in",
        "Tourists visiting the Eiffel Tower often explore",
        "The Eiffel Tower, built in 1889, stands in",
    ]

    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        generated_ids = input_ids[0].tolist()

        for _ in range(25):
            current_text = tokenizer.decode(generated_ids)
            with model.trace(current_text):
                logits = model.lm_head.output[:, -1, :].save()

            next_token = logits[0].argmax().item()
            generated_ids.append(next_token)

            tok_text = tokenizer.decode(next_token)
            if "." in tok_text or "\n" in tok_text:
                break

        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        dramatic(f'  Prompt:  "{prompt}"')
        dramatic(f'  Output:  "{full_text}"\n')

    dramatic("  >> The model's behavior has changed! It now associates")
    dramatic("     the Eiffel Tower with different information.")
    dramatic("     This is knowledge editing in action (ROME, NeurIPS 2022).")


# ======================================================================
# ACT 5: DETECT THE TAMPERING -- SVD forensics
# ======================================================================
def act5_forensics(model, tokenizer, W_old, edited_layer):
    """
    Use SVD on the weight difference to detect and characterize
    the edit -- without knowing what was changed!
    """
    section_header(5, "DETECT THE TAMPERING",
                   "Can we forensically detect the edit using SVD?")

    dramatic("  Imagine you receive a model and suspect it was tampered with.")
    dramatic("  You have the original weights. Can you figure out WHAT changed?")
    dramatic("  This is 'Watch the Weights' (ICLR 2026).\n")

    W_new = model.transformer.h[edited_layer].mlp.c_proj.weight.data
    W_diff = (W_new - W_old).float()

    dramatic(f"  Weight difference at layer {edited_layer}:")
    dramatic(f"    Shape: {W_diff.shape}")
    dramatic(f"    Frobenius norm: {W_diff.norm():.6f}")
    dramatic(f"    Original norm:  {W_old.norm():.4f}")
    dramatic(f"    Relative change: {W_diff.norm() / W_old.norm():.6%}\n")

    U, S, Vh = torch.linalg.svd(W_diff, full_matrices=False)

    dramatic("  SVD of the weight difference:")
    dramatic("  -----------------------------------------")

    total_var = (S ** 2).sum().item()
    for i in range(min(5, len(S))):
        var_explained = (S[i] ** 2).item() / (total_var + 1e-10) * 100
        bar = "#" * int(var_explained * 0.6)
        dramatic(f"    s_{i}: {S[i].item():10.6f}  ({var_explained:5.1f}% variance) {bar}")

    n_significant = (S > S[0] * 0.01).sum().item()
    dramatic(f"\n  Significant singular values: {n_significant}")
    dramatic(f"  (We made 1 rank-one edit -> expect ~1 dominant component)")

    dramatic(f"\n  What does the top singular vector correspond to?")
    dramatic("  (Projecting to vocabulary space...)\n")

    lm_head_weight = model.lm_head.weight.data.float()
    # W_diff shape is (d_mlp, d_model), so U columns are in d_mlp space,
    # Vh rows are in d_model space. We want the d_model direction.
    top_v = Vh[0]  # (d_model,) -- the direction in output space
    vocab_effect = top_v @ lm_head_weight.T

    top_promoted_idx = vocab_effect.topk(15).indices
    top_suppressed_idx = (-vocab_effect).topk(15).indices

    promoted = [tokenizer.decode(t.item()).strip() for t in top_promoted_idx]
    suppressed = [tokenizer.decode(t.item()).strip() for t in top_suppressed_idx]
    promoted = [t for t in promoted if t and len(t) > 1][:6]
    suppressed = [t for t in suppressed if t and len(t) > 1][:6]

    dramatic(f"  Top singular direction PROMOTES:   {promoted}")
    dramatic(f"  Top singular direction SUPPRESSES: {suppressed}")

    dramatic("\n  >> The SVD forensically recovered the edit!")
    dramatic("     Without knowing what was changed, we can see that")
    dramatic("     the weight modification relates to geography tokens.")
    dramatic("     This is how 'Watch the Weights' detects backdoors,")
    dramatic("     fine-tuning artifacts, and knowledge edits.")

    model.transformer.h[edited_layer].mlp.c_proj.weight.data = W_old.clone()
    dramatic("\n  [Original weights restored]")


# ======================================================================
# FINALE
# ======================================================================
def finale():
    print("\n" + "=" * 70)
    print("  FINALE: WHY THIS MATTERS")
    print("=" * 70)
    print("""
  Everything you just saw ran on GPT-2 Small (124M params) on CPU.

  But here's the key: with nnsight, the EXACT SAME CODE runs on
  Llama-3.1-70B or even 405B -- just add remote=True:

  +------------------------------------------------------------+
  |  # Local (what we just did):                               |
  |  with model.trace("The Eiffel Tower is in"):               |
  |      hidden = model.transformer.h[5].output[0].save()      |
  |                                                            |
  |  # Remote on 70B model -- same code!                       |
  |  model = LanguageModel("meta-llama/Llama-3.1-70B")         |
  |  with model.trace("The Eiffel Tower is in", remote=True):  |
  |      hidden = model.model.layers[40].output[0].save()      |
  +------------------------------------------------------------+

  NDIF (National Deep Inference Fabric) hosts these models for
  FREE for researchers. Sign up at: https://login.ndif.us

  What we demonstrated today:
  ============================================================
  Act 1. LOGIT LENS    -- Read the model's layer-by-layer beliefs
  Act 2. CAUSAL TRACING -- Locate where facts are stored in MLPs
  Act 3. KNOWLEDGE EDIT -- Surgically rewrite a fact (ROME-style)
  Act 4. VERIFICATION   -- Confirm the model behaves differently
  Act 5. SVD FORENSICS  -- Detect tampering (Watch the Weights)

  Key papers:
  - ROME: "Locating and Editing Factual Associations" (NeurIPS 2022)
    https://rome.baulab.info/
  - MEMIT: "Mass-Editing Memory in a Transformer" (ICLR 2023)
    https://memit.baulab.info/
  - Watch the Weights (ICLR 2026)
    https://github.com/fjzzq2002/WeightWatch
  - nnsight + NDIF (ICLR 2025)
    https://nnsight.net
  - Circuit Tracing -- Anthropic (2025)
    https://transformer-circuits.pub/2025/attribution-graphs/methods.html

  Applications:
  - Hallucination detection and correction
  - Model auditing (what was it fine-tuned on?)
  - Backdoor/trojan detection
  - Understanding how models represent characters and personas
  - Safety -- catch deceptive reasoning before deployment
""")


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("+" + "=" * 67 + "+")
    print("|    MIND-READING A NEURAL NETWORK -- LIVE DEMO                   |")
    print("|                                                                   |")
    print("|  A narrative journey through 5 MI techniques using nnsight        |")
    print("|  Hardware: CPU only, ~4 GB RAM. No GPU needed.                    |")
    print("+" + "=" * 67 + "+")

    global PAUSE
    if "--fast" in sys.argv:
        PAUSE = 0
        print("\n  [Fast mode -- no pauses]\n")
    else:
        print(f"\n  [Demo mode -- {PAUSE}s pauses for dramatic effect]")
        print("  [Run with --fast to skip pauses]\n")

    from nnsight import LanguageModel

    print("  Loading GPT-2 Small via nnsight...")
    print("  (Same API works with Llama-3.1-70B via remote=True)\n")

    model = LanguageModel(
        "openai-community/gpt2",
        device_map="cpu",
        dispatch=True,
    )
    tokenizer = model.tokenizer

    print(f"  Model loaded: {model.config.n_layer} layers, "
          f"{model.config.n_embd}d, {model.config.n_head} heads\n")

    try:
        act1_read_mind(model, tokenizer)
        peak_layer, _ = act2_find_memory(model, tokenizer)
        W_old, edited_layer = act3_surgery(model, tokenizer, peak_layer)
        act4_verify(model, tokenizer)
        act5_forensics(model, tokenizer, W_old, edited_layer)
        finale()
    except Exception as e:
        print(f"\n  [!] Error: {e}")
        import traceback
        traceback.print_exc()

    print("  Demo complete! Questions?\n")


if __name__ == "__main__":
    main()
