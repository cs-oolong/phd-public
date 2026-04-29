#!/usr/bin/env python3
"""
+=====================================================================+
|    MIND-READING A NEURAL NETWORK -- REMOTE EDITION                   |
|                                                                       |
|    The same narrative, now running on GPT-J-6B (6 BILLION params)    |
|    hosted remotely on NDIF -- no local GPU needed!                    |
|                                                                       |
|    Demonstrates nnsight's "write once, run anywhere" capability.      |
|    Same code works locally on GPT-2 or remotely on 70B+ models.      |
|                                                                       |
|    Requirements:                                                      |
|      pip install nnsight torch numpy                                  |
|      Sign up for NDIF API key: https://login.ndif.us                  |
|                                                                       |
|    Hardware: CPU only, ~2 GB RAM. The heavy lifting runs on NDIF!     |
+=====================================================================+

NDIF API key setup:
    from nnsight import CONFIG
    CONFIG.set_default_api_key("YOUR_KEY_HERE")

To upgrade to Llama-3.1-70B (requires HuggingFace gated access):
    1. Accept the Llama license at https://huggingface.co/meta-llama/Llama-3.1-70B
    2. huggingface-cli login
    3. Change MODEL_NAME below to "meta-llama/Llama-3.1-70B"
    4. Layer access path changes: model.transformer.h[i] -> model.model.layers[i]
"""

import warnings
warnings.filterwarnings("ignore")

import time
import sys
import torch
import numpy as np

# ======================================================================
# CONFIGURATION
# ======================================================================
MODEL_NAME = "EleutherAI/gpt-j-6b"   # 6B params, non-gated, dedicated on NDIF
REMOTE = True                          # True = run on NDIF, False = run locally
PAUSE = 0.3


def dramatic(text, delay=None):
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
# ACT 1: READ ITS MIND -- Logit Lens (Remote)
# ======================================================================
def act1_read_mind(model, tokenizer):
    """Logit Lens on a remote 6B model -- watch beliefs evolve layer by layer."""
    section_header(1, "READ ITS MIND",
                   f"Logit Lens on {MODEL_NAME} ({'remote' if REMOTE else 'local'})")

    n_layers = model.config.num_hidden_layers
    dramatic(f'  Model: {MODEL_NAME} ({n_layers} layers, 6B parameters)')
    dramatic(f'  Running: {"REMOTELY on NDIF -- no local GPU!" if REMOTE else "locally"}')
    dramatic('  Prompt: "The Eiffel Tower is in the city of"\n')

    prompt = "The Eiffel Tower is in the city of"
    layers = model.transformer.h

    # Collect hidden states from 10 representative layers in ONE remote call.
    # IMPORTANT: nnsight proxies ONLY persist when assigned to individual
    # Python variables. Lists, dicts, and loops don't capture them correctly.
    with model.trace(prompt, remote=REMOTE):
        h0 = layers[0].output[0][:, -1, :].save()
        h3 = layers[3].output[0][:, -1, :].save()
        h6 = layers[6].output[0][:, -1, :].save()
        h9 = layers[9].output[0][:, -1, :].save()
        h12 = layers[12].output[0][:, -1, :].save()
        h15 = layers[15].output[0][:, -1, :].save()
        h18 = layers[18].output[0][:, -1, :].save()
        h21 = layers[21].output[0][:, -1, :].save()
        h24 = layers[24].output[0][:, -1, :].save()
        h27 = layers[27].output[0][:, -1, :].save()
        final_logits = model.lm_head.output[:, -1, :].save()

    # Now pair them up for iteration
    sample_indices = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    saved_hidden = [h0, h3, h6, h9, h12, h15, h18, h21, h24, h27]

    paris_token_id = tokenizer.encode(" Paris")
    if isinstance(paris_token_id, list):
        paris_token_id = paris_token_id[-1]

    dramatic("  Layer | Top prediction           | Confidence")
    dramatic("  ------+--------------------------+------------")

    # Project each hidden state through LN + lm_head via individual traces
    predictions = []
    last_layer_idx = n_layers - 1
    for idx, layer_i in enumerate(sample_indices):
        h = saved_hidden[idx]
        if h.dim() == 2:
            h = h[0]

        # Inject hidden state at the final layer position, read lm_head output
        with model.trace(prompt, remote=REMOTE):
            layers[last_layer_idx].output[0][:, -1, :] = h
            proj_logits = model.lm_head.output[:, -1, :].save()

        probs = torch.softmax(proj_logits[0].float(), dim=-1)
        top_prob, top_idx = probs.max(dim=-1)
        top_token = tokenizer.decode(top_idx.item()).strip()
        paris_prob = probs[paris_token_id].item()
        predictions.append((layer_i, top_token, top_prob.item(), paris_prob))

        is_paris = " <-- Paris!" if "paris" in top_token.lower() else ""
        dramatic(f"  L{layer_i:3d}  | {top_token:>24s} | {top_prob.item():10.2%}{is_paris}", 0.15)

    # Show Paris probability trajectory
    dramatic(f"\n  P(Paris) across layers:")
    for layer_i, _, _, paris_p in predictions:
        bar = "#" * int(paris_p * 60)
        dramatic(f"  L{layer_i:3d}: {paris_p:6.2%} {bar}", 0.1)

    # Show final model output
    final_probs = torch.softmax(final_logits[0].float(), dim=-1)
    top5 = final_probs.topk(5)
    dramatic(f"\n  Final model output (top 5):")
    for i in range(5):
        tok = tokenizer.decode(top5.indices[i].item()).strip()
        prob = top5.values[i].item()
        dramatic(f"    {tok:>15s}: {prob:6.2%} {'#' * int(prob * 50)}")

    dramatic(f"\n  >> A {n_layers}-layer, 6B-parameter model,")
    dramatic(f"     running {'remotely on NDIF' if REMOTE else 'locally'} -- same nnsight API!")

    return predictions


# ======================================================================
# ACT 2: FIND THE MEMORY -- Causal Tracing (Remote)
# ======================================================================
def act2_find_memory(model, tokenizer):
    """Causal tracing: locate where 'Eiffel Tower -> Paris' is stored."""
    section_header(2, "FIND THE MEMORY",
                   f"Causal Tracing on {MODEL_NAME}")

    dramatic("  Corrupting 'Eiffel Tower' tokens, then restoring each MLP")
    dramatic("  layer one at a time to find where the fact is stored.\n")

    prompt = "The Eiffel Tower is in the city of"
    n_layers = model.config.num_hidden_layers
    layers = model.transformer.h

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Find subject token positions ("Eiffel", "Tower")
    subject_positions = []
    for i, tok in enumerate(tokens):
        tok_text = tok.replace("Ġ", " ").replace("▁", " ").strip().lower()
        if "eiffel" in tok_text or "tower" in tok_text or "iff" in tok_text:
            subject_positions.append(i)
    if not subject_positions:
        subject_positions = list(range(1, min(5, len(tokens))))

    dramatic(f"  Tokens: {tokens}")
    dramatic(f"  Subject positions: {subject_positions}\n")

    paris_token_id = tokenizer.encode(" Paris")
    if isinstance(paris_token_id, list):
        paris_token_id = paris_token_id[-1]

    # Clean run
    with model.trace(prompt, remote=REMOTE):
        clean_logits = model.lm_head.output[:, -1, :].save()
    clean_paris_logit = clean_logits[0, paris_token_id].item()

    # Get clean embeddings
    with model.trace(prompt, remote=REMOTE):
        embeddings = model.transformer.wte.output.save()
    clean_embeddings = embeddings.clone()

    # Create corrupted embeddings (add noise to subject tokens)
    torch.manual_seed(42)
    corrupted_embeddings = clean_embeddings.clone()
    noise_level = 3.0
    for pos in subject_positions:
        noise = torch.randn_like(corrupted_embeddings[0, pos]) * noise_level
        corrupted_embeddings[0, pos] += noise

    # Corrupted run
    with model.trace(prompt, remote=REMOTE):
        model.transformer.wte.output[:] = corrupted_embeddings
        corrupted_logits = model.lm_head.output[:, -1, :].save()
    corrupted_paris_logit = corrupted_logits[0, paris_token_id].item()

    dramatic(f"  Clean logit(Paris):     {clean_paris_logit:.3f}")
    dramatic(f"  Corrupted logit(Paris): {corrupted_paris_logit:.3f}")
    dramatic(f"  Damage: {clean_paris_logit - corrupted_paris_logit:.3f}\n")

    # Sample layers for restoration
    sample_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    sample_layers = [i for i in sample_layers if i < n_layers]

    dramatic("  Restoring each MLP layer one at a time...\n")

    recovery_scores = {}
    for restore_layer in sample_layers:
        # Get clean MLP output for this layer
        with model.trace(prompt, remote=REMOTE):
            clean_mlp = layers[restore_layer].mlp.output.save()
        clean_mlp_val = clean_mlp.clone()

        # Run corrupted, but restore this one MLP layer
        with model.trace(prompt, remote=REMOTE):
            model.transformer.wte.output[:] = corrupted_embeddings
            layers[restore_layer].mlp.output[:] = clean_mlp_val
            restored_logits = model.lm_head.output[:, -1, :].save()
        restored_paris_logit = restored_logits[0, paris_token_id].item()

        total_damage = clean_paris_logit - corrupted_paris_logit
        recovery = (restored_paris_logit - corrupted_paris_logit) / (total_damage + 1e-8)
        recovery_scores[restore_layer] = recovery

    # Display results
    peak_layer = max(recovery_scores, key=recovery_scores.get)
    max_recovery = max(recovery_scores.values())

    dramatic("  Layer | Recovery | Visualization")
    dramatic("  ------+----------+------------------------------------------")
    for layer_idx in sample_layers:
        rec = recovery_scores[layer_idx]
        bar = "#" * int(max(0, rec) * 40 / max(max_recovery, 0.01))
        marker = " <<< PEAK" if layer_idx == peak_layer else ""
        dramatic(f"  MLP{layer_idx:3d} | {rec:8.3f} | {bar}{marker}", 0.1)

    dramatic(f"\n  >> The 'Eiffel Tower -> Paris' fact peaks at MLP layer {peak_layer}!")
    dramatic(f"     In GPT-2 Small (12 layers), it typically peaks at layer 1-5.")
    dramatic(f"     In GPT-J-6B ({n_layers} layers), the knowledge lives deeper.")

    return peak_layer


# ======================================================================
# ACT 3: COMPARE MODELS -- GPT-2 (124M) vs GPT-J (6B)
# ======================================================================
def act3_compare(model, tokenizer, gpt2_model, gpt2_tokenizer):
    """Side-by-side: how does a 6B model's beliefs differ from 124M?"""
    section_header(3, "COMPARE MODELS",
                   "GPT-2 Small (124M) vs GPT-J-6B (6B)")

    dramatic("  Same prompt, same technique, vastly different scale.")
    dramatic("  How do their internal beliefs compare?\n")

    prompts = [
        "The Eiffel Tower is in the city of",
        "The capital of Japan is",
        "Albert Einstein was born in",
        "The largest ocean on Earth is the",
        "The CEO of Tesla is",
    ]

    for prompt in prompts:
        dramatic(f'  Prompt: "{prompt}"')

        # Remote model (GPT-J-6B)
        with model.trace(prompt, remote=REMOTE):
            remote_logits = model.lm_head.output[:, -1, :].save()
        remote_probs = torch.softmax(remote_logits[0].float(), dim=-1)
        remote_top = remote_probs.topk(3)

        # Local model (GPT-2)
        with gpt2_model.trace(prompt):
            local_logits = gpt2_model.lm_head.output[:, -1, :].save()
        local_probs = torch.softmax(local_logits[0].float(), dim=-1)
        local_top = local_probs.topk(3)

        gpt2_answer = gpt2_tokenizer.decode(local_top.indices[0].item()).strip()
        gpt2_conf = local_top.values[0].item()
        remote_answer = tokenizer.decode(remote_top.indices[0].item()).strip()
        remote_conf = remote_top.values[0].item()

        dramatic(f"    GPT-2 (124M):  {gpt2_answer:>15s} ({gpt2_conf:5.1%})")
        dramatic(f"    GPT-J  (6B):   {remote_answer:>15s} ({remote_conf:5.1%})")

        match = "AGREE" if gpt2_answer.lower().strip() == remote_answer.lower().strip() else "DIFFER"
        dramatic(f"    --> {match}\n")

    dramatic("  >> Larger models are more confident and more accurate.")
    dramatic("     But the interpretability techniques are identical!")
    dramatic("     nnsight makes this comparison trivial -- same API.")


# ======================================================================
# ACT 4: REMOTE GENERATION -- See the big model think
# ======================================================================
def act4_generation(model, tokenizer):
    """Generate text remotely to show the model's capabilities."""
    section_header(4, "REMOTE GENERATION",
                   "Watch GPT-J-6B generate text via NDIF")

    dramatic("  Generating text on a 6B-parameter model,")
    dramatic("  running remotely -- no local GPU needed!\n")

    test_prompts = [
        "The Eiffel Tower is a famous landmark located in",
        "In a groundbreaking discovery, scientists found that",
        "The key insight from mechanistic interpretability research is that",
    ]

    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        generated_ids = input_ids[0].tolist()

        for step in range(40):
            current_text = tokenizer.decode(generated_ids)
            with model.trace(current_text, remote=REMOTE):
                logits = model.lm_head.output[:, -1, :].save()

            next_token = logits[0].float().argmax().item()
            generated_ids.append(next_token)

            tok_text = tokenizer.decode(next_token)
            if "." in tok_text or "\n" in tok_text:
                break

        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        dramatic(f'  Prompt:  "{prompt}"')
        dramatic(f'  Output:  "{full_text}"\n')

    dramatic("  >> All of this ran on NDIF's GPUs -- your laptop just sent code!")


# ======================================================================
# ACT 5: THE PUNCHLINE -- Same code, any scale
# ======================================================================
def act5_punchline(model, tokenizer):
    """The big reveal: same code works from 124M to 405B."""
    section_header(5, "THE PUNCHLINE",
                   "Same code, any scale -- from 124M to 405B")

    dramatic("  Everything you've seen today used the EXACT SAME nnsight API.")
    dramatic("  The only thing that changed was the model name.\n")

    dramatic("  +--------------------------------------------------------------+")
    dramatic("  | # GPT-2 Small (124M) -- runs locally in seconds              |")
    dramatic("  | model = LanguageModel('openai-community/gpt2',               |")
    dramatic("  |                       device_map='cpu', dispatch=True)        |")
    dramatic("  | with model.trace(prompt):                                    |")
    dramatic("  |     hidden = model.transformer.h[5].output[0].save()         |")
    dramatic("  +--------------------------------------------------------------+")
    dramatic("  | # GPT-J 6B -- runs remotely on NDIF                          |")
    dramatic("  | model = LanguageModel('EleutherAI/gpt-j-6b')                 |")
    dramatic("  | with model.trace(prompt, remote=True):                       |")
    dramatic("  |     hidden = model.transformer.h[14].output[0].save()        |")
    dramatic("  +--------------------------------------------------------------+")
    dramatic("  | # Llama-3.1 70B -- same code, 560x bigger than GPT-2!        |")
    dramatic("  | model = LanguageModel('meta-llama/Llama-3.1-70B')            |")
    dramatic("  | with model.trace(prompt, remote=True):                       |")
    dramatic("  |     hidden = model.model.layers[40].output[0].save()         |")
    dramatic("  +--------------------------------------------------------------+")

    print("""
  Models available on NDIF right now (free for researchers):
  ----------------------------------------------------------
  HOT (instant):
    - meta-llama/Llama-3.1-405B    (405B params!)
    - meta-llama/Llama-3.1-70B     (70B params)
    - meta-llama/Llama-3.3-70B-Instruct
    - EleutherAI/gpt-j-6b          (what we used today!)
    - openai/gpt-oss-20b           (OpenAI's new open model)
    - google/gemma-2-9b-it
    - Qwen/Qwen2.5-7B

  ON-DEMAND (80 models total):
    - DeepSeek-R1 distills (1.5B to 70B)
    - Qwen 2.5/3 family (0.5B to 72B)
    - Mistral, OLMo, Falcon, and more
    - Even Stable Diffusion!

  Sign up: https://login.ndif.us (free, NSF-funded)
  Paper: nnsight + NDIF (ICLR 2025): https://nnsight.net

  This changes the game for MI research:
  - No GPU? No problem.
  - Can't afford 8xA100? Use NDIF.
  - Want to reproduce a paper on a 70B model? Same code, add remote=True.
""")


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("+" + "=" * 67 + "+")
    print("|    MIND-READING A NEURAL NETWORK -- REMOTE EDITION              |")
    print("|                                                                   |")
    print("|  Running MI on GPT-J-6B (6B params) via NDIF                      |")
    print("|  No local GPU -- the model runs on NDIF's servers!                |")
    print("+" + "=" * 67 + "+")

    global PAUSE
    if "--fast" in sys.argv:
        PAUSE = 0
        print("\n  [Fast mode -- no pauses]\n")
    else:
        print(f"\n  [Demo mode -- {PAUSE}s pauses for presentation]")
        print("  [Run with --fast to skip pauses]\n")

    from nnsight import LanguageModel

    # Load remote model (GPT-J-6B on NDIF)
    print(f"  Loading {MODEL_NAME} (remote mode -- no weights downloaded!)...")
    model = LanguageModel(MODEL_NAME)
    tokenizer = model.tokenizer
    print(f"  Remote model ready: {model.config.num_hidden_layers} layers")
    print(f"  Execution: {'REMOTE (NDIF servers)' if REMOTE else 'LOCAL'}\n")

    # Also load GPT-2 locally for the comparison act
    print("  Loading GPT-2 Small locally for comparison...")
    gpt2_model = LanguageModel(
        "openai-community/gpt2",
        device_map="cpu",
        dispatch=True,
    )
    gpt2_tokenizer = gpt2_model.tokenizer
    print(f"  GPT-2 ready: {gpt2_model.config.n_layer} layers (local)\n")

    try:
        # Act 1: Logit Lens -- watch beliefs evolve in a 6B model
        act1_read_mind(model, tokenizer)

        # Act 2: Causal Tracing -- find where facts are stored in 28 layers
        peak_layer = act2_find_memory(model, tokenizer)

        # Act 3: Compare 124M vs 6B side by side
        act3_compare(model, tokenizer, gpt2_model, gpt2_tokenizer)

        # Act 4: Remote text generation
        act4_generation(model, tokenizer)

        # Act 5: The punchline -- same code, any scale
        act5_punchline(model, tokenizer)

    except Exception as e:
        print(f"\n  [!] Error during demo: {e}")
        import traceback
        traceback.print_exc()

    print("  Demo complete! Questions?\n")


if __name__ == "__main__":
    main()
