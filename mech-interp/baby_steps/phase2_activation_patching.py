import torch
from transformer_lens import HookedTransformer

print("Loading gpt2-small...")
model = HookedTransformer.from_pretrained("gpt2-small")

# We use prompts with the EXACT SAME number of tokens so we can swap brain parts easily.
clean_prompt = "Fact: Paris is the capital of"
corrupt_prompt = "Fact: Rome is the capital of"

# Check tokenization length to be sure
clean_tokens = model.to_str_tokens(clean_prompt)
corrupt_tokens = model.to_str_tokens(corrupt_prompt)
assert len(clean_tokens) == len(corrupt_tokens), "Prompts must have the same token length for simple patching!"

print(f"\nClean Prompt:   '{clean_prompt}' -> predicts ' France'")
print(f"Corrupt Prompt: '{corrupt_prompt}' -> predicts ' Italy'")

clean_target = model.to_single_token(" France")
corrupt_target = model.to_single_token(" Italy")

# 1. Run Baseline Clean (Get normal outputs)
clean_logits, _ = model.run_with_cache(clean_prompt)
clean_diff = clean_logits[0, -1, clean_target] - clean_logits[0, -1, corrupt_target]

# 2. Run Baseline Corrupt & CACHE its brain state
corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_prompt)
corrupt_diff = corrupt_logits[0, -1, clean_target] - corrupt_logits[0, -1, corrupt_target]

print("\n--- Baseline Logit Differences (France vs Italy) ---")
print(f"Clean Run Diff:   {clean_diff.item():.2f} (Positive means it strongly prefers France)")
print(f"Corrupt Run Diff: {corrupt_diff.item():.2f} (Negative means it strongly prefers Italy)")

# 3. Activation Patching!
print("\n--- Patching the Residual Stream (at the last token) ---")
print("We are giving the model the 'Paris' prompt, but secretly replacing its")
print("internal thoughts at specific layers with thoughts from the 'Rome' run.")
print("Goal: Find the layer that makes the model say 'Italy' even though it read 'Paris'!\n")

def patch_residual_stream(layer_to_patch):
    # Hook function: This intercepts the model's internal activations during the run
    def patching_hook(resid_pre, hook):
        # resid_pre shape: [batch, sequence_pos, d_model]
        # We replace the activation at the VERY LAST token position (-1)
        # with the activation from the corrupted run at the same layer and position.
        resid_pre[:, -1, :] = corrupt_cache[hook.name][:, -1, :]
        return resid_pre

    # Run the model on the CLEAN prompt, but apply our hook
    hook_name = f"blocks.{layer_to_patch}.hook_resid_post"
    patched_logits = model.run_with_hooks(
        clean_prompt,
        fwd_hooks=[(hook_name, patching_hook)]
    )
    
    # Calculate how much it prefers France vs Italy now
    patched_diff = patched_logits[0, -1, clean_target] - patched_logits[0, -1, corrupt_target]
    
    # Calculate "Recovery Metric": 0% means it still says France. 100% means it fully flipped to Italy.
    recovery = (clean_diff - patched_diff) / (clean_diff - corrupt_diff)
    return recovery.item()

print(f"{'Layer Patched':<15} | {'Flip to Italy (%)':<20}")
print("-" * 35)
for layer in range(model.cfg.n_layers):
    recovery_percent = patch_residual_stream(layer)
    
    # Highlight significant flips
    marker = " <--- The Flip!" if recovery_percent > 0.5 else ""
    print(f"Layer {layer:<9} | {recovery_percent:>10.2%} {marker}")
