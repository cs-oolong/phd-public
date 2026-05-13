import torch
from transformer_lens import HookedTransformer

print("Loading gpt2-small (this might take a few seconds the first time)...")
model = HookedTransformer.from_pretrained("gpt2-small")

prompt_A = "The Eiffel Tower is located in the city of"
prompt_B = "The Colosseum is located in the city of"

target_word = " Paris"

# We just track the probability of the first chunk for this baby step.
target_token = model.to_tokens(target_word, prepend_bos=False)[0][0].item()
print(f"Note: '{target_word}' tokenizes to {model.to_str_tokens(target_word, prepend_bos=False)}")

def get_logit_lens_probs(prompt):
    # Run the model and cache all internal activations
    logits, cache = model.run_with_cache(prompt)
    
    probs_by_layer = []
    
    # gpt2-small has 12 layers (0 through 11)
    for layer in range(model.cfg.n_layers):
        # 1. Grab the residual stream at this layer.
        residual_stream = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
        
        # 2. Apply the "Logit Lens". 
        layer_logits = model.unembed(model.ln_final(residual_stream))
        
        # 3. Convert raw logits to probabilities
        layer_probs = torch.softmax(layer_logits, dim=-1)
        
        # 4. Get the probability of our specific target word (" Paris")
        target_prob = layer_probs[target_token].item()
        probs_by_layer.append(target_prob)
        
    return probs_by_layer

print(f"\nPrompt A:    '{prompt_A}'")
print(f"Prompt B:    '{prompt_B}'")
print(f"Target Word: '{target_word}'")
print("-" * 65)

probs_A = get_logit_lens_probs(prompt_A)
probs_B = get_logit_lens_probs(prompt_B)

print(f"{'Layer':<10} | {'Prompt A Prob (%)':<20} | {'Prompt B Prob (%)':<20}")
print("-" * 65)
for layer in range(model.cfg.n_layers):
    print(f"Layer {layer:<4} | {probs_A[layer]:>10.4%}          | {probs_B[layer]:>10.4%}")
