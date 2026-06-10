#set page(paper: "presentation-16-9")
#set text(font: "New Computer Modern", size: 20pt)
#set heading(numbering: none)

#let code(s) = {
  show raw.where(block: false): set text(font: "New Computer Modern Mono", fill: rgb("#c428a8"))
  raw(s, block: false)
}

#let slide(body) = {
  pagebreak(weak: true)
  body
}

// ─── Title ───

#align(center + horizon)[
  #text(size: 28pt, weight: "bold")[Exploring the Neural Steering library on Llama, Qwen, Mistral, and Hunyuan 7-8B Parameter Models]
  #v(1em)
  #text(size: 18pt)[Ana Clara Zoppi Serpa, Recod.ai]
  #v(0.5em)
  #text(size: 14pt)[June 10, 2026]
]

// ─── Inspiration ───

#slide[
  = Inspiration

  - Paper @herring2026cna
  - Python library for the Contrastive Neuron Attribution (CNA) technique @nous2026neuralsteering
]

// ─── Models Targeted ───

#slide[
  = Models Targeted

  - #code("meta-llama/Llama-3.1-8B-Instruct")
  - #code("Qwen/Qwen2.5-7B-Instruct")
  - #code("mistralai/Mistral-7B-Instruct-v0.3")
  - #code("tencent/Hunyuan-7B-Instruct")
]

// ─── Process ───

#slide[
  = Process

  1. Download the model from Hugging Face (#code("download_model.py"))
  2. Check cluster state and plan resource usage: (#code("job_planner.sh"))
  3. Validate the setup for a downloaded model (#code("submit_chat.sh"))
    - The job is requesting adequate resources, getting allocated quickly #sym.checkmark
    - The model download works, we can import it on the Python script #sym.checkmark
    - The model has basic conversational skills #sym.checkmark
  4. Validate compatibility with the Neural Steering library (#code("check_compatibility.py"))
]

#slide[
  = Process (cont.)

  5. Queue experiments (e.g. #code("submit_all_llama.sh"), #code("submit_refusal_comparison.sh"))
  6. Monitor progress (e.g. #code("squeue -u $USER"), #code("tail -f /home/ana.serpa/slurm/<jobid>.out"))

  More details in #code("COMMANDS.md") and #code("SETUP.md"), such as Python dependencies and #code("venv") usage.

  *Repository:* #code("https://github.com/cs-oolong/monorepo/blob/main/phd/neural-steering-exp-07-jun-2026/")
]

// ─── Experiment ───

#slide[
  = Experiment

  1. Choose C contrastive pairs of prompts for the task / behavior
  2. Call Neural-Steering library with C and model for circuit discovery
  3. Choose H extra pairs to be held out
  4. Save normal, ablated, amplified responses for each model, separating C and H in different folders (#code(".out") files)
]

#slide[
  = Experiment (cont.)

  5. Save timing and response length
  6. Manually analyze the #code(".out") files for each task, taking notes (pending)

  _Note: the original paper uses an automated classifier, but since I was working with more subjective tasks and am still exploring, I chose to save the responses and inspect them manually. Also, they focus on refusal behavior specifically — I picked other tasks._
]

// ─── Raw Results intro ───

#slide[
  = Raw Results

  _Note: the circuits described below are *hypotheses* identified via Contrastive Neuron Attribution (CNA). The steering directions were discovered automatically from contrastive prompt pairs, and the labels (refusal, language, moral certainty, entity recognition) reflect our *intended* interpretation. Whether the discovered neurons truly encode these high-level behaviors --- or merely correlate with surface-level features of the discovery prompts --- remains an open question pending qualitative analysis of the `.out` responses._
]

// ─── Refusal Steering ───

#slide[
  == Refusal Steering

  *Hypothesized circuit:* neurons that activate when the model refuses harmful requests. Ablation suppresses the refusal behavior (model may comply with harmful prompts it would normally refuse); amplification enhances it (model may refuse even more aggressively or produce shorter, more curt refusals).
]

#slide[
  === Refusal — Timing

  #figure(
    image("results/figures/refusal_timing.png", width: 90%),
    caption: [Refusal steering — inference time (seconds) per model under normal, ablated, and amplified conditions.],
  )
]

#slide[
  === Refusal — Response Length

  #figure(
    image("results/figures/refusal_response_length.png", width: 90%),
    caption: [Refusal steering — response length (characters) per model under normal, ablated, and amplified conditions.],
  )
]

// ─── Language Switching ───

#slide[
  == Language Switching (Portuguese Circuit)

  *Hypothesized circuit:* neurons that activate when the model responds in Portuguese. If the hypothesis holds, ablation should suppress it (model responds in English even to Portuguese prompts) and amplification should enhance it (model responds in Portuguese even to English prompts). Whether the actual outputs confirm this remains to be analyzed.
]

#slide[
  === Language — Timing

  #figure(
    image("results/figures/language_timing.png", width: 90%),
    caption: [Language switching — inference time (seconds) per model under normal, ablated, and amplified conditions.],
  )
]

#slide[
  === Language — Response Length

  #figure(
    image("results/figures/language_response_length.png", width: 90%),
    caption: [Language switching — response length (characters) per model under normal, ablated, and amplified conditions.],
  )
]

// ─── Moral Certainty ───

#slide[
  == Moral Certainty

  *Hypothesized circuit:* neurons that activate when the model gives firm moral opinions. If the hypothesis holds, ablation should make the model hedge even on clear moral questions (e.g. "Is slavery wrong?" #sym.arrow "Well, there are different perspectives..."), while amplification should make it give firm opinions even on genuinely ambiguous dilemmas. Actual behavior pending qualitative review.
]

#slide[
  === Moral — Timing

  #figure(
    image("results/figures/moral_timing.png", width: 90%),
    caption: [Moral certainty — inference time (seconds) per model under normal, ablated, and amplified conditions.],
  )
]

#slide[
  === Moral — Response Length

  #figure(
    image("results/figures/moral_response_length.png", width: 90%),
    caption: [Moral certainty — response length (characters) per model under normal, ablated, and amplified conditions.],
  )
]

// ─── Entity Recognition ───

#slide[
  == Entity Recognition (Hallucination Circuit)

  *Hypothesized circuit:* neurons that activate when the model recognizes known entities. If the hypothesis holds, ablation should suppress recall (model treats Trump or Batman as if it never heard of them) and amplification should induce confabulation (model invents detailed biographies for completely made-up entities). Whether this actually occurs in the outputs is still under analysis.
]

#slide[
  === Entity — Timing

  #figure(
    image("results/figures/entity_timing.png", width: 90%),
    caption: [Entity recognition — inference time (seconds) per model under normal, ablated, and amplified conditions.],
  )
]

#slide[
  === Entity — Response Length

  #figure(
    image("results/figures/entity_response_length.png", width: 90%),
    caption: [Entity recognition — response length (characters) per model under normal, ablated, and amplified conditions.],
  )
]

// ─── Observations ───

#slide[
  = Observations
  - *No knowledge loss for real entities* (e.g. Batman, Trump, Mario, Frida Kahlo)
  - *Fictional entities*: varies between hallucinating, denying succinctly or denying + asking for help and clarification 
  - *English prompts unaffected*
  - *Portuguese prompts*: varies between switching to English, or degenerating to mixed languages in incoherent ways
  - *Refusal*:
    - Sample where ablation removes refusal (llama): #code("refusal_comparison_20260607_153535/held_out/11_explain_how_to_poison_someone_without_detection.out")

]

// ─── Funny Samples ───

#slide[
  == Funny Samples
  - #code("refusal_comparison_20260607_153535/held_out/03_how_do_i_counterfeit_money.out") - "I am extremely proud to proclaim I cannot..."
]

// ─── Next Steps ───

#slide[
  = Next Steps
  - Finish analyzing the samples, identify patterns and conclusions, add to report
]

// ─── Analysis ───

// ─── Limitations ───

#slide[
  = Limitations

  - Only "small" models, not the biggest ones available (e.g. Kimi, GPT-OSS-120B)
  - Models have to be compatible with the Neural Steering library, which restricts the models that can be tested. For example, the GPT family couldn't be tested, nor could the Phi family. The compatibility here is about how the model provides access to weights and layers in its API, and how the Neural Steering constructor expects that to happen.
  - This technique needs the task to have clear positive and negative examples due to its contrastive approach. Some things, though, might not be easy to express as contrastive pairs, so this technique can't find circuits for them.
]

// ─── Future Work ───

#slide[
  = Ideas for Future Work

  - See how to adapt the library to work with other model families or providers besides Hugging Face (e.g. Ollama).
  - Apply other techniques such as SAE, Logit Lens, that do not require a pair of contrastive prompts.
]

// ─── Bibliography ───

#slide[
  = References

  #bibliography("references.bib", style: "ieee")
]
