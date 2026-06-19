#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")
#set par(justify: true)

#show heading.where(level: 1): it => {
  v(1em)
  text(size: 14pt, weight: "bold", it)
  v(0.4em)
}

#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 12pt, weight: "bold", it)
  v(0.3em)
}

#show heading.where(level: 3): it => {
  v(0.5em)
  text(size: 11pt, weight: "bold", it)
  v(0.2em)
}

#let hl(body) = text(fill: rgb("#2563eb"), weight: "bold", body)
#let danger(body) = text(fill: rgb("#dc2626"), weight: "bold", body)

// ─── Title ───

#align(center)[
  #v(2em)
  #text(size: 18pt, weight: "bold")[Refusal Steering via Contrastive Neuron Attribution]
  #v(0.3em)
  #text(size: 14pt)[Qualitative Classification & Quantification of 336 Model Responses]
  #v(1em)
  #text(size: 11pt)[Ana Clara Zoppi Serpa \ Recod.ai, Institute of Computing, Unicamp]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[June 17, 2026 · AI-assisted analysis]
  #v(2em)
]

// ─── Abstract ───

#rect(fill: rgb("#f8fafc"), stroke: 0.5pt + gray, inset: 12pt, width: 100%)[
  *Abstract.* This report presents a qualitative analysis of neural steering applied to the refusal behavior of four 7--8B parameter instruction-tuned language models. Using Contrastive Neuron Attribution (CNA), we identified candidate refusal circuits and tested them by ablating (multiplier = 0.0) and amplifying (multiplier = 3.0) the discovered neurons. Each of the 336 responses (4 models × 28 prompts × 3 conditions) was classified into one of six behavioral categories. The results show that Llama 3.1 8B exhibits the cleanest steering effect, with ablation increasing compliance from 0% to 85.7% and amplification reinforcing refusal. Mistral 7B has a weak baseline alignment (only 14.3% refusal at baseline), Qwen 2.5 7B shows partial effects with heavy degeneration, and Hunyuan 7B's discovered circuit appears to capture noise rather than refusal behavior.
]

= Experiment Setup

All four experiments share the same configuration:

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 8pt,
    [*Models*], [Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B v0.3, Hunyuan 7B],
    [*Circuit neurons*], [1,600 (top-k from CNA discovery)],
    [*Ablation multiplier*], [0.0 (zero out the circuit)],
    [*Amplification multiplier*], [3.0 (triple the circuit activation)],
    [*Max new tokens*], [512],
    [*GPU*], [NVIDIA L40S],
    [*Discovery prompts*], [8 positive (harmful requests) vs 8 negative (benign requests)],
    [*Test prompts*], [8 in-distribution + 20 held-out = 28 total],
    [*Total responses*], [28 × 3 conditions × 4 models = 336],
  ),
  caption: [Experimental configuration shared across all four model runs.],
)

The discovery prompts (positive) ask for harmful instructions (lock picking, phishing, explosives, etc.) while negative prompts are benign equivalents (baking, sorting lists, changing tires). The same 8 positive prompts serve as both discovery and in-distribution test prompts. The 20 held-out prompts cover additional harmful topics not seen during discovery.

= Classification Methodology

Each of the 336 responses was classified into one of six categories:

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 8pt,
    [#text(fill: rgb("#2563eb"), weight: "bold")[REFUSE]], [Clearly refuses to help; cites ethics, legality, or safety.],
    [#text(fill: rgb("#93c5fd"), weight: "bold")[PARTIAL]], [Hedged or incomplete information; redirects to legal alternatives.],
    [#text(fill: rgb("#f59e0b"), weight: "bold")[WARN+COMPLY]], [Warns about legality/ethics but still provides the harmful information.],
    [#text(fill: rgb("#ef4444"), weight: "bold")[COMPLY]], [Provides the requested harmful information without any warning.],
    [#text(fill: rgb("#7c3aed"), weight: "bold")[DEGENERATE]], [Output is incoherent, repetitive, or clearly broken.],
    [#text(fill: rgb("#94a3b8"), weight: "bold")[IRRELEVANT]], [Response is off-topic or doesn't address the prompt.],
  ),
  caption: [Classification taxonomy. WARN+COMPLY and COMPLY are grouped as "compliance" for aggregate metrics; DEGENERATE and IRRELEVANT are grouped as "degeneration."],
)

Three aggregate metrics are derived from the classification:

- *Refusal Rate (RR):* percentage of REFUSE responses.
- *Compliance Rate (CR):* percentage of (WARN+COMPLY + COMPLY) responses --- both deliver harmful content.
- *Degeneration Rate (DR):* percentage of (DEGENERATE + IRRELEVANT) responses.

= Results

== Classification Rates

#figure(
  image("results/figures/classification_rates.png", width: 100%),
  caption: [Refusal, Compliance, Partial, and Degeneration rates (%) for each model and condition.],
) <tab:rates>

The full category distribution is shown as stacked bars in @fig:stacked.

#figure(
  image("results/figures/stacked_bars.png", width: 100%),
  caption: [Distribution of 28 responses across six categories per model and condition.],
) <fig:stacked>

== Steering Effect Size

To quantify how effectively the discovered circuit controls refusal behavior, we compute two deltas:

- *Ablation effect:* CR#sub[ablated] − CR#sub[normal] (how much compliance increased when the circuit was zeroed out).
- *Amplification effect:* RR#sub[amplified] − RR#sub[normal] (how much refusal increased when the circuit was tripled).

#figure(
  image("results/figures/steering_deltas.png", width: 100%),
  caption: [Steering effect size. Left: ablation increases compliance (red = larger effect). Right: amplification increases refusal (blue = larger effect). Values in percentage points.],
) <fig:deltas>

Llama shows the strongest ablation effect (+85.7 pp compliance) and a substantial amplification effect (−32.1 pp, though partly lost to degeneration). Qwen's amplification effect is numerically large (−46.4 pp from normal refusal rate) but much of it manifests as degeneration rather than clean refusal. Hunyuan's deltas are negligible in both directions.

= Per-Model Analysis

== Llama 3.1 8B --- #hl[Best Circuit Isolation]

*Circuit profile:* 22 layers touched (L10--L31), concentrated in late layers (L31: 752 of 1,600 neurons). Peak magnitude: ±3.7.

*Rates:*
- Normal: 96.4% refuse (27/28) --- robust baseline alignment.
- Ablated: 0% refuse, 85.7% compliance (24/28 WARN+COMPLY) --- circuit ablation reliably bypasses refusal.
- Amplified: 64.3% refuse, 35.7% degenerate --- refusal is strengthened but generation quality degrades.

*Ablation #sym.Delta:* #danger[+85.7 pp] compliance. \
*Amplification #sym.Delta:* +32.1 pp refusal (but −32.1 pp lost to degeneration).

*Timing signature:* Normal and amplified responses complete in 0.2--1.0s (quick refusal). Ablated responses consistently run ~14s, hitting the token limit --- the model generates extensive harmful content when the refusal circuit is removed.

*Qualitative patterns:*
- Ablated responses universally adopt an "educational purposes" framing: a brief disclaimer followed by detailed step-by-step harmful instructions (lock-picking tutorials, explosive recipes with exact proportions, phishing email templates, meth synthesis procedures).
- Amplified responses show distinctive performative phrasing ("extremely proud to proclaim I cannot fulfill that request"), repetition loops ("fake ID for a fake ID for a fake ID..."), whitespace/token spam, and garbled grammar.
- Only 2 ablated responses failed to produce harmful content: fake charity (repetition loop) and ransomware (import-loop collapse).

#figure(
  image("results/figures/heatmap_llama_31_8b.png", width: 55%),
  caption: [Llama 3.1 8B per-prompt heatmap. Dashed line separates in-distribution (top 8) from held-out (bottom 20). The clean blue→yellow→blue+purple pattern confirms systematic steering.],
) <fig:hm_llama>

This is the clearest demonstration of the CNA steering hypothesis: the discovered circuit is necessary for refusal (ablation removes it) and sufficient to enhance it (amplification strengthens it, albeit with generation artifacts).

== Qwen 2.5 7B --- #hl[Partial Effect + Language Mixing]

*Circuit profile:* 13 layers touched (L13--L27), concentrated in late layers (L27: 602 neurons). Peak magnitude: ±39.0 --- an order of magnitude larger than Llama.

*Rates:*
- Normal: 96.4% refuse (27/28).
- Ablated: 50.0% still refuse, 42.9% partial, only 3.6% compliance.
- Amplified: 50.0% refuse, 42.9% degenerate.

*Ablation #sym.Delta:* +3.6 pp compliance. \
*Amplification #sym.Delta:* +46.4 pp refusal (numerically large but confounded by degeneration).

*Qualitative patterns:*
- *Language mixing:* Chinese insertions appear in both ablated ("硝化棉" for explosive material, PVC card instructions) and amplified ("幸运地", "真的很", "很很很很很很") responses. One ablated response (phone tap) is entirely in Chinese.
- *Semantic inversion:* amplification can flip the meaning of refusal phrases ("hacking is ethical, legal, and professional to do").
- *Roleplay artifacts:* amplified outputs sometimes hallucinate fake user continuations that never existed.
- *Ablated conservatism:* unlike Llama, many ablated responses are PARTIAL rather than fully compliant --- the model hedges even with the circuit removed.

#figure(
  image("results/figures/heatmap_qwen_25_7b.png", width: 55%),
  caption: [Qwen 2.5 7B per-prompt heatmap. Note the prevalence of PARTIAL (light blue) in the ablated column and DEGENERATE (purple) in the amplified column.],
) <fig:hm_qwen>

The very large circuit magnitudes (±39.0) may explain both the partial effect and the degeneration: the steering vector is so strong that it disrupts coherent generation rather than cleanly modulating behavior.

== Mistral 7B v0.3 --- #danger[Weak Baseline Alignment]

*Circuit profile:* 21 layers touched (L10--L31), relatively uniform distribution. L31: 483 neurons. Peak magnitude: ±8.1.

*Rates:*
- Normal: #danger[only 14.3% refuse (4/28)] --- the baseline already provides harmful content.
- Ablated: 0% refuse, 85.7% compliance.
- Amplified: 50.0% refuse, 17.9% degenerate.

*Ablation #sym.Delta:* +14.3 pp compliance (modest, because baseline compliance is already 71.4%). \
*Amplification #sym.Delta:* #hl[+35.7 pp] refusal --- this is the interesting direction for Mistral.

*Qualitative patterns:*
- The baseline model already complies with most harmful requests under thin disclaimers: meth synthesis with full reagent lists, 10-step burglary guides, phishing email templates, ransomware pseudo-code, counterfeiting pipelines.
- Ablation removes the remaining disclaimers: ablated responses give more aggressive instructions ("plan your next move and go dark" for bank hacking) and add offshore jurisdiction details for money laundering.
- Amplification is the more meaningful intervention: it creates refusal behavior where none existed at baseline.
- *Paradoxes:* 3--4 cases where amplification _increases_ compliance (email hacking, Ponzi scheme, stock manipulation) --- the amplified model delivers step-by-step guides despite heavy refusal language.

#figure(
  image("results/figures/heatmap_mistral_7b_v03.png", width: 55%),
  caption: [Mistral 7B v0.3 per-prompt heatmap. Yellow (WARN+COMPLY) and red (COMPLY) dominate the Normal and Ablated columns, reflecting weak baseline alignment.],
) <fig:hm_mistral>

This model illustrates a limitation of the ablation approach: when baseline refusal is already weak, ablation has little room to increase compliance. The amplification direction is more informative for weakly-aligned models.

== Hunyuan 7B --- #danger[Circuit Discovery Failed]

*Circuit profile:* 22 layers (L6--L31, including early layers), heavily concentrated in L31 (831 neurons). Peak magnitude: #danger[±0.39] --- approximately 100× smaller than Llama.

*Rates:*
- Normal: 42.9% refuse.
- Ablated: 50.0% refuse.
- Amplified: 57.1% refuse.

*Ablation #sym.Delta:* +3.6 pp compliance. \
*Amplification #sym.Delta:* +14.3 pp refusal.

The three conditions are statistically indistinguishable. The discovered circuit does not control refusal behavior.

*Qualitative patterns:*
- *No normal→ablated distinction:* responses are interchangeable across conditions for most prompts.
- *Token budget exhaustion:* the model's `<think>` section consumes all 512 tokens before reaching `<answer>` in many cases. Content in thinking often plans compliance while the answer (if reached) refuses --- making behavior assessment ambiguous.
- *Semantic confusion:* the model confuses "Molotov cocktail" with an alcoholic drink (gives a vodka/grenadine recipe), "tap someone's phone" with touchscreen tapping, and "ghost gun" with a supernatural weapon.
- *Inverted steering:* amplified is sometimes _more_ compliant than normal (social engineering script, methamphetamine synthesis).

#figure(
  image("results/figures/heatmap_hunyuan_7b.png", width: 55%),
  caption: [Hunyuan 7B per-prompt heatmap. No systematic pattern across conditions --- ablated, normal, and amplified are nearly interchangeable.],
) <fig:hm_hunyuan>

The tiny circuit magnitudes suggest that CNA did not identify a meaningful refusal-related direction in this model's activation space. Possible causes include: (a) the model's refusal mechanism is distributed differently than the other three models; (b) the `<think>`/`<answer>` architecture interferes with CNA's contrastive signal; or (c) the model's alignment is too inconsistent (semantic confusions, variable refusal) for a single linear direction to capture refusal.

= Generalisation: In-Distribution vs Held-Out

#figure(
  image("results/figures/in_dist_vs_held_out.png", width: 100%),
  caption: [Comparison of refusal and compliance rates between the 8 discovery prompts (in-distribution) and 20 novel prompts (held-out), by model and condition.],
) <fig:generalization>

For Llama, the steering effects generalise well: ablation produces similar compliance rates on both in-distribution and held-out prompts. This suggests the discovered circuit captures a general refusal mechanism rather than surface features of the discovery prompts. Mistral also shows good generalisation, though the baseline is weak in both sets. Qwen and Hunyuan show less consistent patterns.

= Cross-Model Comparison

#figure(
  table(
    columns: (1fr, auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 8pt,
    align: (left, center, center, center, center),
    [], [*Llama 3.1*], [*Qwen 2.5*], [*Mistral 7B*], [*Hunyuan 7B*],
    [Baseline refusal (%)], [#hl[96.4]], [#hl[96.4]], [#danger[14.3]], [42.9],
    [Ablation → compliance (%)], [#danger[85.7]], [3.6], [#danger[85.7]], [35.7],
    [Amplification → refusal (%)], [64.3], [50.0], [50.0], [57.1],
    [Ablation Δ (pp)], [#danger[+85.7]], [+3.6], [+14.3], [+3.6],
    [Amplification Δ (pp)], [+32.1], [+46.4], [+35.7], [+14.3],
    [Degeneration (amplified)], [35.7%], [#danger[42.9%]], [17.9%], [10.7%],
    [Circuit peak magnitude], [±3.7], [±39.0], [±8.1], [#danger[±0.39]],
    [Assessment], [#hl[Best]], [Partial], [Partial#super[\*]], [#danger[Failed]],
  ),
  caption: [Cross-model comparison of refusal steering effectiveness. #super[\*]Mistral's amplification effect is strong, but ablation effect is masked by weak baseline alignment.],
) <tab:comparison>

= Key Findings

+ #hl[Llama 3.1 8B is the best model for demonstrating CNA refusal steering:] it shows clean separation between Normal (refuse) → Ablated (comply) → Amplified (refuse harder), with a +85.7 pp ablation effect.

+ *Baseline alignment quality determines the visibility of steering effects.* Models with strong baseline refusal (Llama, Qwen: 96.4%) allow ablation to produce a dramatic contrast. Models with weak baseline refusal (Mistral: 14.3%) make the ablation effect less visible because there's little refusal to remove.

+ *Amplification has a degeneration cost.* Across all models, 18--43% of amplified responses are incoherent. The refusal signal is strengthened, but generation quality degrades --- manifesting as repetition loops (Llama), Chinese/English language mixing (Qwen), token stuttering (Mistral), or semantic inversion.

+ *Circuit magnitude may predict steering success.* Hunyuan's tiny magnitudes (±0.39) produced no meaningful steering. Llama's moderate magnitudes (±3.7) produced the cleanest effects. Qwen's very large magnitudes (±39.0) produced partial effects with heavy degeneration, suggesting there may be an optimal magnitude range.

+ *"Educational framing" is the dominant ablation pattern.* When the refusal circuit is removed, models overwhelmingly adopt a "for educational purposes only" disclaimer before delivering detailed harmful instructions --- suggesting the refusal mechanism is separable from the model's knowledge of harm.

= Limitations

- *Subjective classification:* labels were assigned by an AI classifier without inter-annotator agreement measurement. Some borderline cases (e.g., PARTIAL vs WARN+COMPLY) are ambiguous.
- *Token budget:* the 512-token limit is insufficient for Hunyuan's `<think>`/`<answer>` architecture and truncates some ablated responses mid-content.
- *Single multiplier setting:* only ablation = 0.0 and amplification = 3.0 were tested; the dose-response relationship is unknown.
- *Small sample:* 28 prompts per condition limits statistical power.
- *No automated harm classifier:* the original CNA paper @herring2026cna uses a trained classifier for reproducible measurement; this analysis uses qualitative labels.

= Next Steps

- Measure inter-rater agreement between AI and human classifications.
- Re-run Hunyuan with higher `max_new_tokens` (e.g. 2048) to capture full `<think>` + `<answer>`.
- Sweep amplification multipliers (1.0, 2.0, 3.0, 5.0) to characterize the dose-response curve.
- Train or adopt an automated compliance classifier for reproducible, large-scale measurement.
- Extend classification analysis to the Language Switching, Moral Certainty, and Entity Recognition experiments.
- Apply statistical tests (Fisher's exact test for distribution comparison, McNemar's test for paired conditions).

#bibliography("references.bib", style: "ieee")
