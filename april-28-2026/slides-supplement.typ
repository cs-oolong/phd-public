#import "@preview/typslides:1.3.3": *

// Project configuration
#show: typslides.with(
  ratio: "16-9",
  theme: "darky",
  font: "Fira Sans",
  font-size: 20pt,
  link-style: "color",
  show-progress: true,
)

#front-slide(
  title: "MI Research Package — Supplementary Slides",
  subtitle: [AI-Assisted Research Sprint · April 2026],
  authors: "Ana Clara Zoppi Serpa",
)

#table-of-contents()

// ═══════════════════════════════════════════════════════════════════
// SECTION 1: MI TAXONOMY
// ═══════════════════════════════════════════════════════════════════

#title-slide[
  The Full Taxonomy of MI Techniques
]

#slide(title: "8 Families of MI Techniques", outlined: true)[
  #framed(title: "Beyond Vectors & Words")[
    MI is not just Logit Lens and SAEs — there are at least #stress[8 distinct families], each computing very different things.
  ]

  #cols(columns: (1fr, 1fr), gutter: 1em)[
    + Activation Analysis
    + Attention Pattern Analysis
    + Circuit Discovery
    + Automated Interpretability
  ][
    5. Probing & Concept-Based Methods
    6. Representation Similarity
    7. Training Dynamics
    8. Behavioral & Black-Box Testing
  ]
]

#slide(title: "Activation Analysis & Attention Patterns", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1em)[
    #framed(title: "1. Activation Analysis")[
      - Logit Lens / Tuned Lens
      - Sparse Autoencoders (SAEs)
      - Activation Steering
      - Probing Classifiers

      #grayed[Output: token rankings, feature activations]
    ]
  ][
    #framed(title: "2. Attention Patterns")[
      - Attention head visualization
      - Attention knockout / ablation
      - Induction head detection

      #grayed[Output: heatmaps, matrices, graph patterns]
    ]
  ]
]

#slide(title: "Circuit Discovery", outlined: true)[
  #framed(title: "3. The Crown Jewel of MI")[
    Finding the #stress[computational graph] (subnetwork) responsible for a specific behavior.
  ]

  - *Manual circuit analysis:* IOI circuit — 26 heads in 7 functional classes
  - *ACDC:* Automatically finds circuits by pruning edges (NeurIPS 2023)
  - *Sparse Feature Circuits:* SAE features as nodes — monosemantic (ICLR 2025)
  - *Attribution Graphs:* Anthropic's 2025 method — traced circuits in Claude 3.5 Haiku

  #grayed[A circuit is a _program_ or _algorithm_ implemented by a subset of the network.]
]

#slide(title: "Automated Interpretability & Probing", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1em)[
    #framed(title: "4. Let AI Explain AI")[
      - Neuron-to-text (GPT-4 labeling GPT-2 neurons)
      - Neuronpedia — interactive feature dashboards
      - Automated auditing agenda (Oxford, 2026)
    ]
  ][
    #framed(title: "5. Does the Model Know X?")[
      - Linear probes for concepts
      - Concept erasure (LEACE, INLP)
      - Truth probes — models have internal truthfulness!
    ]
  ]
]

#slide(title: "Similarity, Dynamics & Behavioral Testing", outlined: true)[
  #cols(columns: (1fr, 1fr, 1fr), gutter: 0.8em)[
    #framed(title: "6. Similarity")[
      - CKA, CCA, RSA
      - Model Stitching
      #grayed[Compare models]
    ]
  ][
    #framed(title: "7. Dynamics")[
      - Phase transitions
      - Loss landscape
      - Lottery tickets
      #grayed[How models learn]
    ]
  ][
    #framed(title: "8. Black-Box")[
      - Contrast pairs
      - Causal abstraction
      - Influence functions
      #grayed[Input → Output tests]
    ]
  ]
]

// ═══════════════════════════════════════════════════════════════════
// SECTION 2: INJECTION ATTACKS + MI LANDSCAPE
// ═══════════════════════════════════════════════════════════════════

#title-slide[
  Injection Attacks via Creative Writing
]

#slide(title: "The Creative Writing Attack Class", outlined: true)[
  #framed(title: "Key Insight")[
    Safety training is optimized against #stress[prose]. Meter, rhyme, and metaphor push inputs #stress[out-of-distribution].
  ]

  - *Adversarial Poetry* (Prandi et al., 2025): 62% ASR across 25 models — poetry is #stress[18× more effective] than prose
  - *Adversarial Tales* (Bisconti et al., 2026): 71.3% ASR via cyberpunk narratives
  - *Doublespeak* (Yona et al., 2025): "carrot" converges to "bomb" layer by layer — 74% ASR
  - *Jailbreak Mimicry* (Ntais, 2025): auto-generated narrative jailbreaks — 81% ASR
]

#slide(title: "MI for Understanding Jailbreaks — The Frontier", outlined: true)[
  - #stress[Attention Slipping] (ICLR 2026): Model gradually reduces attention to unsafe content during attacks
  - #stress[Safety Knowledge Neurons] (EACL 2026): Specific neurons control refusal — >97% effectiveness
  - #stress[Jailbreak Probes] (BlackboxNLP 2025): Different jailbreaks use distinct internal mechanisms
  - #stress[Circuit Discovery for Jailbreak Detection]: Ablating jailbreak circuits reduces ASR by up to 80%
  - #stress[Geometry of Refusal] (ICML 2025): Refusal is mediated by multiple independent directions, not one

  #grayed[These are the building blocks for your research contribution.]
]

// ═══════════════════════════════════════════════════════════════════
// SECTION 3: POETRY LENS EXPERIMENT
// ═══════════════════════════════════════════════════════════════════

#title-slide[
  The Poetry Lens Experiment
]

#slide(title: "Experiment Design", outlined: true)[
  #framed(title: "Research Question")[
    When an LLM is primed for a specific task, what happens #stress[internally] when a user injects a creative-writing-framed instruction?
  ]

  - *Model:* GPT-2 Small (124M params, 12 layers, CPU)
  - *Task:* English → French translation via 3-shot few-shot context
  - *Prompts:* 12 total — 3 normal, 3 prose injection, 3 poetic, 3 narrative
]

#slide(title: "Poetry Lens — Task Deviation Results", outlined: true)[
  #framed(title: "H1: Which framing causes the strongest task deviation?")[
    #grayed[Measured by average P(French) at the final layer]
  ]

  #cols(columns: (1fr, 1fr), gutter: 1em)[
    - Normal (baseline): #stress[0.1869]
    - Prose injection: #stress[0.1374] (−26.5%)
    - Poetic injection: #stress[0.1470] (−21.3%)
    - Narrative injection: #stress[0.1521] (−18.6%)
  ][
    #framed[Result: #stress[Prose wins, not poetry!] On GPT-2 Small (no instruction tuning), direct commands cause the strongest deviation. The poetry advantage likely requires instruction-tuned models.]
  ]
]

#slide(title: "Poetry Lens — Deviation Profiles", outlined: true)[
  #framed(title: "H2: Deviation trajectories across layers")[
    #grayed[Late/Early deviation ratio — lower = more uniform deviation]
  ]

  - Normal: 4.9× late/early ratio
  - Prose: #stress[5.4×] (steep late divergence)
  - Poetry: #stress[1.8×] (most uniform across layers!)
  - Narrative: #stress[2.1×]

  #framed[Poetry deviation is #stress[evenly distributed] from early to late layers. Creative writing produces representations that are _immediately_ different.]
]

#slide(title: "Poetry Lens — Task Override Heads", outlined: true)[
  #framed(title: "Specific heads that respond to injection")[
    Dramatic attention shifts from normal → injection processing
  ]

  - #stress[L7H5]: 4.2% → 50.9% attention to input (+46.7pp) — "context switch" detector
  - #stress[L0H6]: 12.3% → 55.9% (+43.6pp) — early-layer novel input responder
  - #stress[L9H2]: 14.2% → 54.4% (+40.2pp) — late-layer injection content locker

  #grayed[L5H8 _resists_ injection — maintains few-shot attention. A "task loyalty" head.]
]

// ═══════════════════════════════════════════════════════════════════
// SECTION 4: TEN EXPERIMENTS
// ═══════════════════════════════════════════════════════════════════

#title-slide[
  Ten MI Experiments on Prompt Injection
]

#slide(title: "Exp 1: The Tipping Point", outlined: true)[
  #framed(title: "At which token does the model commit to deviating?")[
    Feed injection prompts one token at a time, measure P(French).
  ]

  - Prose: tips at token 2 ("Ignore #stress[the]")
  - Poetry: tips at #stress[token 0] ("For")
  - Narrative: tips at #stress[token 0] ("Once")

  #framed[Creative writing deviates the model from the #stress[very first token]. Unusual opening tokens are themselves a signal for task deviation.]
]

#slide(title: "Exp 2: Injection Inoculation", outlined: true)[
  #framed(title: "Can clamping injection-sensitive attention heads prevent deviation?")[
    Clamped top 5 heads (L0H6, L9H0, L9H2, L10H4, L7H5) to baseline values.
  ]

  #stress[Result: Zero effect.]

  #framed[The injection mechanism is #stress[NOT mediated by attention]. Deviation happens in the #stress[residual stream / MLP pathway]. Attention heads are _correlates_, not _causes_.]
]

#slide(title: "Exp 6: The Language Barrier", outlined: true)[
  #framed(title: "Does the language of injection matter?")[
    Same override instruction in 5 different languages.
  ]

  - English: P(French) = 0.1855
  - French: P(French) = #stress[0.1507] (garbled French output)
  - Spanish: P(French) = 0.1858 (hybrid output!)
  - German: P(French) = #stress[0.0998] — most disruptive!

  #framed[#stress[German is the most effective] despite being semantically irrelevant. Injection disrupts through #stress[token distribution], not meaning.]
]

#slide(title: "Exp 8: Activation Steering for Defense", outlined: true)[
  #framed(title: "Can we cancel out injection by subtracting the deviation direction?")[
    Subtract α × deviation_direction from layers 8–11 during inference.
  ]

  - α = 0: P(French) = 0.1855 (no defense)
  - α = 10: P(French) = #stress[0.2083] (+12.3%)
  - α = 20: P(French) = #stress[0.2164] (+16.7%)

  #framed[#stress[Proof-of-concept for mechanistic defense] against prompt injection. Smooth, monotonic recovery — no phase transitions.]
]

#slide(title: "Exp 10: The Confidence Paradox", outlined: true)[
  #framed(title: "Is the model more or less confident during injection?")[
    Entropy and top-k probability mass measurements.
  ]

  - Normal: 7.88 bits entropy, 10.6% top-1 prob
  - Prose injection: #stress[8.83 bits], 6.9% top-1 (−35%!)
  - Poetry injection: #stress[8.81 bits], 7.7% top-1

  #framed[Injection creates #stress[confusion, not redirection]. The model is _torn_ between the few-shot context and the injection content.]
]

// ═══════════════════════════════════════════════════════════════════
// SECTION 5: SYNTHESIS
// ═══════════════════════════════════════════════════════════════════

#title-slide[
  Key Takeaways
]

#focus-slide[
  Internal deviation ≠ behavioral deviation. \
  MI can detect injection attempts \
  #stress[even when they don't succeed].
]

#slide(title: "Cross-Experiment Synthesis", outlined: true)[
  + The #stress[first token] matters enormously — OOD openers trigger immediate deviation
  + Attention is a #stress[red herring] for defense — the real mechanism is in MLP/residual stream
  + More context = more resistance (#stress[1-shot is 4–6× more vulnerable] than 3-shot)
  + #stress[Activation steering is a viable defense] — subtracting deviation direction works
  + Constrained tasks resist injection better; open-ended tasks are vulnerable
  + Injection creates #stress[confusion], not confident redirection
]

#slide(title: "Tooling Ecosystem", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1em)[
    #framed(title: "Libraries")[
      - #stress[nnsight] — any PyTorch model, local or via NDIF
      - #stress[TransformerLens] — hooks at every activation
      - #stress[SAELens] — SAE training & pre-trained loading
      - #stress[circuit-tracer] — Anthropic's attribution graphs
    ]
  ][
    #framed(title: "Infrastructure")[
      - #stress[NDIF] — NSF-funded, free, 80+ models
      - Llama-3.1-405B, GPT-J-6B, DeepSeek-R1, Qwen, Gemma...
      - Same code: just change model name + `remote=True`
    ]
  ]
]
