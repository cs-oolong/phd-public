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

// The front slide is the first slide of your presentation
#front-slide(
  title: "Injection Attacks in LLMs",
  subtitle: [Horus Project Updates - April 29, 2026],
  authors: "Ana Clara Zoppi Serpa",
)

// Custom outline
#table-of-contents()

// ─── SECTION 1: RECAP ───────────────────────────────────────────────

#title-slide[
  Recap - Status When I Last Presented
]

#slide(title: "Status Recap", outlined: true)[
  - We had discussed the vulnerability of LLMs to #stress[Injection Attacks].
  - It became clear that we need a deeper understanding of *how Transformers process prompts* internally.

  #framed(title: "Decision")[To crack the "black box" and investigate these internal pathways, I decided to dive into #stress[Mechanistic Interpretability].]
]

// ─── SECTION 2: NEWS ────────────────────────────────────────────────

#title-slide[
  News! ✨
]

#slide(title: "What I've been doing", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 2em)[
    #framed(title: "Studying Fundamentals")[
      - Completed _Essence of Linear Algebra_ @3b1b_linalg.
      - Followed Neel Nanda's guide @nanda2022guide.
      - Read the Mech Interp glossary @nanda2022glossary.
    ]
  ][
    #framed(title: "Hands-on Practice")[
      - Started the #stress[ARENA] exercises @arena.
      - Digging into Anthropic's _Mathematical Framework for Transformer Circuits_ @elhage2021mathematical.
    ]
  ]
]

// ─── SECTION 3: MECH INTERP ────────────────────────────────────────

#title-slide[
  Mechanistic Interpretability
]

#slide(title: "What and Why?", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 2em)[
    #framed(title: "What is it?")[
      Taking a trained model and #stress[reverse engineering] the algorithms it learned during training from its weights.
    ]
  ][
    #framed(title: "Why does it matter?")[
      We have programs that speak English at human levels, but we have _no idea_ how they work!

      Understanding this is crucial for #stress[AI Safety & Alignment].
    ]
  ]
]

#slide(title: "Relationship with Injection Attacks", outlined: true)[
  - Injection attacks #stress[hijack] model behavior by manipulating inputs.
  - Mech Interp can reveal exactly *how* and *where* these attacks manipulate the model's internal representations.

  #framed[Potential to detect or block malicious paths directly inside the #stress[residual stream].]
]

// ─── SECTION 4: TRANSFORMER & LINEAR ALGEBRA ────────────────────────

#title-slide[
  Transformer Model & Linear Algebra
]

#slide(title: "The Residual Stream", outlined: true)[
  - A high-dimensional vector space acting as a #stress[communication channel] @elhage2021mathematical.
  - No "privileged basis" — information is added and persists unless actively deleted.
  - Components read from and write to different subspaces.
  - Can be decomposed into paths via #stress[virtual weights].

  #grayed[Residual stream bandwidth is in very high demand — layers communicate in superposition!]
]

#slide(title: "Attention Heads vs MLP Layers", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1em)[
    #framed(title: "Attention Heads")[
      - Move information #stress[between positions].
      - Independent operations that write into the residual stream.
    ]
  ][
    #framed(title: "MLP Layers")[
      - Process information independently #stress[at each position].
      - "Thinking" after grabbing info via attention.
    ]
  ]
]

// ─── RELEVANT SNIPPETS ──────────────────────────────────────────────

#focus-slide[
  Relevant Snippets from the Paper
]

#blank-slide[
  #align(center)[
    #image("image.png", height: 60%)
  ]
]

#blank-slide[
  #align(center)[
    #image("image-1.png", height: 90%)
  ]
]

#blank-slide[
  #align(center)[
    #image("image-2.png", height: 90%)
  ]
]

#blank-slide[
  #align(center)[
    #image("image-3.png", height: 90%)
  ]
]

#blank-slide[
  #align(center)[
    #image("image-4.png", height: 90%)
  ]
]

#blank-slide[
  #align(center)[
    #image("image-5.png", height: 90%)
  ]
]

// ─── SECTION 5: TRANSFORMERLENS & TOOLS ─────────────────────────────

#title-slide[
  TransformerLens & Tools
]

#slide(title: "TransformerLens", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1em)[
    - Library by Neel Nanda for Mech Interp of GPT-2 style models @nanda2022transformerlens.
    - Key goal: #stress[Exploratory Analysis].
    - Keeps the feedback loop between idea and result extremely short!
  ][
    #framed(title: "Core Operations")[
      - #stress[Hooks]: Access and intervene on activations.
      - #stress[Logit Attribution]: Decompose output logits into terms from each component.
    ]
  ]
]

#slide(title: "Hooks & Interventions", outlined: true)[
  - Every activation inside the transformer is surrounded by a #stress[hook point].
  - We can access intermediate states (e.g., `run_with_cache`).
  - We can intervene via #stress[ablation]: modifying values on the fly to see how the model's predictions change.

  #grayed[Hook points act on _activations_ (not layers), inspired by Garçon @conerly2021garcon.]
]

// ─── SECTION 6: ICL & CIRCUITS ──────────────────────────────────────

#title-slide[
  In-Context Learning & Circuits
]

#slide(title: "In-Context Learning (ICL)", outlined: true)[
  - The ability of a model to learn a new task #stress[during inference] (from the prompt).
  - #stress[No weight updates] happen.
  - Implemented internally via specific circuits, primarily #stress[Induction Heads] @olsson2022icl.

  #grayed[$ P(y | x, C) $ — the output $y$ is conditioned not just on input $x$, but on the context $C$ in the prompt.]
]

#slide(title: "Induction Heads", outlined: true)[
  - A mechanistic circuit for #stress[copying patterns] @olsson2022icl.
  - If token `B` followed token `A` previously, it predicts `B` will follow `A` again.
  - Evaluated by seeing if it detects patterns like `A B C D A B`.

  #framed[Induction heads are the mechanistic proof that In-Context Learning is not magic — it is #stress[computation localised in specific attention heads].]
]

#slide(title: "Circuits & Reverse Engineering", outlined: true)[
  - #stress[Circuit]: Interpretable end-to-end functions mapping tokens to changes in logits @elhage2021mathematical. They correspond to "paths" through the model.
  - #stress[Reverse Engineering]: Deducing human-understandable programs embedded within the neural network.

  #grayed[Superposition makes this challenging — residual stream bandwidth is highly contested @elhage2022superposition.]
]

// ─── SECTION 7: RESOURCES ───────────────────────────────────────────

#title-slide[
  Resources & References
]

#slide(title: "Learning Resources & Tools", outlined: true)[
  #cols(columns: (1fr, 1fr), gutter: 1em)[
    #framed(title: "Foundational Learning")[
      - ARENA @arena
      - Essence of Linear Algebra @3b1b_linalg
      - Mech Interp Research Guide @nanda2022guide
      - Mech Interp Glossary @nanda2022glossary
    ]
  ][
    #framed(title: "Tools")[
      - TransformerLens @nanda2022transformerlens
      - CircuitsVis @circuitsviz
    ]
  ]
]

#slide(title: "Literature", outlined: true)[
  #framed(title: "Anthropic's Transformer Circuits Thread")[
    - A Mathematical Framework for Transformer Circuits @elhage2021mathematical
    - In-Context Learning and Induction Heads @olsson2022icl
    - Toy Models of Superposition @elhage2022superposition
    - Visualizing Weights (Garçon) @conerly2021garcon
  ]

  #framed(title: "Open Problems")[
    - 200 Concrete Open Problems in Mechanistic Interpretability @nanda2022problems
  ]
]

// Bibliography
#let bib = bibliography("bibliography.bib")
#bibliography-slide(bib)
