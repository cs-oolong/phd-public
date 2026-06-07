%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\clearpage
\subsection{Injection Attacks in Large Language Models}
\label{sec:serpa}
\textbf{Responsible}: Ana Clara Zoppi Serpa (Ph.D. student)

\subsubsection{Contextualization and details}

Large Language Models (LLMs) have rapidly evolved from text-generation tools into general-purpose reasoning engines, now powering agentic systems, multimodal pipelines, and industrial applications. This widespread deployment has brought to the forefront serious concerns regarding model vulnerabilities—spanning both cybersecurity risks in the context of software engineering workflows~\citep{greshake2023youvesignedforcompromising} as well as broader societal harms, e.g. the development of parasocial relationships with AI systems~\citep{hwang2025aicompanionshipdevelopsevidence}, the generation of non-consensual synthetic media~\citep{brigham2024violationbodyperceptionsaigenerated}, the erosion of human agency in high-stakes decision-making processes~\citep{pataranutaporn2025algorithmicinheritancesurnamebias, TJSP2026_PromptInjection}.

A common thread underlying many of these risks is \textit{prompt injection}~\citep{damacena_2026_prompt_injection}: the ability to manipulate a model's behavior by crafting adversarial inputs that subvert its intended instructions or safety guardrails. The research community has responded with vigorous efforts to categorize, mitigate, and understand jailbreak attacks.

Parallel to this, the field of \textit{mechanistic interpretability (MI)} has emerged as a powerful complementary lens~\citep{rai2025practicalreviewmechanisticinterpretability}. MI seeks to reverse-engineer the internal computations of these large models—identifying which specific components are responsible for each behavior, such as refusal, instruction following, or task deviation. MI efforts are not exclusive to security concerns, but the intersection is particularly promising: while \textit{attack research documents that models can be compromised}, MI can explain \textit{why and how}, opening avenues for more robust and principled defenses.

\subsubsection{Key results}

We conducted a review of the literature that spans approximately 150 works on the topics of prompt injection attacks, jailbreak, mechanistic interpretability~\citep{nanda2024extremely}, and the historical evolution of Transformer architectures~\citep{hannibal0462023awesome}, covering publications from 2017 to 2026.

From this body of work, we highlight the following:

\begin{description}
    \item[\citep{bisconti2026adversarialpoetryuniversalsingleturn}] Demonstrates that adversarial poetry serves as a jailbreak mechanism, exploiting rhythmic and stylistic patterns to bypass safety alignment.
    \item[\citep{elhage2021mathematical}] Establishes the foundational mathematical framework for transformer circuits, formalizing how attention heads compose and interact to process information across layers. It differs from the framework \citep{vaswani2023attentionneed} that is most commonly  used when discussing the Transformer achitecture, establishing definitions (e.g. ``residual stream") that aid in finding interpretable components of the models. 
    \item[\citep{elhage2022toymodelssuperposition}] Investigates superposition in toy neural networks, demonstrating how models encode more features than available dimensions through non-orthogonal, interfered representations.
    \item[\citep{bricken2023towards}] Applies dictionary learning via sparse autoencoders to decompose neural activations into monosemantic features, addressing the polysemanticity problem and advancing the interpretability of internal representations.
    \item[\citep{turner2024steeringlanguagemodelsactivation}] Proposes activation engineering techniques for steering language models, showing how additive interventions on internal representations can reliably modify outputs and behavioral tendencies.
    \item[\citep{wang2022interpretabilitywildcircuitindirect}] Identifies and characterizes the complete circuit for Indirect Object Identification (IOI) in GPT-2 small, demonstrating end-to-end mechanistic interpretability on a real-world linguistic capability.
    \item[\citep{herring2026targetedneuronmodulationcontrastive}] Introduces contrastive pair search as a method for targeted neuron modulation, enabling precise steering of model behavior by identifying and adjusting specific neuronal activations.
    \item[\citep{arditi2024refusallanguagemodelsmediated}] Discovers that refusal behavior in language models is mediated by a single direction in the residual stream, providing a mechanistic understanding of safety-aligned responses and a concrete intervention point.
\end{description}

Next, we conducted initial exploratory experiments to acquire familiarity with prompt injection and MI tools, from which we highlight the following:

We carried out an exploratory experimental campaign across three model families (GPT-2 Small, GPT-J-6B, and instruction-tuned Qwen/Llama) using libraries such as TransformerLens, SAELens, and the NDIF remote execution framework. The experiments covered eight families of MI techniques---logit lens tracing, activation patching, sparse autoencoder feature analysis, residual-stream steering, and contrastive neuron attribution, among others. Our main observations are:

\begin{itemize}
    \item \textbf{Scale-dependent switching.} While GPT-2 exhibits gradual degradation of task adherence under injection, GPT-J-6B shows near-binary behavior: it either remains firmly on-task or collapses immediately, with the decisive shift concentrated in late layers (notably L24).
    \item \textbf{Defensive interventions.} Clamping the residual-stream direction at a single critical layer recovered up to 93\% of on-task behavior in GPT-J-6B, whereas neuron-level and SAE-level ablations proved ineffective, suggesting that task-relevant information is encoded in directions rather than individual units.
    \item \textbf{Creative-framing effects.} On non-instruction-tuned models, direct prose commands caused stronger deviation than poetic or narrative framings. Poetry produced the highest internal uncertainty (entropy) in GPT-J-6B even when output deviation was moderate, indicating a decoupling between representational shift and behavioral change.
    \item \textbf{Emergent robustness.} GPT-J-6B displayed unexpected robustness mechanisms, including ``injection absorption'' (translating the injection text into the target language rather than obeying it) and quotation awareness (recovering task adherence when the injection is enclosed in quotes).
    \item \textbf{Refusal circuits.} Reproducing the Contrastive Neuron Attribution method on Qwen2.5 and Llama-3.1 confirmed that a small subset of MLP neurons (top 0.1\%) mediates refusal behavior, with ablation reducing refusal rates on the JBB-Behaviors dataset.
\end{itemize}

\subsubsection{Next steps}

\begin{enumerate}[label=\textbf{\Roman*.}]
    \item \textbf{Cluster infrastructure.} Deploy the full MI toolchain (TransformerLens, SAELens, nnsight/NDIF, and custom hooking pipelines) on the lab's GPU cluster to enable systematic experiments beyond the current local/remote split.
    \item \textbf{Model access pipeline.} Request API or hosted access to small open-weight models (e.g., Qwen2.5-3B, Llama-3.2-1B) and progressively scale to larger checkpoints (8B--70B parameter range) to study how injection circuits evolve with capacity.
    \item \textbf{Targeted circuit tracing.} Isolate neurons and attention heads responsible for specific injection modalities---adversarial poetry, role-playing narratives, direct request overrides, and multilingual attacks---using contrastive activation patching and automated circuit-discovery protocols.
    \item \textbf{Datasets and evaluation.} Curate standardized evaluation suites that pair benign task prompts with diverse injection types, enabling reproducible benchmarking of both attack success and defensive intervention efficacy.
    \item \textbf{Dissemination.} Package findings into reproducible research artifacts (code, datasets, and model cards) and prepare manuscripts targeting venues at the intersection of machine-learning security and interpretability.
\end{enumerate}

%\begin{enumerate}[label=\textbf{\Roman*.}]
    %\item ARENA (?)
    %\item Select mechanistic interpretability techniques, datasets and models, do experiments and report findings (?)
    %\item Create dataset (?)
    %\item b
%\end{enumerate}
