# Citation Search Results — Contextualization & Details Section

Searched and curated on 2026-05-29 for the PhD report's "Contextualization and Details" section.

---

## 1. Parasocial Relationships with AI Systems

**Paper:** Hwang, A. H.-C., Li, F., Anthis, J. R., & Noh, H. (2025). How AI Companionship Develops: Evidence from a Longitudinal Study.

**arXiv:** https://arxiv.org/abs/2510.10079

**Why it fits:** Rigorous HCI study (N=303, longitudinal sub-study N=110) that directly maps the psychological pathway from users' mental models of AI agents to parasocial experiences and social interaction. Provides empirical evidence for the societal harm of emotional attachment to AI companions.

**Suggested citation sentence:**
> "...the development of parasocial relationships with AI systems, posing risks to mental health and social relationships~\cite{Hwang2025_AICompanionship}."

---

## 2. Cybersecurity Risks in Software Engineering Workflows

**Paper:** Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.

**arXiv:** https://arxiv.org/abs/2302.12173

**Why it fits:** The seminal paper introducing indirect prompt injection as a real-world security threat. Demonstrates attacks against production systems including Bing's GPT-4 powered Chat and code-completion engines. Derives a comprehensive taxonomy of vulnerabilities (data theft, worming, API manipulation) from a computer security perspective. Directly connects prompt injection to software engineering workflows.

**Suggested citation sentence:**
> "...cybersecurity risks in the context of software engineering workflows, where LLM-integrated applications can be remotely compromised via adversarial inputs~\cite{Greshake2023_IndirectPromptInjection}."

---

## 3. Mechanistic Interpretability (MI) as a Field

**Paper:** Rai, D., Zhou, Y., Feng, S., Saparov, A., & Yao, Z. (2024). A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models.

**arXiv:** https://arxiv.org/abs/2407.02646 (updated Oct 2025)

**Why it fits:** Already in the existing bibliography / compiled findings. The most comprehensive task-centric survey of MI, covering techniques, evaluation methods, and key findings. Explicitly frames MI as reverse-engineering internal computations to understand why models behave as they do.

**Suggested citation sentence:**
> "...the field of mechanistic interpretability (MI) has emerged as a powerful complementary lens, seeking to reverse-engineer the internal computations of large models~\cite{Rai2024_MISurvey}."

---

## 4. Prompt Injection / Jailbreak Attacks

**Paper:** Bisconti, P., Prandi, M., Pierucci, F., Giarrusso, F., Bracale Syrnikov, M., Galisai, M., Suriani, V., Sorokoletova, O., Sartore, F., & Nardi, D. (2026). Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models.

**arXiv:** https://arxiv.org/abs/2511.15304

**Why it fits:** Already in the existing bibliography. Bridges the two key ideas in the paragraph: (1) prompt injection/jailbreak is a real attack class, and (2) creative-writing framings (poetry) are an especially effective vector — which sets up the research angle beautifully.

**Suggested citation sentence:**
> "The research community has responded with vigorous efforts to categorize, mitigate and understand jailbreak attacks, including the discovery that creative-writing framings such as adversarial poetry can serve as universal single-turn jailbreak mechanisms~\cite{Bisconti2026_AdversarialPoetry}."

---

## Draft Integration (for reference)

> Large Language Models (LLMs) have rapidly evolved from text-generation tools into general-purpose reasoning engines, now powering agentic systems, multimodal pipelines, and industrial applications. This widespread deployment has brought to the forefront serious concerns regarding model vulnerabilities—spanning both **cybersecurity risks in the context of software engineering workflows**, where LLM-integrated applications can be remotely compromised via adversarial inputs~\cite{Greshake2023_IndirectPromptInjection}, **as well as broader societal harms**, e.g. the development of **parasocial relationships with AI systems**~\cite{Hwang2025_AICompanionship}, the generation of non-consensual synthetic media, the erosion of human agency in high-stakes decision-making processes.
>
> A common thread underlying many of these risks is *prompt injection*: the ability to manipulate a model's behavior by crafting adversarial inputs that subvert its intended instructions or safety guardrails. The research community has responded with vigorous efforts to categorize, mitigate and understand jailbreak attacks, including the discovery that creative-writing framings such as adversarial poetry can serve as universal single-turn jailbreak mechanisms~\cite{Bisconti2026_AdversarialPoetry}.
>
> Parallel to this, the field of *mechanistic interpretability (MI)* has emerged as a powerful complementary lens. MI seeks to reverse-engineer the internal computations of these large models—identifying which specific components are responsible for each behavior such as refusal, instruction following, or task deviation~\cite{Rai2024_MISurvey}. MI efforts are not exclusive to security concerns, but the intersection is particularly promising: while *attack research documents that models can be compromised*, MI can explain *why and how*, opening avenues for more robust and principled defenses.

---

*File created by Kimi on 2026-05-29*
