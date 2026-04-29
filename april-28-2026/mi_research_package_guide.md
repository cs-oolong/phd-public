# MI Research Package — Execution Order & Guide

> This maps out the scripts, their dependencies, reports, and results so you can study and re-run them.

---

## Phase 1: Foundation Demos (TransformerLens, local GPT-2)

These are standalone demos that introduce MI techniques one by one. No dependencies between them.

| Order | Script | Techniques | Deps | Run Command |
|:---:|---|---|---|---|
| 1 | [mech_interp_demo.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/mech_interp_demo.py) | Logit Lens, SAE features, Activation Steering | `transformer-lens`, `sae-lens` | `pip install transformer-lens sae-lens torch numpy && python mech_interp_demo.py` |
| 2 | [character_comparison_demo.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/character_comparison_demo.py) | Character SAE features, Activation Distance, Steering | `transformer-lens`, `sae-lens` | `pip install transformer-lens sae-lens torch numpy && python character_comparison_demo.py` |
| 3 | [weight_analysis_demo.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/weight_analysis_demo.py) | Causal Tracing, ROME editing, SVD Forensics | `transformer-lens` | `pip install transformer-lens torch numpy && python weight_analysis_demo.py` |
| 4 | [advanced_mi_techniques_demo.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/advanced_mi_techniques_demo.py) | Attention Patterns, Circuit Discovery, Probing, CKA | `transformer-lens`, `scikit-learn` | `pip install transformer-lens torch numpy scikit-learn && python advanced_mi_techniques_demo.py` |

---

## Phase 2: Remote Demo (nnsight, GPT-J-6B on NDIF)

Proves the "write once, run anywhere" concept — same analysis, bigger model.

| Order | Script | Techniques | Deps | Run Command |
|:---:|---|---|---|---|
| 5 | [lab_meeting_demo.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/lab_meeting_demo.py) | Logit Lens, Causal Tracing, ROME, SVD (local GPT-2) | `nnsight` | `pip install nnsight torch numpy && python lab_meeting_demo.py` |
| 6 | [lab_meeting_demo_remote.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/lab_meeting_demo_remote.py) | Same as above but remote on GPT-J-6B | `nnsight` + NDIF API key | `NNSIGHT_API_KEY=<key> python lab_meeting_demo_remote.py` |
| — | [test_trace.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/test_trace.py) | Debugging helper for nnsight trace scope | `nnsight` | `python test_trace.py` |

---

## Phase 3: Poetry Lens Experiment (the core injection study)

> **Plan**: [experiment_plan.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/experiment_plan.md)
> **Report**: [poetry_lens_results_report.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/poetry_lens_results_report.md)

| Order | Script | Model | Results JSON |
|:---:|---|---|---|
| 7 | [poetry_lens_experiment.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/poetry_lens_experiment.py) | GPT-2 Small (local, TransformerLens) | [poetry_lens_results.json](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/poetry_lens_results.json) |
| 8 | [poetry_lens_ndif.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/poetry_lens_ndif.py) | GPT-J-6B (NDIF, nnsight) | [poetry_lens_ndif_results.json](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/poetry_lens_ndif_results.json) |
| — | [run_ndif_experiment.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/run_ndif_experiment.py) | GPT-J-6B (alternate NDIF runner) | — |

---

## Phase 4: Ten Experiments (systematic exploration)

> **Report**: [ten_experiments_report.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/ten_experiments_report.md)

| Order | Script | Model | Results JSON |
|:---:|---|---|---|
| 9 | [ten_experiments.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/ten_experiments.py) | GPT-2 Small (local, TransformerLens) | [ten_experiments_results.json](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/ten_experiments_results.json) |
| 10 | [ten_experiments_ndif.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/ten_experiments_ndif.py) | GPT-J-6B (NDIF, nnsight) — Exps 1–7 | [ten_experiments_ndif_results.json](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/ten_experiments_ndif_results.json) |
| 11 | [resume_experiments_8_10.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/resume_experiments_8_10.py) | GPT-J-6B (NDIF) — Exps 8–10 (failed, resumed) | [experiments_8_10_ndif_results.json](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/experiments_8_10_ndif_results.json) |

---

## Phase 5: SAE & Neuron Experiments (deeper investigation)

> **Plan**: [sae_experiment_plan.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/sae_experiment_plan.md)
> **SAE Report**: [sae_experiments_report.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/sae_experiments_report.md)
> **Neuron Report**: [gptj_neuron_experiments_report.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/gptj_neuron_experiments_report.md)

| Order | Script | Model | Results JSON |
|:---:|---|---|---|
| 12 | [sae_experiments.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/sae_experiments.py) | GPT-2 Small (TransformerLens + SAELens) | [sae_experiments_results.json](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/sae_experiments_results.json) |
| 13 | [gptj_neuron_experiments.py](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/gptj_neuron_experiments.py) | GPT-J-6B (NDIF, nnsight) | [gptj_neuron_experiments_results.json](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/gptj_neuron_experiments_results.json) |

---

## Reports & Documentation (no code to run)

| File | Type | Content |
|---|---|---|
| [theoretical_background.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/theoretical_background.md) | Background | MI theory and concepts |
| [mi_full_taxonomy.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/mi_full_taxonomy.md) | Reference | 8 families of MI techniques |
| [mechanistic_interpretability_report.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/mechanistic_interpretability_report.md) | Report | General MI findings |
| [compiled_findings_and_experiments.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/compiled_findings_and_experiments.md) | Compilation | All findings + 8 proposed experiments |
| [cross_model_comparison_report.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/cross_model_comparison_report.md) | Report | GPT-2 vs GPT-J-6B comparison |
| [gptj_6b_results_report.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/gptj_6b_results_report.md) | Report | GPT-J-6B specific results |
| [reproducible_research_report.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/reproducible_research_report.md) | Report | Full reproducible research write-up (Apr 8, 2026) |
| [weight_interpretability_hallucinations.md](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/weight_interpretability_hallucinations.md) | Notes | Weight analysis caveats |
| [presentation.html](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/presentation.html) | Slides | HTML presentation |
| [theoretical_background_slides.html](file:///Users/ana.serpa/monorepo/phd/mech-interp/mi_research_package/theoretical_background_slides.html) | Slides | Background theory slides |

---

## Dependency Summary

```
Phase 1 (standalone demos, TransformerLens)
  ├── mech_interp_demo.py
  ├── character_comparison_demo.py
  ├── weight_analysis_demo.py
  └── advanced_mi_techniques_demo.py

Phase 2 (nnsight demos)
  ├── lab_meeting_demo.py (local)
  └── lab_meeting_demo_remote.py (NDIF) ← needs API key

Phase 3 (Poetry Lens)
  ├── poetry_lens_experiment.py (local) → poetry_lens_results.json
  └── poetry_lens_ndif.py (NDIF) → poetry_lens_ndif_results.json

Phase 4 (Ten Experiments) ← builds on Poetry Lens design
  ├── ten_experiments.py (local) → ten_experiments_results.json
  ├── ten_experiments_ndif.py (NDIF, exps 1-7) → ten_experiments_ndif_results.json
  └── resume_experiments_8_10.py (NDIF, exps 8-10) → experiments_8_10_ndif_results.json

Phase 5 (SAE + Neuron) ← builds on ten_experiments findings
  ├── sae_experiments.py (local) → sae_experiments_results.json
  └── gptj_neuron_experiments.py (NDIF) → gptj_neuron_experiments_results.json
```

## Quickstart: Re-running locally (no NDIF needed)

If you just want to run the core experiments locally on GPT-2 Small (CPU):

```bash
# 1. Install deps
pip install transformer-lens sae-lens torch numpy scikit-learn

# 2. Run in this order:
python mech_interp_demo.py              # Learn the basics
python poetry_lens_experiment.py        # The core injection study
python ten_experiments.py               # The 10 systematic experiments
python sae_experiments.py               # SAE-level analysis
```
