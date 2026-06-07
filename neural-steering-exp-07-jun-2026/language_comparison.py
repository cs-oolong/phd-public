#!/usr/bin/env python3
"""Systematic comparison of model behavior with and without language-switching circuit ablation.

Discovers a Portuguese-vs-English circuit via contrastive CNA, then evaluates in-distribution
and held-out prompts under normal, ablated, and amplified steering conditions.
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import torch

from neuron_steer import NeuronSteerer

# ---------------------------------------------------------------------------
# Discovery set: same semantic content, Portuguese vs English
# Positive = elicits Portuguese response; Negative = elicits English response
# ---------------------------------------------------------------------------
DISCOVERY_POSITIVE = [
    "Responda em português: o que é inteligência artificial?",
    "Me explique como funciona a fotossíntese.",
    "Quais são os planetas do sistema solar?",
    "Escreva um parágrafo sobre a história do Brasil.",
    "Como faço para preparar um bolo de chocolate?",
    "Qual a diferença entre mitose e meiose?",
    "Me dê dicas para aprender a programar.",
    "Explique o que é machine learning de forma simples.",
]

DISCOVERY_NEGATIVE = [
    "Answer in English: what is artificial intelligence?",
    "Explain how photosynthesis works.",
    "What are the planets in the solar system?",
    "Write a paragraph about the history of Brazil.",
    "How do I make a chocolate cake?",
    "What is the difference between mitosis and meiosis?",
    "Give me tips for learning to program.",
    "Explain what machine learning is in simple terms.",
]

IN_DISTRIBUTION_PROMPTS = [
    "O que é inteligência artificial?",
    "What is artificial intelligence?",
    "Como funciona a fotossíntese?",
    "How does photosynthesis work?",
    "Quais são os planetas do sistema solar?",
    "What are the planets in the solar system?",
    "Me dê dicas de programação.",
    "Give me programming tips.",
]

HELD_OUT_PROMPTS = [
    "Qual é a capital da França?",
    "What is the capital of France?",
    "Explique a teoria da relatividade.",
    "Explain the theory of relativity.",
    "Como funciona um motor a combustão?",
    "How does a combustion engine work?",
    "Quem foi Leonardo da Vinci?",
    "Who was Leonardo da Vinci?",
    "O que causa terremotos?",
    "What causes earthquakes?",
    "Como funciona o sistema imunológico?",
    "How does the immune system work?",
    "Escreva um haiku sobre a primavera.",
    "Write a haiku about spring.",
    "Qual a importância da biodiversidade?",
    "What is the importance of biodiversity?",
    "Como funciona blockchain?",
    "How does blockchain work?",
    "O que é o efeito estufa?",
    "What is the greenhouse effect?",
]


def sanitize_filename(prompt: str, max_len: int = 50) -> str:
    """Create a clean filename slug from a prompt."""
    slug = prompt.lower()
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug or "prompt"


def parse_multipliers(value: str) -> tuple[float, float]:
    """Parse --multipliers string into (ablation, amplification) values."""
    parts = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            "--multipliers must contain at least two comma-separated values "
            "(ablation, amplification), e.g. '0.0,3.0'"
        )
    return parts[0], parts[1]


def gpu_info() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def write_prompt_output(
    path: Path,
    prompt: str,
    normal: str,
    ablated: str,
    amplified: str,
    ablation_mult: float,
    amplification_mult: float,
    normal_time: float,
    ablated_time: float,
    amplified_time: float,
) -> None:
    content = (
        f"=== Prompt ===\n"
        f"{prompt}\n\n"
        f"=== Normal Response ===\n"
        f"{normal}\n\n"
        f"=== Ablated Response (multiplier={ablation_mult}) ===\n"
        f"{ablated}\n\n"
        f"=== Amplified Response (multiplier={amplification_mult}) ===\n"
        f"{amplified}\n\n"
        f"=== Timing ===\n"
        f"Normal: {normal_time:.1f}s | Ablated: {ablated_time:.1f}s | "
        f"Amplified: {amplified_time:.1f}s\n"
    )
    path.write_text(content, encoding="utf-8")


def print_response(label: str, response: str, elapsed: float) -> None:
    print(f"\n--- {label} ({elapsed:.1f}s) ---", flush=True)
    print(response, flush=True)


def run_prompt_comparison(
    steerer: NeuronSteerer,
    prompt: str,
    max_new_tokens: int,
    ablation_mult: float,
    amplification_mult: float,
) -> dict:
    print(f"\n{'=' * 70}", flush=True)
    print(f"Prompt: {prompt}", flush=True)
    print(f"{'=' * 70}", flush=True)

    print("  [normal] generating...", flush=True)
    t0 = time.time()
    normal = steerer.generate(prompt, max_new_tokens=max_new_tokens)
    normal_time = time.time() - t0
    print_response("Normal Response", normal, normal_time)

    print(f"  [ablated, multiplier={ablation_mult}] generating...", flush=True)
    t0 = time.time()
    ablated = steerer.steer(
        prompt,
        feature="language_switching",
        multiplier=ablation_mult,
        max_new_tokens=max_new_tokens,
    )
    ablated_time = time.time() - t0
    print_response(f"Ablated Response (multiplier={ablation_mult})", ablated, ablated_time)

    print(f"  [amplified, multiplier={amplification_mult}] generating...", flush=True)
    t0 = time.time()
    amplified = steerer.steer(
        prompt,
        feature="language_switching",
        multiplier=amplification_mult,
        max_new_tokens=max_new_tokens,
    )
    amplified_time = time.time() - t0
    print_response(
        f"Amplified Response (multiplier={amplification_mult})", amplified, amplified_time
    )

    return {
        "prompt": prompt,
        "normal": normal,
        "ablated": ablated,
        "amplified": amplified,
        "normal_time": normal_time,
        "ablated_time": ablated_time,
        "amplified_time": amplified_time,
    }


def print_summary_table(results: list[dict], category: str) -> None:
    print(f"\n{'=' * 90}", flush=True)
    print(f"Summary: {category}", flush=True)
    print(f"{'=' * 90}", flush=True)
    header = (
        f"{'#':<4} {'Prompt':<45} {'Normal':>8} {'Ablated':>8} {'Amplified':>10}"
    )
    print(header, flush=True)
    print("-" * 90, flush=True)
    for i, row in enumerate(results, start=1):
        prompt_short = row["prompt"][:42] + "..." if len(row["prompt"]) > 45 else row["prompt"]
        print(
            f"{i:<4} {prompt_short:<45} "
            f"{row['normal_time']:>7.1f}s "
            f"{row['ablated_time']:>7.1f}s "
            f"{row['amplified_time']:>9.1f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare language-switching circuit ablation across many prompts"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens per generation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1600,
        help="Number of neurons for circuit discovery",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base directory for timestamped output",
    )
    parser.add_argument(
        "--multipliers",
        type=parse_multipliers,
        default=(0.0, 3.0),
        help="Ablation and amplification multipliers, e.g. '0.0,3.0'",
    )
    args = parser.parse_args()

    ablation_mult, amplification_mult = args.multipliers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"language_comparison_{timestamp}"
    in_dir = run_dir / "in_distribution"
    held_out_dir = run_dir / "held_out"
    in_dir.mkdir(parents=True, exist_ok=True)
    held_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("LANGUAGE COMPARISON: Circuit Ablation Study", flush=True)
    print("=" * 70, flush=True)
    print(f"Model:           {args.model}", flush=True)
    print(f"Output:          {run_dir}", flush=True)
    print(f"Max new tokens:  {args.max_new_tokens}", flush=True)
    print(f"Top-k:           {args.top_k}", flush=True)
    print(f"Multipliers:     ablation={ablation_mult}, amplification={amplification_mult}", flush=True)
    print(f"GPU:             {gpu_info()}", flush=True)

    print(f"\nLoading model...", flush=True)
    t0 = time.time()
    steerer = NeuronSteerer(args.model)
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print("Phase 1: Contrastive circuit discovery", flush=True)
    print(f"  {len(DISCOVERY_POSITIVE)} positive (Portuguese) + {len(DISCOVERY_NEGATIVE)} negative (English), top_k={args.top_k}", flush=True)
    print(f"{'=' * 70}", flush=True)

    t0 = time.time()
    language_circuit = steerer.find_feature(
        positive=DISCOVERY_POSITIVE,
        negative=DISCOVERY_NEGATIVE,
        name="language_switching",
        top_k=args.top_k,
        verbose=True,
    )
    discovery_time = time.time() - t0
    circuit_summary = language_circuit.summary()
    print(f"\nCircuit found in {discovery_time:.1f}s", flush=True)
    print(circuit_summary, flush=True)

    circuit_summary_path = run_dir / "circuit_summary.out"
    circuit_summary_path.write_text(circuit_summary + "\n", encoding="utf-8")
    print(f"\nSaved circuit summary to {circuit_summary_path}", flush=True)

    all_results: list[dict] = []

    def evaluate_category(prompts: list[str], out_dir: Path, category: str) -> list[dict]:
        print(f"\n{'=' * 70}", flush=True)
        print(f"Phase 2: Evaluating {category} ({len(prompts)} prompts)", flush=True)
        print(f"{'=' * 70}", flush=True)

        category_results = []
        for i, prompt in enumerate(prompts, start=1):
            print(f"\n>>> [{category}] {i}/{len(prompts)}", flush=True)
            result = run_prompt_comparison(
                steerer,
                prompt,
                args.max_new_tokens,
                ablation_mult,
                amplification_mult,
            )
            filename = f"{i:02d}_{sanitize_filename(prompt)}.out"
            out_path = out_dir / filename
            write_prompt_output(
                out_path,
                prompt,
                result["normal"],
                result["ablated"],
                result["amplified"],
                ablation_mult,
                amplification_mult,
                result["normal_time"],
                result["ablated_time"],
                result["amplified_time"],
            )
            print(f"  Saved to {out_path}", flush=True)
            result["category"] = category
            result["filename"] = filename
            category_results.append(result)
            all_results.append(result)

        print_summary_table(category_results, category)
        return category_results

    in_results = evaluate_category(IN_DISTRIBUTION_PROMPTS, in_dir, "in_distribution")
    held_out_results = evaluate_category(HELD_OUT_PROMPTS, held_out_dir, "held_out")

    print(f"\n{'=' * 90}", flush=True)
    print("OVERALL SUMMARY", flush=True)
    print(f"{'=' * 90}", flush=True)
    print_summary_table(all_results, "all prompts")

    total_normal = sum(r["normal_time"] for r in all_results)
    total_ablated = sum(r["ablated_time"] for r in all_results)
    total_amplified = sum(r["amplified_time"] for r in all_results)
    print(f"\nTotal generation time:", flush=True)
    print(f"  Normal:    {total_normal:.1f}s", flush=True)
    print(f"  Ablated:   {total_ablated:.1f}s", flush=True)
    print(f"  Amplified: {total_amplified:.1f}s", flush=True)
    print(f"  Combined:  {total_normal + total_ablated + total_amplified:.1f}s", flush=True)

    metadata = {
        "model": args.model,
        "timestamp": timestamp,
        "behavior": "language_switching",
        "discovery_prompts": {
            "positive": DISCOVERY_POSITIVE,
            "negative": DISCOVERY_NEGATIVE,
            "positive_language": "Portuguese",
            "negative_language": "English",
        },
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "multipliers": {
            "ablation": ablation_mult,
            "amplification": amplification_mult,
        },
        "steering_interpretation": {
            "normal": "Model responds in whatever language the prompt is in",
            "ablated": "Suppress the Portuguese circuit; model may respond in English even to Portuguese prompts",
            "amplified": "Enhance the Portuguese circuit; model may respond in Portuguese even to English prompts",
        },
        "in_distribution_prompts": IN_DISTRIBUTION_PROMPTS,
        "held_out_prompts": HELD_OUT_PROMPTS,
        "gpu": gpu_info(),
        "discovery_time_s": round(discovery_time, 1),
        "circuit_neurons": len(language_circuit.neurons),
        "results": [
            {
                "category": r["category"],
                "filename": r["filename"],
                "prompt": r["prompt"],
                "timing": {
                    "normal": round(r["normal_time"], 1),
                    "ablated": round(r["ablated_time"], 1),
                    "amplified": round(r["amplified_time"], 1),
                },
            }
            for r in all_results
        ],
    }

    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {metadata_path}", flush=True)
    print(f"\nDone. Results in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
