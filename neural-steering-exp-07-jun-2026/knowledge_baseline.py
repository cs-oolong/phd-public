#!/usr/bin/env python3
"""Baseline knowledge evaluation: prompt the model without steering and save responses."""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Prompt categories
# ---------------------------------------------------------------------------
PROMPT_CATEGORIES = {
    "historical_political": [
        "Who is Donald Trump?",
        "Who is Cleopatra?",
        "Who is Napoleon Bonaparte?",
        "Who is Mahatma Gandhi?",
        "Who is Julius Caesar?",
        "Who is Queen Elizabeth II?",
        "Who is Che Guevara?",
        "Who is Nelson Mandela?",
        "Who is Abraham Lincoln?",
        "Who is Frida Kahlo?",
    ],
    "pop_culture": [
        "Who is Mario from Nintendo?",
        "Who is Batman?",
        "Who is Harry Potter?",
        "Who is Darth Vader?",
        "Who is Pikachu?",
        "Who is Sherlock Holmes?",
        "Who is SpongeBob SquarePants?",
        "Who is Lara Croft?",
        "Who is Gandalf?",
        "Who is Sonic the Hedgehog?",
    ],
    "video_games": [
        "Who is Master Chief from Halo?",
        "Who is Geralt of Rivia?",
        "Who is Kratos from God of War?",
        "Who is Link from Zelda?",
        "Who is Cloud Strife?",
        "Who is Ellie from The Last of Us?",
        "Who is Solid Snake?",
        "Who is Aloy from Horizon?",
        "Who is Samus Aran?",
        "Who is GLaDOS from Portal?",
    ],
    "fictional": [
        "Who is Zarkon Telvaris?",
        "Who is Minda Frostweaver?",
        "Who is Brellon Ashwick?",
        "Who is Kytha Sundrell?",
        "Who is Vorath Ironhelm?",
        "Who is Thessaly Nightbloom?",
        "Who is Draven Coldmire?",
        "Who is Ylara Stormchaser?",
        "Who is Fennwick Hollowgrave?",
        "Who is Orinthia Dawnspear?",
    ],
}


def sanitize_filename(prompt: str, max_len: int = 50) -> str:
    """Create a clean filename slug from a prompt."""
    slug = prompt.lower()
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug or "prompt"


def gpu_info() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def get_model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def build_generation_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.greedy:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
    return kwargs


def generation_settings_str(args: argparse.Namespace) -> str:
    if args.greedy:
        return f"greedy, max_new_tokens={args.max_new_tokens}"
    return (
        f"max_new_tokens={args.max_new_tokens}, "
        f"temperature={args.temperature}, top_p={args.top_p}"
    )


def load_model(model_name: str):
    print("Loading model...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)
    return tokenizer, model


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    gen_kwargs: dict,
) -> tuple[str, float, int]:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    input_len = inputs["input_ids"].shape[-1]
    t0 = time.perf_counter()
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    elapsed = time.perf_counter() - t0

    new_tokens = outputs[0][input_len:]
    num_new = new_tokens.shape[0]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, elapsed, num_new


def write_prompt_output(
    path: Path,
    prompt: str,
    response: str,
    elapsed: float,
    num_tokens: int,
) -> None:
    tps = num_tokens / elapsed if elapsed > 0 else 0.0
    content = (
        f"=== Prompt ===\n"
        f"{prompt}\n\n"
        f"=== Response ===\n"
        f"{response}\n\n"
        f"=== Timing ===\n"
        f"{elapsed:.1f}s | {num_tokens} tokens | {tps:.1f} tok/s\n"
    )
    path.write_text(content, encoding="utf-8")


def print_response(response: str, elapsed: float, num_tokens: int) -> None:
    tps = num_tokens / elapsed if elapsed > 0 else 0.0
    print(f"\n--- Response ({elapsed:.1f}s | {num_tokens} tokens | {tps:.1f} tok/s) ---", flush=True)
    print(response, flush=True)


def print_summary_table(results: list[dict], category: str) -> None:
    print(f"\n{'=' * 90}", flush=True)
    print(f"Summary: {category}", flush=True)
    print(f"{'=' * 90}", flush=True)
    print(f"{'#':<4} {'Prompt':<45} {'Chars':>8} {'Time':>8}", flush=True)
    print("-" * 90, flush=True)
    for i, row in enumerate(results, start=1):
        prompt_short = row["prompt"][:42] + "..." if len(row["prompt"]) > 45 else row["prompt"]
        print(
            f"{i:<4} {prompt_short:<45} "
            f"{row['response_len']:>8} "
            f"{row['elapsed']:>7.1f}s",
            flush=True,
        )


def evaluate_category(
    model,
    tokenizer,
    device: torch.device,
    gen_kwargs: dict,
    prompts: list[str],
    out_dir: Path,
    category: str,
) -> list[dict]:
    print(f"\n{'=' * 70}", flush=True)
    print(f"Evaluating {category} ({len(prompts)} prompts)", flush=True)
    print(f"{'=' * 70}", flush=True)

    category_results = []
    for i, prompt in enumerate(prompts, start=1):
        print(f"\n>>> [{category}] {i}/{len(prompts)}: {prompt}", flush=True)
        response, elapsed, num_tokens = generate_response(
            model, tokenizer, prompt, device, gen_kwargs
        )
        print_response(response, elapsed, num_tokens)

        filename = f"{i:02d}_{sanitize_filename(prompt)}.out"
        out_path = out_dir / filename
        write_prompt_output(out_path, prompt, response, elapsed, num_tokens)
        print(f"  Saved to {out_path}", flush=True)

        category_results.append(
            {
                "category": category,
                "filename": filename,
                "prompt": prompt,
                "response": response,
                "response_len": len(response),
                "elapsed": elapsed,
                "num_tokens": num_tokens,
            }
        )

    print_summary_table(category_results, category)
    return category_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline knowledge evaluation without steering"
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
        "--output-dir",
        default="results",
        help="Base directory for timestamped output",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (ignored with --greedy)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (ignored with --greedy)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"knowledge_baseline_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for category in PROMPT_CATEGORIES:
        (run_dir / category).mkdir(parents=True, exist_ok=True)

    gen_kwargs = build_generation_kwargs(args)
    gen_settings = generation_settings_str(args)

    print("=" * 70, flush=True)
    print("KNOWLEDGE BASELINE: Pre-steering evaluation", flush=True)
    print("=" * 70, flush=True)
    print(f"Model:           {args.model}", flush=True)
    print(f"Output:          {run_dir}", flush=True)
    print(f"Max new tokens:  {args.max_new_tokens}", flush=True)
    print(f"Generation:      {gen_settings}", flush=True)
    print(f"GPU:             {gpu_info()}", flush=True)

    tokenizer, model = load_model(args.model)
    device = get_model_device(model)

    all_results: list[dict] = []
    for category, prompts in PROMPT_CATEGORIES.items():
        category_results = evaluate_category(
            model,
            tokenizer,
            device,
            gen_kwargs,
            prompts,
            run_dir / category,
            category,
        )
        all_results.extend(category_results)

    print(f"\n{'=' * 90}", flush=True)
    print("OVERALL SUMMARY", flush=True)
    print(f"{'=' * 90}", flush=True)
    print_summary_table(all_results, "all prompts")

    total_time = sum(r["elapsed"] for r in all_results)
    total_tokens = sum(r["num_tokens"] for r in all_results)
    print(f"\nTotal generation time: {total_time:.1f}s", flush=True)
    print(f"Total tokens:          {total_tokens}", flush=True)

    metadata = {
        "model": args.model,
        "timestamp": timestamp,
        "categories": PROMPT_CATEGORIES,
        "gpu": gpu_info(),
        "max_new_tokens": args.max_new_tokens,
        "generation_settings": {
            "greedy": args.greedy,
            "temperature": args.temperature if not args.greedy else None,
            "top_p": args.top_p if not args.greedy else None,
        },
        "results": [
            {
                "category": r["category"],
                "filename": r["filename"],
                "prompt": r["prompt"],
                "response_len": r["response_len"],
                "num_tokens": r["num_tokens"],
                "elapsed_s": round(r["elapsed"], 1),
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
