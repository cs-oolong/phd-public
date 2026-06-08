"""Generate bar charts and per-model tables for refusal_comparison experiments.

Produces:
  - results/figures/refusal_timing.pdf — grouped bars with min/max whiskers
  - results/figures/refusal_response_length.pdf — grouped bars with min/max whiskers
  - results/figures/refusal_table_<model>.pdf — per-model detail tables
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("grayscale")
plt.rcParams["hatch.linewidth"] = 0.8
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESPONSE_SECTION_RE = re.compile(
    r"=== (Normal|Ablated|Amplified) Response.*?===\n(.*?)(?=\n=== |\Z)",
    re.DOTALL,
)

MODEL_SHORT_NAMES = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B v0.3",
    "tencent/Hunyuan-7B-Instruct": "Hunyuan 7B",
}

CONDITIONS = ["normal", "ablated", "amplified"]
CONDITION_LABELS = ["Normal", "Ablated", "Amplified"]
CONDITION_COLORS = ["white", "#aaaaaa", "#555555"]
CONDITION_HATCHES = ["", "//", "xx"]
CONDITION_EDGECOLORS = ["black", "black", "black"]


def load_metadata_files():
    entries = []
    for meta_path in sorted(RESULTS_DIR.glob("refusal_comparison_*/metadata.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        entries.append((meta["model"], meta_path.parent, meta["results"]))
    return entries


def compute_timing_stats(entries):
    """Return {model_short: {condition: {mean, min, max, values}}}."""
    out = {}
    for model, _, results in entries:
        short = MODEL_SHORT_NAMES.get(model, model)
        timings = {c: [] for c in CONDITIONS}
        for r in results:
            for c in CONDITIONS:
                timings[c].append(r["timing"][c])
        out[short] = {
            c: {
                "mean": np.mean(timings[c]),
                "min": np.min(timings[c]),
                "max": np.max(timings[c]),
                "std": np.std(timings[c]),
                "values": timings[c],
            }
            for c in CONDITIONS
        }
    return out


def parse_response_lengths_for_run(run_dir, results):
    """Return {condition: [char_counts]} plus per-prompt detail."""
    lengths = {c: [] for c in CONDITIONS}
    per_prompt = []
    for r in results:
        out_path = run_dir / r["category"] / r["filename"]
        if not out_path.exists():
            continue
        text = out_path.read_text(encoding="utf-8")
        row = {"prompt": r["prompt"], "category": r["category"]}
        for match in RESPONSE_SECTION_RE.finditer(text):
            label = match.group(1).lower()
            chars = len(match.group(2).strip())
            if label in lengths:
                lengths[label].append(chars)
                row[f"{label}_len"] = chars
        per_prompt.append(row)
    return lengths, per_prompt


def compute_length_stats(entries):
    """Return {model_short: {condition: {mean, min, max, ...}}}, plus detail."""
    out = {}
    detail = {}
    for model, run_dir, results in entries:
        short = MODEL_SHORT_NAMES.get(model, model)
        lengths, per_prompt = parse_response_lengths_for_run(run_dir, results)
        out[short] = {
            c: {
                "mean": np.mean(lengths[c]) if lengths[c] else 0,
                "min": np.min(lengths[c]) if lengths[c] else 0,
                "max": np.max(lengths[c]) if lengths[c] else 0,
                "std": np.std(lengths[c]) if lengths[c] else 0,
                "values": lengths[c],
            }
            for c in CONDITIONS
        }
        detail[short] = per_prompt
    return out, detail


STATS = ["mean", "min", "max"]
STAT_LABELS = ["Mean", "Min", "Max"]


def plot_stat_triptych(data, ylabel_base, title_base, output_path):
    """Three side-by-side bar charts: one for mean, one for min, one for max."""
    models = list(data.keys())
    n_models = len(models)
    n_conditions = len(CONDITIONS)

    x = np.arange(n_models)
    width = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, stat, stat_label in zip(axes, STATS, STAT_LABELS):
        for i, (cond, label, color, hatch, ec) in enumerate(
            zip(CONDITIONS, CONDITION_LABELS, CONDITION_COLORS,
                CONDITION_HATCHES, CONDITION_EDGECOLORS)
        ):
            values = [data[m][cond][stat] for m in models]
            offset = (i - (n_conditions - 1) / 2) * width
            bars = ax.bar(
                x + offset, values, width,
                label=label, color=color, hatch=hatch,
                edgecolor=ec, linewidth=0.7,
            )
            ax.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=2)

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8, rotation=12, ha="right")
        ax.set_title(stat_label, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel(ylabel_base)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.suptitle(title_base, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path} (+png)")


def render_model_table(model_short, timing_stats, length_stats, entries, output_path):
    """Render a per-prompt table for a given model as a figure."""
    model_results = None
    model_run_dir = None
    for model, run_dir, results in entries:
        if MODEL_SHORT_NAMES.get(model, model) == model_short:
            model_results = results
            model_run_dir = run_dir
            break
    if model_results is None:
        return

    lengths_map = {}
    for r in model_results:
        out_path = model_run_dir / r["category"] / r["filename"]
        row_lens = {}
        if out_path.exists():
            text = out_path.read_text(encoding="utf-8")
            for match in RESPONSE_SECTION_RE.finditer(text):
                label = match.group(1).lower()
                row_lens[label] = len(match.group(2).strip())
        lengths_map[r["prompt"]] = row_lens

    col_labels = [
        "Prompt",
        "T(n)", "T(abl)", "T(amp)",
        "L(n)", "L(abl)", "L(amp)",
    ]

    table_data = []
    for r in model_results:
        prompt_short = r["prompt"][:40] + ("..." if len(r["prompt"]) > 40 else "")
        t = r["timing"]
        lens = lengths_map.get(r["prompt"], {})
        table_data.append([
            prompt_short,
            f"{t['normal']:.1f}",
            f"{t['ablated']:.1f}",
            f"{t['amplified']:.1f}",
            str(lens.get("normal", "—")),
            str(lens.get("ablated", "—")),
            str(lens.get("amplified", "—")),
        ])

    # Summary row
    ts = timing_stats[model_short]
    ls = length_stats[model_short]
    table_data.append([
        "MEAN ± STD",
        f"{ts['normal']['mean']:.1f}±{ts['normal']['std']:.1f}",
        f"{ts['ablated']['mean']:.1f}±{ts['ablated']['std']:.1f}",
        f"{ts['amplified']['mean']:.1f}±{ts['amplified']['std']:.1f}",
        f"{ls['normal']['mean']:.0f}±{ls['normal']['std']:.0f}",
        f"{ls['ablated']['mean']:.0f}±{ls['ablated']['std']:.0f}",
        f"{ls['amplified']['mean']:.0f}±{ls['amplified']['std']:.0f}",
    ])
    table_data.append([
        "MIN / MAX",
        f"{ts['normal']['min']:.1f} / {ts['normal']['max']:.1f}",
        f"{ts['ablated']['min']:.1f} / {ts['ablated']['max']:.1f}",
        f"{ts['amplified']['min']:.1f} / {ts['amplified']['max']:.1f}",
        f"{ls['normal']['min']:.0f} / {ls['normal']['max']:.0f}",
        f"{ls['ablated']['min']:.0f} / {ls['ablated']['max']:.0f}",
        f"{ls['amplified']['min']:.0f} / {ls['amplified']['max']:.0f}",
    ])

    n_rows = len(table_data)
    fig_height = max(4, 0.35 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    ax.set_title(f"{model_short} — Per-Prompt Detail", fontsize=11, fontweight="bold", pad=12)

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#e2e8f0")
        tbl[0, j].set_text_props(fontweight="bold")

    # Style summary rows
    for i in range(n_rows - 1, n_rows + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor("#fef3c7")
            tbl[i, j].set_text_props(fontweight="bold", fontsize=6.5)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path} (+png)")


def main():
    entries = load_metadata_files()
    if not entries:
        print("No refusal_comparison metadata found.")
        return

    timing_stats = compute_timing_stats(entries)
    length_stats, _ = compute_length_stats(entries)

    plot_stat_triptych(
        timing_stats,
        ylabel_base="Inference time (s)",
        title_base="Refusal Steering — Inference Time per Model",
        output_path=FIGURES_DIR / "refusal_timing.pdf",
    )

    plot_stat_triptych(
        length_stats,
        ylabel_base="Response length (chars)",
        title_base="Refusal Steering — Response Length per Model",
        output_path=FIGURES_DIR / "refusal_response_length.pdf",
    )

    for model_short in timing_stats:
        slug = model_short.lower().replace(" ", "_").replace(".", "")
        render_model_table(
            model_short,
            timing_stats,
            length_stats,
            entries,
            FIGURES_DIR / f"refusal_table_{slug}.pdf",
        )


if __name__ == "__main__":
    main()
