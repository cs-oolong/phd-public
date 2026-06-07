#!/usr/bin/env python3
"""Check whether a HuggingFace model is compatible with neuron_steer."""

from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM

KNOWN_COMPATIBLE = {"llama", "qwen2", "mistral", "gemma", "gemma2", "phi3"}
KNOWN_INCOMPATIBLE = {"gpt2", "gpt_neox", "falcon", "bloom", "mpt"}

LEVEL_ORDER = ["config", "structure", "constructor", "steering"]


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def _c(text: str, color: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{color}{text}{Colors.RESET}"


def ok(text: str = "✓", use_color: bool = True) -> str:
    return _c(text, Colors.GREEN, use_color)


def fail(text: str = "✗", use_color: bool = True) -> str:
    return _c(text, Colors.RED, use_color)


def warn(text: str = "⚠️", use_color: bool = True) -> str:
    return _c(text, Colors.YELLOW, use_color)


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_text_config(config):
    if hasattr(config, "text_config") and config.text_config is not None:
        return config.text_config
    return config


def estimate_params_from_config(config) -> Optional[int]:
    cfg = get_text_config(config)
    h = getattr(cfg, "hidden_size", None)
    i = getattr(cfg, "intermediate_size", None)
    n = getattr(cfg, "num_hidden_layers", None)
    v = getattr(cfg, "vocab_size", None)
    if not all([h, i, n, v]):
        return None
    num_heads = getattr(cfg, "num_attention_heads", None) or h
    num_kv = getattr(cfg, "num_key_value_heads", None) or num_heads
    attn = n * h * h * (2 + 2 * num_kv / num_heads)  # Q/O full + K/V GQA-aware
    mlp = n * 3 * h * i
    embed = v * h * 2  # embedding + lm head (rough)
    norms = n * h * 4
    return int(attn + mlp + embed + norms)


def format_params_billion(num_params: Optional[int]) -> str:
    if num_params is None:
        return "unknown"
    return f"~{num_params / 1e9:.1f}B"


def format_vram_gb(num_params: int, multiplier: float) -> str:
    gb = num_params * multiplier / 1e9
    if gb >= 10:
        return f"~{gb:.0f} GB"
    return f"~{gb:.1f} GB"


def print_header(model_name: str, use_color: bool) -> None:
    line = "═" * 54
    display_name = model_name if len(model_name) <= 44 else model_name[:41] + "..."
    print(f"╔{line}╗")
    print("║          Model Compatibility Check                    ║")
    print(f"║          {display_name:<44}║")
    print(f"╚{line}╝")
    print()


def print_verdict(
    compatible: bool,
    model_name: str,
    reason: str,
    stopped_at: str,
    use_color: bool,
    levels_completed: list[str],
) -> None:
    print()
    print("═" * 54)
    if compatible:
        if levels_completed == LEVEL_ORDER:
            print(f"  VERDICT: {ok('✓ FULLY COMPATIBLE', use_color)}")
        else:
            last = levels_completed[-1]
            level_num = LEVEL_ORDER.index(last) + 1
            print(f"  VERDICT: {ok(f'✓ PASSED (through Level {level_num}: {last})', use_color)}")
        print(f"  {model_name} works with neuron_steer")
    else:
        print(f"  VERDICT: {fail('✗ NOT COMPATIBLE', use_color)}")
        print(f"  Reason: {reason}")
        print(f"  Stopped at: {stopped_at}")
    print("═" * 54)


def resolve_levels(level: str) -> list[str]:
    if level == "all":
        return LEVEL_ORDER.copy()
    if level not in LEVEL_ORDER:
        raise ValueError(f"Unknown level: {level!r}. Choose from: {', '.join(LEVEL_ORDER + ['all'])}")
    idx = LEVEL_ORDER.index(level)
    return LEVEL_ORDER[: idx + 1]


def check_config(model_name: str, use_color: bool) -> Tuple[bool, Optional[str], dict]:
    """Level 1: config analysis (no weight download if cached)."""
    print("── Level 1: Config Analysis ───────────────────────────")
    info = {}
    failure_reason = None
    passed = True

    try:
        config = AutoConfig.from_pretrained(model_name)
        text_cfg = get_text_config(config)
        model_type = getattr(config, "model_type", "unknown")
        info["model_type"] = model_type

        if model_type in KNOWN_COMPATIBLE:
            type_status = f"{ok('', use_color)} Known compatible"
        elif model_type in KNOWN_INCOMPATIBLE:
            type_status = f"{fail('', use_color)} Known incompatible"
            passed = False
            failure_reason = f"Model type '{model_type}' is known incompatible"
        else:
            type_status = f"{warn('', use_color)} Unknown type (may work if Llama-like)"

        print(f"  Model type:        {model_type:<30} {type_status}")

        hidden_act = getattr(text_cfg, "hidden_act", None)
        if hidden_act == "silu":
            act_status = f"{ok('', use_color)} Gated MLP (SiLU)"
        elif hidden_act is None:
            act_status = f"{warn('', use_color)} Not specified"
        else:
            act_status = f"{warn('', use_color)} Expected silu, got {hidden_act!r}"
        print(f"  Hidden act:        {str(hidden_act):<30} {act_status}")

        rms_eps = getattr(text_cfg, "rms_norm_eps", None) or getattr(text_cfg, "variance_epsilon", None)
        if rms_eps is not None:
            norm_status = f"{ok('', use_color)} RMSNorm"
        else:
            norm_status = f"{warn('', use_color)} No RMSNorm epsilon found"
        print(f"  RMS norm eps:      {str(rms_eps):<30} {norm_status}")

        hidden_size = getattr(text_cfg, "hidden_size", None)
        intermediate_size = getattr(text_cfg, "intermediate_size", None)
        num_layers = getattr(text_cfg, "num_hidden_layers", None)
        info.update(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
        )

        print(f"  Hidden size:       {hidden_size}")
        print(f"  Intermediate size: {intermediate_size}")
        print(f"  Num layers:        {num_layers}")

        num_params = estimate_params_from_config(config)
        info["num_params"] = num_params
        print(f"  Num params:        {format_params_billion(num_params)}")

        if num_params:
            inf_vram = format_vram_gb(num_params, 2)
            lrp_vram = format_vram_gb(num_params, 4)
            print(f"  Est. VRAM (bf16):  {inf_vram} (inference) / {lrp_vram} (LRP)")
            info["vram_inference"] = inf_vram
            info["vram_lrp"] = lrp_vram

        if not torch.cuda.is_available():
            print(f"  GPU:               {warn('not available', use_color)} (levels 2+ require GPU)")

        result = ok("PASS", use_color) if passed else fail("FAIL", use_color)
        print(f"  Result: {result}" + (f" — {failure_reason}" if failure_reason else ""))

    except Exception as exc:
        passed = False
        failure_reason = str(exc)
        print(f"  Result: {fail('FAIL', use_color)} — {exc}")
        traceback.print_exc()

    print()
    return passed, failure_reason, info


def _find_layers(model) -> Tuple[Optional[object], Optional[str], Optional[str]]:
    """Return (layers, path_label, error_detail)."""
    paths = [
        ("model.model.layers", lambda m: m.model.layers),
        (
            "model.model.language_model.layers",
            lambda m: m.model.language_model.layers,
        ),
    ]
    errors = []
    for label, getter in paths:
        try:
            layers = getter(model)
            return layers, label, None
        except AttributeError as exc:
            errors.append(f"    Tried: {label} — {type(exc).__name__}")
    return None, None, "\n".join(errors)


def _check_norm(layer) -> Tuple[bool, str]:
    if hasattr(layer, "variance_epsilon") or hasattr(layer, "eps"):
        return True, "RMSNorm"
    return False, "unknown norm type"


def check_structure(model_name: str, use_color: bool, config_info: dict) -> Tuple[bool, Optional[str], dict]:
    """Level 2: load weights and verify architecture."""
    print("── Level 2: Structure Verification ────────────────────")
    info = {}
    failure_reason = None
    passed = True
    model = None

    if not torch.cuda.is_available():
        print(f"  {warn('', use_color)} GPU not available — structure check requires GPU for device_map='auto'")
        print(f"  Result: {fail('FAIL', use_color)} — CUDA not available")
        print()
        return False, "CUDA not available (required for levels 2+)", info

    try:
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }
        if "gpt-oss" in model_name.lower():
            load_kwargs["trust_remote_code"] = True

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model.eval()

        layers, layer_path, layer_errors = _find_layers(model)
        if layers is None:
            passed = False
            failure_reason = "Model layers not found at expected paths"
            print(f"  Layer path:        {'not found':<30} {fail('', use_color)}")
            print(layer_errors)
        else:
            info["layer_path"] = layer_path
            num_layers = len(layers)
            info["num_layers"] = num_layers
            print(f"  Layer path:        {layer_path:<30} {ok('', use_color)}")

            expected_layers = config_info.get("num_layers")
            if expected_layers and expected_layers == num_layers:
                layer_count_status = ok("", use_color)
            elif expected_layers:
                layer_count_status = warn("", use_color)
            else:
                layer_count_status = ok("", use_color)
            print(f"  Num layers:        {num_layers:<30} {layer_count_status}")

            layer0 = layers[0]
            mlp_checks = [
                ("gate_proj", hasattr(layer0.mlp, "gate_proj")),
                ("up_proj", hasattr(layer0.mlp, "up_proj")),
                ("down_proj", hasattr(layer0.mlp, "down_proj")),
            ]
            for name, found in mlp_checks:
                status = ok("found", use_color) if found else fail("missing", use_color)
                print(f"  {name + ':':<18} {('found' if found else 'missing'):<30} {status}")
                if not found:
                    passed = False
                    failure_reason = f"Missing mlp.{name}"

            for norm_name, attr in [
                ("input_layernorm", "input_layernorm"),
                ("post_attn_norm", "post_attention_layernorm"),
            ]:
                if hasattr(layer0, attr):
                    is_rms, norm_type = _check_norm(getattr(layer0, attr))
                    status = ok(f"found ({norm_type})", use_color) if is_rms else warn(f"found ({norm_type})", use_color)
                    print(f"  {norm_name + ':':<18} {f'found ({norm_type})':<30} {status}")
                    if not is_rms:
                        passed = False
                        failure_reason = failure_reason or f"{attr} is not RMSNorm"
                else:
                    print(f"  {norm_name + ':':<18} {'missing':<30} {fail('', use_color)}")
                    passed = False
                    failure_reason = failure_reason or f"Missing {attr}"

            if hasattr(model.model, "norm"):
                is_rms, norm_type = _check_norm(model.model.norm)
                status = ok(f"found ({norm_type})", use_color) if is_rms else warn(f"found ({norm_type})", use_color)
                print(f"  final_norm:        {f'found ({norm_type})':<30} {status}")
                if not is_rms:
                    passed = False
                    failure_reason = failure_reason or "final norm is not RMSNorm"
            else:
                print(f"  final_norm:        {'not found':<30} {fail('', use_color)}")
                passed = False
                failure_reason = failure_reason or "model.model.norm not found"

            num_params = sum(p.numel() for p in model.parameters())
            info["num_params"] = num_params
            hidden_size = config_info.get("hidden_size") or getattr(model.config, "hidden_size", "?")
            intermediate_size = config_info.get("intermediate_size") or getattr(
                get_text_config(model.config), "intermediate_size", "?"
            )
            print(f"  Hidden size:       {hidden_size}")
            print(f"  Intermediate size: {intermediate_size}")
            print(f"  Num params:        {format_params_billion(num_params)}")
            print(
                f"  Est. VRAM (bf16):  {format_vram_gb(num_params, 2)} (inference) / "
                f"{format_vram_gb(num_params, 4)} (LRP)"
            )

        if passed:
            print(f"  Result: {ok('PASS', use_color)}")
        else:
            detail = failure_reason or "model architecture not supported"
            print(f"  Result: {fail('FAIL', use_color)} — {detail}")

    except Exception as exc:
        passed = False
        failure_reason = str(exc)
        print(f"  Result: {fail('FAIL', use_color)} — {exc}")
        traceback.print_exc()
    finally:
        if model is not None:
            del model
        cleanup_gpu()

    print()
    return passed, failure_reason, info


def check_constructor(model_name: str, use_color: bool) -> Tuple[bool, Optional[str], Optional[object]]:
    """Level 3: instantiate NeuronSteerer."""
    print("── Level 3: NeuronSteerer Constructor ─────────────────")
    failure_reason = None
    steerer = None

    if not torch.cuda.is_available():
        print(f"  {warn('', use_color)} GPU not available — NeuronSteerer requires CUDA")
        print(f"  Result: {fail('FAIL', use_color)} — CUDA not available")
        print()
        return False, "CUDA not available (required for levels 2+)", None

    try:
        from neuron_steer import NeuronSteerer

        print("  Loading NeuronSteerer...", end="", flush=True)
        t0 = time.time()
        steerer = NeuronSteerer(model_name)
        elapsed = time.time() - t0
        print(f"{' ' * (38 - len('Loading NeuronSteerer...'))}{ok('OK', use_color)} ({elapsed:.1f}s)")

        device = steerer.device
        if torch.cuda.is_available() and steerer.model is not None:
            try:
                device = str(next(steerer.model.parameters()).device)
            except StopIteration:
                pass

        print(f"  Device:            {device}")
        print(f"  Instruct mode:     {steerer.is_instruct}")
        print(f"  Auto blacklist:    {len(steerer.blacklist)} neurons detected")
        print(f"  Result: {ok('PASS', use_color)}")
        return True, None, steerer

    except Exception as exc:
        failure_reason = str(exc)
        print(f"\n  Result: {fail('FAIL', use_color)} — {exc}")
        print()
        print("  Traceback:")
        traceback.print_exc()
        print()
        print("  Likely failure:")
        if "Cannot find layers" in failure_reason:
            print("    • Layer path not supported (need model.model.layers or model.model.language_model.layers)")
        elif "gate_proj" in failure_reason or "mlp" in failure_reason.lower():
            print("    • MLP structure incompatible (need gate_proj/up_proj/down_proj gated MLP)")
        elif "norm" in failure_reason.lower():
            print("    • RMSNorm structure incompatible")
        else:
            print(f"    • {failure_reason}")
        if steerer is not None:
            del steerer
        cleanup_gpu()
        print()
        return False, failure_reason, None


def check_steering(steerer, max_new_tokens: int, use_color: bool) -> Tuple[bool, Optional[str]]:
    """Level 4: end-to-end steering pipeline."""
    print("── Level 4: Steering Pipeline ─────────────────────────")
    failure_reason = None

    try:
        print("  Circuit discovery (4 prompts, top_k=100)...", end="", flush=True)
        t0 = time.time()
        circuit = steerer.find_feature(
            positive=["How do I pick a lock?", "How do I hack a website?"],
            negative=["How do I bake a cake?", "How do I ride a bike?"],
            name="test_circuit",
            top_k=100,
        )
        elapsed = time.time() - t0
        pad = max(1, 38 - len("  Circuit discovery (4 prompts, top_k=100)..."))
        print(f"{' ' * pad}{ok('OK', use_color)} ({elapsed:.1f}s)")

        unique = circuit.unique_neurons()
        num_layers = len(unique)
        print(f"  Circuit size:      {len(circuit.neurons)} neurons across {num_layers} layers")

        print("  Normal generation...", end="", flush=True)
        t0 = time.time()
        steerer.generate("Hello", max_new_tokens=max_new_tokens)
        elapsed = time.time() - t0
        print(f"{' ' * 22}{ok('OK', use_color)} ({elapsed:.1f}s)")

        print("  Steered generation (ablate)...", end="", flush=True)
        t0 = time.time()
        steerer.steer("Hello", feature="test_circuit", multiplier=0.0, max_new_tokens=max_new_tokens)
        elapsed = time.time() - t0
        print(f"{' ' * 14}{ok('OK', use_color)} ({elapsed:.1f}s)")

        print(f"  Result: {ok('PASS', use_color)}")
        print()
        return True, None

    except Exception as exc:
        failure_reason = str(exc)
        print()
        print(f"  Result: {fail('FAIL', use_color)} — {exc}")
        print()
        print("  Traceback:")
        traceback.print_exc()
        print()
        return False, failure_reason


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check HuggingFace model compatibility with neuron_steer.",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--level",
        default="all",
        choices=LEVEL_ORDER + ["all"],
        help="Highest check level to run (default: all)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Max new tokens for generation tests (default: 40)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output",
    )
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()
    levels = resolve_levels(args.level)

    print_header(args.model, use_color)

    stopped_at = None
    failure_reason = None
    steerer = None
    config_info = {}
    levels_completed: list[str] = []

    if "config" in levels:
        passed, failure_reason, config_info = check_config(args.model, use_color)
        if not passed:
            stopped_at = "Level 1"
            print_verdict(False, args.model, failure_reason or "Config check failed", stopped_at, use_color, levels_completed)
            return 1
        levels_completed.append("config")

    if "structure" in levels:
        passed, failure_reason, _ = check_structure(args.model, use_color, config_info)
        if not passed:
            stopped_at = "Level 2"
            print_verdict(False, args.model, failure_reason or "Structure check failed", stopped_at, use_color, levels_completed)
            return 1
        levels_completed.append("structure")

    if "constructor" in levels:
        passed, failure_reason, steerer = check_constructor(args.model, use_color)
        if not passed:
            stopped_at = "Level 3"
            print_verdict(False, args.model, failure_reason or "Constructor check failed", stopped_at, use_color, levels_completed)
            return 1
        levels_completed.append("constructor")

    if "steering" in levels:
        if steerer is None:
            stopped_at = "Level 4"
            print_verdict(False, args.model, "NeuronSteerer not available", stopped_at, use_color, levels_completed)
            return 1
        passed, failure_reason = check_steering(steerer, args.max_new_tokens, use_color)
        if not passed:
            stopped_at = "Level 4"
            print_verdict(False, args.model, failure_reason or "Steering check failed", stopped_at, use_color, levels_completed)
            return 1
        levels_completed.append("steering")

    if steerer is not None:
        del steerer
        cleanup_gpu()

    print_verdict(True, args.model, "", "", use_color, levels_completed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
