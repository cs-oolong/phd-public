import argparse
import gc
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive multi-turn chat with a Hugging Face causal LM."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model name or local path",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Disable sampling (greedy decoding)",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "int8", "int4"],
        default="none",
        help="Quantization mode for GPU loading (none, int8, int4)",
    )
    return parser.parse_args()


def get_model_device(model):
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def cleanup_gpu(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_generation_kwargs(args):
    kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.greedy:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
    return kwargs


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if torch.cuda.is_available():
        if args.quantize == "int8":
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        elif args.quantize == "int4":
            load_kwargs["load_in_4bit"] = True
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    model.eval()
    device = get_model_device(model)
    gen_kwargs = build_generation_kwargs(args)

    if args.greedy:
        gen_settings = f"greedy, max_new_tokens={args.max_new_tokens}"
    else:
        gen_settings = (
            f"max_new_tokens={args.max_new_tokens}, "
            f"temperature={args.temperature}, top_p={args.top_p}"
        )

    print("=" * 60)
    print("Interactive chat")
    print(f"Model:        {args.model}")
    print(f"Device:       {device}")
    print(f"Quantization: {args.quantize}")
    print(f"Generation:   {gen_settings}")
    print("=" * 60)
    print('Commands: "quit"/"exit" to leave, "clear" to reset, "system: ..." for system prompt')
    print()

    messages = []
    system_prompt = None

    try:
        while True:
            try:
                user_input = input(">>> ").strip()
            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                print()
                break

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ("quit", "exit"):
                break
            if cmd == "clear":
                messages = []
                system_prompt = None
                print("Conversation history cleared.")
                continue
            if user_input.lower().startswith("system:"):
                system_prompt = user_input[7:].strip()
                print(f"System prompt set: {system_prompt!r}")
                continue

            messages.append({"role": "user", "content": user_input})

            chat_messages = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            chat_messages.extend(messages)

            try:
                inputs = tokenizer.apply_chat_template(
                    chat_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(device)

                input_len = inputs["input_ids"].shape[-1]
                start = time.perf_counter()
                inputs.pop("token_type_ids", None)
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                elapsed = time.perf_counter() - start

                new_tokens = outputs[0][input_len:]
                num_new = new_tokens.shape[0]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                messages.append({"role": "assistant", "content": response})

                print(response)
                if elapsed > 0:
                    tps = num_new / elapsed
                    print(f"\n[{elapsed:.2f}s | {num_new} tokens | {tps:.1f} tok/s]")
                else:
                    print(f"\n[{elapsed:.2f}s | {num_new} tokens]")
                print()

            except Exception:
                messages.pop()
                print(traceback.format_exc())

    finally:
        cleanup_gpu(model)
        print("Goodbye!")


if __name__ == "__main__":
    main()
