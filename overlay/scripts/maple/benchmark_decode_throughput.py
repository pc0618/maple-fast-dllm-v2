#!/usr/bin/env python3
"""Measure Maple AR and Fast-dLLM-v2 prefill/decode throughput."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from scripts.maple.decode_fast_dllm_v2 import decode, prefill_prefix
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.auto import build_foundation_model
from veomni.models.transformers.maple.modeling_utils import materialize_ternary_parameters


def timed(call):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("ar", "bdlm"), required=True)
    parser.add_argument("--isl", type=int, required=True)
    parser.add_argument("--osl", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prefill-repeats", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--subblock-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--attn-implementation", choices=("flex_attention", "sdpa"))
    args = parser.parse_args()
    if min(args.isl, args.osl, args.batch_size, args.prefill_repeats) < 1:
        parser.error("ISL, OSL, batch size, and prefill repeats must be positive")

    torch.cuda.set_device(args.device)
    torch.manual_seed(1234)
    attn_implementation = args.attn_implementation or ("sdpa" if args.mode == "ar" else "flex_attention")
    ops = OpsImplementationConfig(
        attn_implementation=attn_implementation,
        moe_implementation="fused_quack",
        load_balancing_loss_implementation="eager",
    )
    model = build_foundation_model(
        args.model,
        weights_path=args.model,
        torch_dtype="bfloat16",
        init_device="cuda",
        config_kwargs={
            "training_objective": "fast_dllm_v2",
            "bdlm_block_size": args.block_size,
            "qat_ternary": True,
        },
        ops_implementation=ops,
    ).eval()
    materialized_parameters = materialize_ternary_parameters(model)
    input_ids = torch.randint(100, model.config.vocab_size - 1024, (args.batch_size, args.isl), device="cuda")

    def prefill():
        return prefill_prefix(model, input_ids, bdlm_decode=args.mode == "bdlm")

    warmup = prefill()
    del warmup
    prefill_times = []
    for _ in range(args.prefill_repeats):
        output, seconds = timed(prefill)
        prefill_times.append(seconds)
        del output

    if args.mode == "ar":
        warmup = prefill()
        token = warmup.logits[:, -1].argmax(-1, keepdim=True)
        cache = warmup.past_key_values
        for _ in range(8):
            warmup = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            token = warmup.logits[:, -1].argmax(-1, keepdim=True)
            cache = warmup.past_key_values
        del warmup, cache

        torch.cuda.reset_peak_memory_stats()
        initial = prefill()
        token = initial.logits[:, -1].argmax(-1, keepdim=True)
        cache = initial.past_key_values

        def run_decode():
            nonlocal token, cache
            for _ in range(args.osl):
                output = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                token = output.logits[:, -1].argmax(-1, keepdim=True)
                cache = output.past_key_values

        _, decode_seconds = timed(run_decode)
    else:
        mask_id = model.config.mask_token_id
        if mask_id is None:
            raise ValueError("The model config does not define mask_token_id")
        warmup = prefill()
        decode(
            model,
            input_ids,
            mask_id=mask_id,
            eos_id=-1,
            max_new_tokens=args.block_size,
            block_size=args.block_size,
            subblock_size=args.subblock_size,
            threshold=args.threshold,
            prefix_cache=warmup.past_key_values,
            prefix_logits=warmup.logits[:, -1:],
        )
        del warmup

        torch.cuda.reset_peak_memory_stats()
        initial = prefill()
        decode_stats = {}

        def run_decode():
            return decode(
                model,
                input_ids,
                mask_id=mask_id,
                eos_id=-1,
                max_new_tokens=args.osl,
                block_size=args.block_size,
                subblock_size=args.subblock_size,
                threshold=args.threshold,
                prefix_cache=initial.past_key_values,
                prefix_logits=initial.logits[:, -1:],
                stats=decode_stats,
            )

        generated, decode_seconds = timed(run_decode)
        if generated.shape[1] != args.isl + args.osl or generated[:, args.isl :].eq(mask_id).any():
            raise RuntimeError("BDLM decode did not produce the requested number of completed tokens")

    prefill_seconds = statistics.median(prefill_times)
    result = {
        "model": args.model,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "isl": args.isl,
        "osl": args.osl,
        "prefill_seconds_median": prefill_seconds,
        "prefill_tokens_per_second": args.batch_size * args.isl / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": args.batch_size * args.osl / decode_seconds,
        "prefill_repeats": args.prefill_repeats,
        "block_size": args.block_size if args.mode == "bdlm" else None,
        "subblock_size": args.subblock_size if args.mode == "bdlm" else None,
        "threshold": args.threshold if args.mode == "bdlm" else None,
        "denoise_forwards": decode_stats.get("denoise_forwards") if args.mode == "bdlm" else None,
        "cache_update_forwards": decode_stats.get("cache_update_forwards") if args.mode == "bdlm" else None,
        "tokens_per_forward": decode_stats.get("tokens_per_forward") if args.mode == "bdlm" else None,
        "prefix_cache": args.mode == "bdlm",
        "attn_implementation": attn_implementation,
        "materialized_ternary_parameters": materialized_parameters,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "gpu": torch.cuda.get_device_name(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
