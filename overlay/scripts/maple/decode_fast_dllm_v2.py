#!/usr/bin/env python3
"""Correctness-first Fast-dLLM v2 decoder for a gated Maple checkpoint."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.auto import build_foundation_model
from veomni.models.transformers.maple.modeling_utils import ReadOnlyCache, materialize_ternary_parameters


@torch.inference_mode()
def prefill_prefix(model, input_ids, *, bdlm_decode=True):
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
    return model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=positions,
        use_cache=True,
        bdlm_decode=bdlm_decode,
        logits_to_keep=1,
    )


@torch.inference_mode()
def decode(
    model,
    input_ids,
    *,
    mask_id,
    eos_id,
    max_new_tokens,
    block_size,
    subblock_size,
    threshold,
    prefix_cache=None,
    prefix_logits=None,
    use_prefix_cache=True,
    stats=None,
):
    prompt_length = input_ids.shape[1]
    target_length = prompt_length + max_new_tokens
    denoise_forwards = 0
    cache_update_forwards = 0
    accepted_tokens = 0
    if use_prefix_cache and prefix_cache is None:
        prefill = prefill_prefix(model, input_ids)
        prefix_cache = prefill.past_key_values
        prefix_logits = prefill.logits[:, -1:]
    if use_prefix_cache and prefix_logits is None:
        raise ValueError("prefix_logits are required with a caller-provided prefix_cache")
    while input_ids.shape[1] < target_length:
        block_start = input_ids.shape[1]
        fill = min(block_size, target_length - block_start)
        input_ids = torch.cat((input_ids, input_ids.new_full((input_ids.shape[0], fill), mask_id)), dim=1)
        block_input = input_ids[:, block_start:]
        block_attention = torch.ones_like(block_input)
        block_positions = torch.arange(block_start, input_ids.shape[1], device=input_ids.device).expand_as(block_input)
        block_cache = ReadOnlyCache(prefix_cache) if use_prefix_cache else None

        for subblock_start in range(block_start, input_ids.shape[1], subblock_size):
            subblock_end = min(subblock_start + subblock_size, input_ids.shape[1])
            while input_ids[:, subblock_start:subblock_end].eq(mask_id).any():
                if use_prefix_cache:
                    relative_start = subblock_start - block_start
                    relative_end = subblock_end - block_start
                    logit_positions = torch.arange(
                        max(0, relative_start - 1), relative_end - 1, device=input_ids.device
                    )
                    output = model(
                        input_ids=block_input,
                        attention_mask=block_attention,
                        position_ids=block_positions,
                        past_key_values=block_cache,
                        use_cache=False,
                        bdlm_decode=True,
                        logits_to_keep=logit_positions,
                    )
                    logits = output.logits
                    if relative_start == 0:
                        logits = torch.cat((prefix_logits, logits), dim=1)
                else:
                    positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
                    logit_positions = torch.arange(subblock_start - 1, subblock_end - 1, device=input_ids.device)
                    output = model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        position_ids=positions,
                        use_cache=False,
                        bdlm_decode=True,
                        logits_to_keep=logit_positions,
                    )
                    logits = output.logits
                denoise_forwards += 1
                probabilities, candidates = logits.softmax(-1).max(-1)
                masked = input_ids[:, subblock_start:subblock_end].eq(mask_id)
                confidence = probabilities.masked_fill(~masked, -torch.inf)
                reveal = (confidence > threshold) & masked
                unfinished = masked.any(-1)
                rows = torch.where(unfinished)[0]
                reveal[rows, confidence[rows].argmax(-1)] = True
                span = input_ids[:, subblock_start:subblock_end]
                span[reveal] = candidates[reveal]
                accepted_tokens += int(reveal.sum())
                del output, logits

        if use_prefix_cache:
            del block_cache
            block_input = input_ids[:, block_start:]
            output = model(
                input_ids=block_input,
                attention_mask=block_attention,
                position_ids=block_positions,
                past_key_values=prefix_cache,
                use_cache=True,
                bdlm_decode=True,
                logits_to_keep=1,
            )
            prefix_cache = output.past_key_values
            prefix_logits = output.logits[:, -1:]
            cache_update_forwards += 1

        generated = input_ids[0, prompt_length:]
        eos = torch.where(generated.eq(eos_id))[0]
        if eos.numel():
            input_ids = input_ids[:, : prompt_length + int(eos[0]) + 1]
            break
    if stats is not None:
        batch_size = input_ids.shape[0]
        stats.update(
            denoise_forwards=denoise_forwards,
            cache_update_forwards=cache_update_forwards,
            accepted_tokens=accepted_tokens,
            tokens_per_forward=accepted_tokens / max(denoise_forwards * batch_size, 1),
            prefix_cache=use_prefix_cache,
        )
    return input_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--subblock-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    if args.block_size % args.subblock_size:
        raise ValueError("block-size must be divisible by subblock-size.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="right",
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    ops = OpsImplementationConfig(
        attn_implementation="flex_attention",
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
    materialize_ternary_parameters(model)
    mask_id = model.config.mask_token_id
    if mask_id is None:
        raise ValueError("The model config does not define mask_token_id.")
    results = []
    for prompt_text in args.prompt:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not isinstance(prompt, torch.Tensor):
            prompt = prompt.input_ids
        prompt = prompt.cuda()
        started = time.perf_counter()
        decode_stats = {}
        output = decode(
            model,
            prompt,
            mask_id=mask_id,
            eos_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            subblock_size=args.subblock_size,
            threshold=args.threshold,
            stats=decode_stats,
        )
        seconds = time.perf_counter() - started
        generated = output[0, prompt.shape[1] :]
        result = {
            "prompt": prompt_text,
            "text": tokenizer.decode(generated, skip_special_tokens=True),
            "generated_tokens": generated.numel(),
            "mask_tokens_remaining": generated.eq(mask_id).sum().item(),
            "seconds": seconds,
            "tokens_per_second": generated.numel() / seconds,
            **decode_stats,
        }
        results.append(result)
        print(result["text"])
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
