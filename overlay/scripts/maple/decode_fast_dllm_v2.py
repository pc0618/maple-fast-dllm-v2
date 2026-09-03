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


@torch.no_grad()
def decode(model, input_ids, *, mask_id, eos_id, max_new_tokens, block_size, subblock_size, threshold):
    prompt_length = input_ids.shape[1]
    target_length = prompt_length + max_new_tokens
    while input_ids.shape[1] < target_length:
        fill = min(block_size - input_ids.shape[1] % block_size, target_length - input_ids.shape[1])
        input_ids = torch.cat((input_ids, torch.full((1, fill), mask_id, device=input_ids.device)), dim=1)
        block_start = input_ids.shape[1] - input_ids.shape[1] % block_size
        if block_start == input_ids.shape[1]:
            block_start -= block_size

        for subblock_start in range(block_start, input_ids.shape[1], subblock_size):
            subblock_end = min(subblock_start + subblock_size, input_ids.shape[1])
            while input_ids[:, subblock_start:subblock_end].eq(mask_id).any():
                positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
                output = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    position_ids=positions,
                    use_cache=False,
                    bdlm_decode=True,
                )
                shifted_logits = torch.cat((output.logits[:, :1], output.logits[:, :-1]), dim=1)
                probabilities, candidates = shifted_logits[:, subblock_start:subblock_end].softmax(-1).max(-1)
                masked = input_ids[:, subblock_start:subblock_end].eq(mask_id)
                confidence = probabilities.masked_fill(~masked, -torch.inf)
                reveal = (confidence > threshold) & masked
                reveal[:, confidence.argmax(-1)] = True
                span = input_ids[:, subblock_start:subblock_end]
                span[reveal] = candidates[reveal]

        generated = input_ids[0, prompt_length:]
        eos = torch.where(generated.eq(eos_id))[0]
        if eos.numel():
            return input_ids[:, : prompt_length + int(eos[0]) + 1]
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
    results = []
    for prompt_text in args.prompt:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not isinstance(prompt, torch.Tensor):
            prompt = prompt.input_ids
        prompt = prompt.cuda()
        started = time.perf_counter()
        output = decode(
            model,
            prompt,
            mask_id=tokenizer.mask_token_id,
            eos_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            subblock_size=args.subblock_size,
            threshold=args.threshold,
        )
        generated = output[0, prompt.shape[1] :]
        result = {
            "prompt": prompt_text,
            "text": tokenizer.decode(generated, skip_special_tokens=True),
            "generated_tokens": generated.numel(),
            "mask_tokens_remaining": generated.eq(tokenizer.mask_token_id).sum().item(),
            "seconds": time.perf_counter() - started,
        }
        results.append(result)
        print(result["text"])
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
