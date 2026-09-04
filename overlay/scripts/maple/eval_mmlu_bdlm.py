#!/usr/bin/env python3
"""Generation-based MMLU evaluation for Maple AR and BDLM checkpoints."""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from scripts.maple.decode_fast_dllm_v2 import decode
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.auto import build_foundation_model
from veomni.models.transformers.maple.modeling_utils import materialize_ternary_parameters


LETTERS = "ABCD"
ANSWER_PATTERNS = (
    re.compile(r"\\boxed\s*\{\s*(?:\\text\s*\{\s*)?([A-D])", re.IGNORECASE),
    re.compile(r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:|-)?\s*\**\(?([A-D])\b", re.IGNORECASE),
)


def extract_answer(text: str) -> str | None:
    matches = [
        (match.start(), match.group(1).upper()) for pattern in ANSWER_PATTERNS for match in pattern.finditer(text)
    ]
    return max(matches)[1] if matches else None


def format_question(question: str, choices: list[str], subject: str, *, one_token: bool = False) -> str:
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices, strict=True))
    instruction = (
        "Respond with only A, B, C, or D."
        if one_token
        else "Think step by step, then end your response with exactly \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
    )
    return (
        f"Answer the following multiple-choice question about {subject.replace('_', ' ')}. "
        f"{instruction}\n\n{question}\n{options}"
    )


def self_test() -> None:
    assert extract_answer("reasoning... \\boxed{C}") == "C"
    assert extract_answer("The final answer is **B**.") == "B"
    assert extract_answer("A and D are discussed without a conclusion") is None
    assert format_question("2+2?", ["1", "2", "3", "4"], "elementary_mathematics").endswith("D. 4")
    print("self-test passed")


def save_results(path: Path, summary: dict, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"summary": summary, "records": records}, indent=2) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--mode", choices=("ar", "bdlm"), default="bdlm")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--subblock-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--reveal-per-forward", type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--save-text", action="store_true")
    parser.add_argument("--one-token", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if not args.model or not args.output_json:
        parser.error("--model and --output-json are required")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.block_size % args.subblock_size:
        parser.error("--block-size must be divisible by --subblock-size")
    return args


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    dataset = load_dataset("cais/mmlu", "all", split="test")
    dataset = dataset.add_column("sample_id", range(len(dataset)))
    if args.limit:
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.limit, len(dataset))))
    dataset = dataset.shard(args.num_shards, args.shard_index, contiguous=False)

    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, fix_mistral_regex=True)
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
    choice_ids = [tokenizer.encode(f" {letter}", add_special_tokens=False) for letter in LETTERS]
    if any(len(ids) != 1 for ids in choice_ids):
        raise ValueError(f"Expected single-token answer choices, got {choice_ids}.")
    choice_ids = [ids[0] for ids in choice_ids]

    run = None
    if args.wandb_name:
        import wandb

        run = wandb.init(project="maple-bdlm", entity=args.wandb_entity, name=args.wandb_name, config=vars(args))

    records = []
    correct = parsed = generated_tokens = 0
    started = time.perf_counter()
    for row in dataset:
        prompt_text = format_question(row["question"], row["choices"], row["subject"], one_token=args.one_token)
        messages = [{"role": "user", "content": prompt_text}]
        if args.one_token:
            prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            prompt = tokenizer(
                prompt_text + "<|im_start|>assistant\nThe answer is",
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids
        else:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = (prompt if isinstance(prompt, torch.Tensor) else prompt.input_ids).cuda()
        item_started = time.perf_counter()
        if args.one_token:
            model_input = input_ids
            if args.mode == "bdlm":
                model_input = torch.cat((input_ids, torch.full((1, 1), mask_id, device=input_ids.device)), dim=1)
            positions = torch.arange(model_input.shape[1], device=input_ids.device).unsqueeze(0)
            output = model(
                input_ids=model_input,
                attention_mask=torch.ones_like(model_input),
                position_ids=positions,
                use_cache=False,
                bdlm_decode=args.mode == "bdlm",
            )
            prediction = LETTERS[output.logits[0, -2 if args.mode == "bdlm" else -1, choice_ids].argmax().item()]
            tokens = torch.tensor([choice_ids[LETTERS.index(prediction)]], device=input_ids.device)
            text = f" {prediction}"
        elif args.mode == "ar":
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            output = decode(
                model,
                input_ids,
                mask_id=mask_id,
                eos_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=args.block_size,
                subblock_size=args.subblock_size,
                threshold=args.threshold,
                reveal_per_forward=args.reveal_per_forward,
            )
        if not args.one_token:
            tokens = output[0, input_ids.shape[1] :]
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            prediction = extract_answer(text)
        gold = LETTERS[row["answer"]]
        is_correct = prediction == gold
        correct += is_correct
        parsed += prediction is not None
        generated_tokens += tokens.numel()
        record = {
            "sample_id": row["sample_id"],
            "subject": row["subject"],
            "gold": gold,
            "prediction": prediction,
            "correct": is_correct,
            "generated_tokens": tokens.numel(),
            "seconds": time.perf_counter() - item_started,
        }
        if args.save_text:
            record["text"] = text
        records.append(record)
        count = len(records)
        if count % 10 == 0 or count == len(dataset):
            metrics = {
                "eval/accuracy": correct / count,
                "eval/parse_rate": parsed / count,
                "eval/examples": count,
                "eval/generated_tokens": generated_tokens,
            }
            print(json.dumps(metrics), flush=True)
            if run:
                run.log(metrics, step=count)
        if count % 100 == 0:
            save_results(args.output_json, {"complete": False, **metrics}, records)

    elapsed = time.perf_counter() - started
    subjects = {}
    for subject in sorted({record["subject"] for record in records}):
        subset = [record for record in records if record["subject"] == subject]
        subjects[subject] = {
            "accuracy": sum(record["correct"] for record in subset) / len(subset),
            "count": len(subset),
        }
    summary = {
        "model": args.model,
        "mode": args.mode,
        "accuracy": correct / len(records),
        "parse_rate": parsed / len(records),
        "correct": correct,
        "examples": len(records),
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / elapsed,
        "examples_per_second": len(records) / elapsed,
        "seconds": elapsed,
        "seed": args.seed,
        "num_fewshot": 0,
        "max_new_tokens": 1 if args.one_token else args.max_new_tokens,
        "one_token": args.one_token,
        "block_size": args.block_size if args.mode == "bdlm" else None,
        "subblock_size": args.subblock_size if args.mode == "bdlm" and not args.one_token else None,
        "threshold": args.threshold if args.mode == "bdlm" and not args.one_token else None,
        "reveal_per_forward": args.reveal_per_forward if args.mode == "bdlm" and not args.one_token else None,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "subjects": subjects,
        "complete": True,
    }
    save_results(args.output_json, summary, records)
    if run:
        run.log({f"final/{key}": value for key, value in summary.items() if isinstance(value, (int, float))})
        run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
