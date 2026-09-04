#!/usr/bin/env python3
"""Zero-shot, one-token MMLU evaluation for Maple AR and BDLM checkpoints."""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.auto import build_foundation_model


LETTERS = "ABCD"


def format_question(question: str, choices: list[str], subject: str) -> str:
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices, strict=True)
    )
    return (
        f"Answer the following multiple-choice question about {subject.replace('_', ' ')}. "
        f"Respond with only A, B, C, or D.\n\n{question}\n{options}"
    )


def self_test() -> None:
    assert format_question(
        "2+2?", ["1", "2", "3", "4"], "elementary_mathematics"
    ).endswith("D. 4")
    print("self-test passed")


def save_results(path: Path, summary: dict, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2) + "\n"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--mode", choices=("ar", "bdlm"), default="bdlm")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if not args.model or not args.output_json:
        parser.error("--model and --output-json are required")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
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
        dataset = dataset.shuffle(seed=args.seed).select(
            range(min(args.limit, len(dataset)))
        )
    dataset = dataset.shard(args.num_shards, args.shard_index, contiguous=False)

    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, fix_mistral_regex=True
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
    mask_id = model.config.mask_token_id
    if mask_id is None:
        raise ValueError("The model config does not define mask_token_id.")
    choice_ids = [
        tokenizer.encode(f" {letter}", add_special_tokens=False) for letter in LETTERS
    ]
    if any(len(ids) != 1 for ids in choice_ids):
        raise ValueError(f"Expected single-token answer choices, got {choice_ids}.")
    choice_ids = [ids[0] for ids in choice_ids]

    run = None
    if args.wandb_name:
        import wandb

        run = wandb.init(
            project="maple-bdlm",
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        )

    records = []
    correct = 0
    started = time.perf_counter()
    for row in dataset:
        prompt_text = format_question(row["question"], row["choices"], row["subject"])
        messages = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        input_ids = tokenizer(
            prompt_text + "<|im_start|>assistant\nThe answer is",
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.cuda()
        item_started = time.perf_counter()
        model_input = input_ids
        if args.mode == "bdlm":
            model_input = torch.cat(
                (input_ids, torch.full((1, 1), mask_id, device=input_ids.device)), dim=1
            )
        positions = torch.arange(
            model_input.shape[1], device=input_ids.device
        ).unsqueeze(0)
        output = model(
            input_ids=model_input,
            attention_mask=torch.ones_like(model_input),
            position_ids=positions,
            use_cache=False,
            bdlm_decode=args.mode == "bdlm",
        )
        prediction = LETTERS[
            output.logits[0, -2 if args.mode == "bdlm" else -1, choice_ids]
            .argmax()
            .item()
        ]
        gold = LETTERS[row["answer"]]
        is_correct = prediction == gold
        correct += is_correct
        record = {
            "sample_id": row["sample_id"],
            "subject": row["subject"],
            "gold": gold,
            "prediction": prediction,
            "correct": is_correct,
            "seconds": time.perf_counter() - item_started,
        }
        records.append(record)
        count = len(records)
        if count % 10 == 0 or count == len(dataset):
            metrics = {
                "eval/accuracy": correct / count,
                "eval/examples": count,
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
        "correct": correct,
        "examples": len(records),
        "examples_per_second": len(records) / elapsed,
        "seconds": elapsed,
        "seed": args.seed,
        "num_fewshot": 0,
        "max_new_tokens": 1,
        "one_token": True,
        "block_size": args.block_size if args.mode == "bdlm" else None,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "subjects": subjects,
        "complete": True,
    }
    save_results(args.output_json, summary, records)
    if run:
        run.log(
            {
                f"final/{key}": value
                for key, value in summary.items()
                if isinstance(value, (int, float))
            }
        )
        run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
