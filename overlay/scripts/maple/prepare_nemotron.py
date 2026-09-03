#!/usr/bin/env python3
"""Prepare an exact, block-aligned Nemotron mixture for Maple Fast-dLLM v2."""

import argparse
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, IterableDataset, load_dataset
from transformers import AutoTokenizer

from veomni.data.chat_template import TokenizerTemplate


DATASET = "nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2"
DATASET_REVISION = "7d7a14dbc1ec673e9fad558785d6d2ccd4651fe8"
MODEL = "deepgrove/maple-preview"
MODEL_REVISION = "ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07"
MIXTURE = {"chat": 4, "code": 6, "math": 5, "stem": 5}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=640_000)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", type=Path)
    return parser.parse_args()


def valid_messages(messages):
    return (
        isinstance(messages, list)
        and len(messages) >= 2
        and messages[-1].get("role") == "assistant"
        and all(message.get("role") in {"system", "user", "assistant"} for message in messages)
        and all(
            isinstance(message.get("content"), str)
            and (message["role"] == "system" or message["content"].strip())
            for message in messages
        )
    )


def packed_sequences(split, template, tokenizer, args):
    if args.dataset_dir is None:
        stream = load_dataset(DATASET, split="train", streaming=True, revision=DATASET_REVISION)
        stream = stream.filter(lambda sample: sample.get("category") == split)
    else:
        files = sorted((args.dataset_dir / "data").glob(f"{split}-*.parquet"))
        if files:

            def rows():
                for path in files:
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["messages"]):
                        yield from batch.to_pylist()

        else:
            files = sorted((args.dataset_dir / "train").glob("*.arrow"))
            if not files:
                raise FileNotFoundError(f"No local Nemotron shards found for category {split!r}.")

            def rows():
                for path in files:
                    yield from (sample for sample in Dataset.from_file(str(path)) if sample.get("category") == split)

        stream = IterableDataset.from_generator(rows)
    stream = stream.shuffle(seed=args.seed + list(MIXTURE).index(split), buffer_size=args.shuffle_buffer)
    fields = {name: [] for name in ("input_ids", "attention_mask", "labels", "position_ids")}

    for sample in stream:
        messages = sample.get("messages")
        if not valid_messages(messages):
            continue
        encoded = template.encode_messages(messages, max_seq_len=args.sequence_length + 1)
        length = len(encoded["input_ids"])
        if length == 0 or length > args.sequence_length or not any(label != -100 for label in encoded["labels"]):
            continue
        padded_length = math.ceil(length / args.block_size) * args.block_size
        if len(fields["input_ids"]) + padded_length > args.sequence_length:
            tail = args.sequence_length - len(fields["input_ids"])
            fields["input_ids"].extend([tokenizer.pad_token_id] * tail)
            fields["attention_mask"].extend([0] * tail)
            fields["labels"].extend([-100] * tail)
            fields["position_ids"].extend(range(len(fields["position_ids"]), args.sequence_length))
            yield fields
            fields = {name: [] for name in fields}

        fields["input_ids"].extend(encoded["input_ids"])
        fields["attention_mask"].extend(encoded["attention_mask"])
        fields["labels"].extend(encoded["labels"])
        fields["position_ids"].extend(range(length))
        padding = padded_length - length
        fields["input_ids"].extend([tokenizer.pad_token_id] * padding)
        fields["attention_mask"].extend([0] * padding)
        fields["labels"].extend([-100] * padding)
        fields["position_ids"].extend(range(length, padded_length))

    if fields["input_ids"]:
        tail = args.sequence_length - len(fields["input_ids"])
        fields["input_ids"].extend([tokenizer.pad_token_id] * tail)
        fields["attention_mask"].extend([0] * tail)
        fields["labels"].extend([-100] * tail)
        fields["position_ids"].extend(range(len(fields["position_ids"]), args.sequence_length))
        yield fields


def main():
    args = parse_args()
    if args.sequence_length % args.block_size:
        raise ValueError("sequence-length must be divisible by block-size.")
    cycle = [split for split, count in MIXTURE.items() for _ in range(count)]
    if args.num_sequences % len(cycle):
        raise ValueError(f"num-sequences must be divisible by {len(cycle)} for an exact mixture.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, trust_remote_code=True)
    tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
    mask_token_id = tokenizer.convert_tokens_to_ids("<|mask|>")
    if mask_token_id >= 151936:
        raise ValueError(f"The added mask token id {mask_token_id} exceeds Maple's padded vocabulary.")
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    template = TokenizerTemplate(tokenizer)
    generators = {split: iter(packed_sequences(split, template, tokenizer, args)) for split in MIXTURE}

    rows = []
    shard = 0
    for index in range(args.num_sequences):
        split = cycle[index % len(cycle)]
        try:
            row = next(generators[split])
        except StopIteration as error:
            raise RuntimeError(f"Nemotron split {split!r} exhausted before its mixture quota.") from error
        row["source"] = split
        rows.append(row)
        if len(rows) == args.shard_size or index + 1 == args.num_sequences:
            pq.write_table(pa.Table.from_pylist(rows), data_dir / f"train-{shard:05d}.parquet")
            rows.clear()
            shard += 1

    manifest = {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "mixture": {key: value / len(cycle) for key, value in MIXTURE.items()},
        "num_sequences": args.num_sequences,
        "sequence_length": args.sequence_length,
        "block_size": args.block_size,
        "shard_size": args.shard_size,
        "mask_token_id": mask_token_id,
        "source_token_slots": args.num_sequences * args.sequence_length,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
