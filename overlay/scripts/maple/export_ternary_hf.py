#!/usr/bin/env python3
"""Export a Maple QAT DCP checkpoint as standard ternary-valued HF safetensors."""

import argparse
import gc
import json
import shutil
from collections import OrderedDict
from pathlib import Path

from safetensors.torch import save_file

from veomni.checkpoint.dcp_checkpointer import _get_sharding_plan, _process_shard
from veomni.models.transformers.maple.modeling_utils import twn_torch_ref


def public_tensors(name, tensor):
    if name == "model.embed_tokens.weight":
        yield "model.word_embeddings.weight", tensor
        return
    attention_suffix = ".parametrizations.weight.original"
    if ".self_attn." in name and name.endswith(attention_suffix):
        yield name.removesuffix(attention_suffix) + ".weight", twn_torch_ref(tensor)
        return
    gate_up_suffix = ".experts.parametrizations.gate_up_proj.original"
    down_suffix = ".experts.parametrizations.down_proj.original"
    if name.endswith(gate_up_suffix):
        prefix = name.removesuffix(gate_up_suffix) + ".experts"
        gate, up = twn_torch_ref(tensor).chunk(2, dim=1)
        for expert in range(tensor.shape[0]):
            yield f"{prefix}.{expert}.gate_proj.weight", gate[expert]
            yield f"{prefix}.{expert}.up_proj.weight", up[expert]
        return
    if name.endswith(down_suffix):
        prefix = name.removesuffix(down_suffix) + ".experts"
        tensor = twn_torch_ref(tensor)
        for expert in range(tensor.shape[0]):
            yield f"{prefix}.{expert}.down_proj.weight", tensor[expert]
        return
    yield name, tensor


def copy_assets(source: Path, destination: Path):
    for path in source.iterdir():
        if path.is_file() and not path.name.endswith((".safetensors", ".bin")) and "safetensors.index" not in path.name:
            shutil.copy2(path, destination / path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-assets", type=Path, required=True)
    parser.add_argument("--tokenizer-assets", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=5_000_000_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_assets(args.model_assets, args.output_dir)
    copy_assets(args.tokenizer_assets, args.output_dir)
    shards, total_size, _ = _get_sharding_plan(args.checkpoint, args.shard_size, "bfloat16")
    weight_map = OrderedDict()
    shard_count = len(shards)
    for shard_index, shard in enumerate(shards, start=1):
        loaded = _process_shard(shard, args.checkpoint, "bfloat16")
        exported = OrderedDict()
        for name, tensor in loaded.items():
            for public_name, public_tensor in public_tensors(name, tensor):
                exported[public_name] = public_tensor.contiguous().clone()
        filename = f"model-{shard_index:05d}-of-{shard_count:05d}.safetensors"
        save_file(exported, args.output_dir / filename, metadata={"format": "pt"})
        weight_map.update((name, filename) for name in exported)
        del loaded, exported
        gc.collect()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (args.output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    config_path = args.output_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["mask_token_id"] = 151669
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
