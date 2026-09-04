# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Small runtime helpers for Maple ternary QAT and Fast-dLLM v2 training."""

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.utils import parametrize
from transformers.cache_utils import Cache


def twn_torch_ref(weight: torch.Tensor) -> torch.Tensor:
    """Map each row to ``{-alpha, 0, +alpha}`` using Maple's TWN rule."""
    weight_fp = weight.float()
    abs_weight = weight_fp.abs()
    mask = abs_weight > abs_weight.mean(dim=-1, keepdim=True) * 0.7
    mask_fp = mask.float()
    alpha = (abs_weight * mask_fp).sum(dim=-1, keepdim=True) / mask_fp.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return (weight_fp.sign() * mask_fp * alpha).to(weight.dtype)


class QuantizeTernary(torch.autograd.Function):
    """Ternary forward with a straight-through backward."""

    @staticmethod
    @torch.compile(fullgraph=True)
    def forward(ctx, input):
        return twn_torch_ref(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class TernaryParametrization(nn.Module):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return QuantizeTernary.apply(weight)


def ternarize_parameter(module: nn.Module, name: str = "weight") -> None:
    """Keep a latent parameter and expose its ternary STE value at ``module.<name>``."""
    parametrize.register_parametrization(module, name, TernaryParametrization(), unsafe=True)


def materialize_ternary_parameters(module: nn.Module) -> int:
    """Replace inference-time QAT parametrizations with their ternary values."""
    materialized = 0
    for child in module.modules():
        for name in ("weight", "gate_up_proj", "down_proj"):
            if parametrize.is_parametrized(child, name):
                parametrize.remove_parametrizations(child, name, leave_parametrized=True)
                materialized += 1
    return materialized


class ReadOnlyCache(Cache):
    """Expose a completed-prefix cache without appending the noisy block to it."""

    def __init__(self, source: Cache):
        super().__init__(layers=source.layers)
        self.source = source
        self.buffers = {}

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        layer = self.layers[layer_idx]
        if not layer.is_initialized:
            return key_states, value_states
        if layer_idx not in self.buffers:
            self.buffers[layer_idx] = (
                torch.cat((layer.keys, key_states), dim=-2),
                torch.cat((layer.values, value_states), dim=-2),
            )
        else:
            keys, values = self.buffers[layer_idx]
            keys[..., layer.keys.shape[-2] :, :].copy_(key_states)
            values[..., layer.values.shape[-2] :, :].copy_(value_states)
        return self.buffers[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.source.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return self.source.get_mask_sizes(query_length, layer_idx)


def prepare_fast_dllm_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    *,
    block_size: int,
    mask_token_id: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Fast-dLLM v2 complementary views and concatenate each as ``[x_t, x_0]``."""
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("Fast-dLLM v2 requires 2D input_ids and labels with identical shapes.")
    batch_size, sequence_length = input_ids.shape
    if sequence_length % block_size:
        raise ValueError(f"sequence length {sequence_length} must be divisible by block_size {block_size}.")

    if position_ids is None:
        position_ids = torch.arange(sequence_length, device=input_ids.device).expand(batch_size, -1)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if position_ids.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
        raise ValueError("position_ids and attention_mask must match input_ids before Fast-dLLM expansion.")

    reset_columns = torch.where(position_ids.eq(0))[1]
    if torch.any(reset_columns.remainder(block_size)):
        raise ValueError("Packed sequence boundaries must be aligned to the Fast-dLLM block size.")

    blocks = input_ids.reshape(-1, block_size)
    timesteps = torch.rand(blocks.shape[0], device=input_ids.device)
    probabilities = ((1 - eps) * timesteps + eps).unsqueeze(1)
    mask = torch.rand(blocks.shape, device=input_ids.device) < probabilities

    views = []
    view_labels = []
    for view_mask in (mask, ~mask):
        noised = input_ids.clone()
        sampled = torch.where(view_mask, mask_token_id, blocks).reshape_as(input_ids)
        eligible = labels.ne(-100)
        noised[eligible] = sampled[eligible]
        masked_labels = labels.clone()
        masked_labels[noised.ne(mask_token_id)] = -100
        views.append(torch.cat((noised, input_ids), dim=1))
        view_labels.append(masked_labels)

    expanded_positions = torch.cat((position_ids, position_ids), dim=1).repeat(2, 1)
    expanded_attention = torch.cat((attention_mask, attention_mask), dim=1).repeat(2, 1)
    return torch.cat(views), torch.cat(view_labels), expanded_positions, expanded_attention


def create_fast_dllm_mask(
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    source_length: int,
    block_size: int,
    sliding_window: int | None = None,
):
    """Build the Fast-dLLM v2 block mask, including packed-sequence boundaries."""
    physical_length = source_length * 2
    if position_ids.shape != attention_mask.shape or position_ids.shape[1] != physical_length:
        raise ValueError("Fast-dLLM mask metadata must have shape [batch, 2 * source_length].")

    source_segment_ids = position_ids[:, :source_length].eq(0).cumsum(dim=-1)
    segment_ids = torch.cat((source_segment_ids, source_segment_ids), dim=-1)

    def mask_mod(batch, head, query_index, key_index):
        del head
        query_clean = query_index >= source_length
        key_clean = key_index >= source_length
        query_logical = query_index.remainder(source_length)
        key_logical = key_index.remainder(source_length)
        query_block = query_logical // block_size
        key_block = key_logical // block_size

        block_diagonal = (query_block == key_block) & (query_clean == key_clean)
        offset_block_causal = (query_block > key_block) & ~query_clean & key_clean
        block_causal = (query_block >= key_block) & query_clean & key_clean
        visible = block_diagonal | offset_block_causal | block_causal
        same_segment = segment_ids[batch, query_index] == segment_ids[batch, key_index]
        valid = attention_mask[batch, query_index].bool() & attention_mask[batch, key_index].bool()
        if sliding_window is not None:
            same_block = query_block == key_block
            visible = visible & (same_block | ((query_logical - key_logical) <= sliding_window))
        return visible & same_segment & valid

    return create_block_mask(
        mask_mod,
        B=position_ids.shape[0],
        H=None,
        Q_LEN=physical_length,
        KV_LEN=physical_length,
        device=position_ids.device,
    )


def create_block_causal_mask(
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    block_size: int,
    sliding_window: int | None = None,
):
    """Build the Fast-dLLM v2 inference mask: causal between blocks, dense within each block."""
    if position_ids.shape != attention_mask.shape:
        raise ValueError("position_ids and attention_mask must have identical shapes.")
    sequence_length = position_ids.shape[1]
    segment_ids = position_ids.eq(0).cumsum(dim=-1)

    def mask_mod(batch, head, query_index, key_index):
        del head
        query_block = query_index // block_size
        key_block = key_index // block_size
        visible = query_block >= key_block
        if sliding_window is not None:
            visible = visible & ((query_block == key_block) | ((query_index - key_index) <= sliding_window))
        same_segment = segment_ids[batch, query_index] == segment_ids[batch, key_index]
        valid = attention_mask[batch, query_index].bool() & attention_mask[batch, key_index].bool()
        return visible & same_segment & valid

    return create_block_mask(
        mask_mod,
        B=position_ids.shape[0],
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=position_ids.device,
    )


def create_cached_block_mask(
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    past_key_values: Cache,
    layer_idx: int,
    sliding_window: int | None = None,
):
    """Attend from one noisy block to an immutable clean-prefix KV cache."""
    if position_ids.shape != attention_mask.shape:
        raise ValueError("Cached block position_ids and attention_mask must have identical shapes.")
    query_length = position_ids.shape[1]
    prefix_length = past_key_values.get_seq_length(layer_idx)
    kv_length, kv_offset = past_key_values.get_mask_sizes(query_length, layer_idx)

    def mask_mod(batch, head, query_index, key_index):
        del head
        query_position = position_ids[batch, query_index]
        key_position = key_index + kv_offset
        current_key = (key_position >= prefix_length) & (key_position < prefix_length + query_length)
        visible = torch.ones_like(current_key, dtype=torch.bool)
        if sliding_window is not None:
            visible = current_key | ((query_position - key_position) <= sliding_window)
        current_index = (key_position - prefix_length).clamp(min=0, max=query_length - 1)
        key_valid = current_key & attention_mask[batch, current_index].bool()
        key_valid = (key_position < prefix_length) | key_valid
        return visible & attention_mask[batch, query_index].bool() & key_valid

    return create_block_mask(
        mask_mod,
        B=position_ids.shape[0],
        H=None,
        Q_LEN=query_length,
        KV_LEN=kv_length,
        device=position_ids.device,
    )
