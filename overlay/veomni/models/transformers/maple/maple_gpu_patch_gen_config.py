# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Generate Maple modeling from Transformers 5.9's supported Qwen3-MoE source.

Regen command:
patchgen veomni.models.transformers.maple.maple_gpu_patch_gen_config \
  -o veomni/models/transformers/maple/generated --diff
"""

from typing import Optional

import torch
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import MoeCausalLMOutputWithLogProbs


config = PatchConfig(
    source_module="transformers.models.qwen3_moe.modeling_qwen3_moe",
    target_file="patched_modeling_maple_gpu.py",
    description="Maple hybrid NoPE/partial-RoPE attention, ternary QAT, and Fast-dLLM v2",
)
config.add_import(
    "veomni.models.transformers.maple.modeling_utils",
    names=["create_block_causal_mask", "create_fast_dllm_mask", "prepare_fast_dllm_batch", "ternarize_parameter"],
)
config.add_import(
    "veomni.models.transformers.masking_utils",
    names=["create_causal_mask", "create_sliding_window_causal_mask"],
)
config.drop_import_names("create_causal_mask", "create_sliding_window_causal_mask")
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "MoeCausalLMOutputWithLogProbs"],
)
config.drop_import_names("MoeCausalLMOutputWithPast")
config.add_post_import_block(
    """
    from veomni.ops.dispatch import OpSlot
    veomni_rms_norm = OpSlot("rms_norm", "standard")
    veomni_apply_rotary_pos_emb = OpSlot("rotary_pos_emb", "full")
    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_load_balancing_loss = OpSlot("load_balancing_loss", "standard")
    """
)


@config.override_method("Qwen3MoeRMSNorm.forward", description="Use VeOmni's fused RMSNorm when available")
def maple_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if veomni_rms_norm.use_non_eager_impl:
        return veomni_rms_norm(hidden_states, self.weight, self.variance_epsilon)
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


@config.replace_class("Qwen3MoeExperts", description="Fused Maple experts with ternary latent weights")
class PatchedMapleExperts(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.limit = 7.0
        self.gate_up_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = torch.nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if veomni_moe_experts_forward.use_non_eager_impl:
            return veomni_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights)

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            current = self.act_fn(gate) * up
            current = torch.nn.functional.linear(current, self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
        return final_hidden_states


@config.override_method("Qwen3MoeTopKRouter.forward", description="Match Maple's FP32 router")
def maple_router_forward(self, hidden_states: torch.Tensor):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = torch.nn.functional.linear(hidden_states.float(), self.weight.float())
    routing_weights = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    return router_logits, router_top_value, router_indices


@config.override_method(
    "Qwen3MoeRotaryEmbedding.compute_default_rope_parameters",
    description="Build frequencies only for Maple's rotary head fraction",
)
def maple_compute_default_rope_parameters(
    config=None,
    device: Optional[torch.device] = None,
    seq_len: int | None = None,
) -> tuple[torch.Tensor, float]:
    del seq_len
    base = config.rope_parameters["rope_theta"]
    dim = int(config.head_dim * config.partial_rotary_factor)
    if dim % 2:
        raise ValueError("Maple rotary dimension must be even.")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


@config.replace_function("apply_rotary_pos_emb", description="Apply RoPE to only the leading Maple head channels")
def maple_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    del position_ids
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    if veomni_apply_rotary_pos_emb.use_non_eager_impl:
        q_rot, k_rot = veomni_apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=unsqueeze_dim)
    else:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


rotate_half = None


@config.override_method(
    "Qwen3MoeAttention.__init__", description="Configure Maple hybrid attention and ternary projections"
)
def maple_attention_init(self, config, layer_idx: int):
    torch.nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = config.head_dim
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.q_proj = torch.nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.use_bias)
    self.k_proj = torch.nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias)
    self.v_proj = torch.nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias)
    self.o_proj = torch.nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.use_bias)
    self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None


@config.override_method(
    "Qwen3MoeAttention.forward", description="Use partial RoPE on sliding layers and NoPE globally"
)
def maple_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if self.sliding_window is not None:
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


@config.override_method(
    "Qwen3MoeModel.forward", description="Select Maple global/sliding and Fast-dLLM masks per layer"
)
def maple_model_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    bdlm_source_length: int | None = None,
    bdlm_decode: bool = False,
    **kwargs: Unpack[TransformersKwargs],
) -> MoeModelOutputWithPast:
    """
    cache_position (`torch.LongTensor`, *optional*):
        Absolute positions used by cache implementations.
    bdlm_source_length (`int`, *optional*):
        Unexpanded sequence length when the input contains Fast-dLLM ``[x_t, x_0]`` pairs.
    bdlm_decode (`bool`, *optional*):
        Use block-causal Fast-dLLM inference attention without KV caching.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if (bdlm_source_length is not None or bdlm_decode) and (past_key_values is not None or use_cache):
        raise ValueError("Fast-dLLM training does not support KV caching.")
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if bdlm_decode:
        global_mask = create_block_causal_mask(
            position_ids,
            attention_mask,
            block_size=self.config.bdlm_block_size,
        )
        sliding_mask = create_block_causal_mask(
            position_ids,
            attention_mask,
            block_size=self.config.bdlm_block_size,
            sliding_window=self.config.sliding_window,
        )
    elif bdlm_source_length is not None:
        global_mask = create_fast_dllm_mask(
            position_ids,
            attention_mask,
            source_length=bdlm_source_length,
            block_size=self.config.bdlm_block_size,
        )
        sliding_mask = create_fast_dllm_mask(
            position_ids,
            attention_mask,
            source_length=bdlm_source_length,
            block_size=self.config.bdlm_block_size,
            sliding_window=self.config.sliding_window,
        )
    else:
        mask_kwargs = dict(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cu_seq_lens_q=kwargs.get("cu_seq_lens_q"),
        )
        global_mask = create_causal_mask(**mask_kwargs)
        sliding_mask = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        layer_mask = sliding_mask if decoder_layer.self_attn.sliding_window is not None else global_mask
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=layer_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    hidden_states = self.norm(hidden_states)
    return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


@config.override_method(
    "Qwen3MoeForCausalLM.__init__", description="Install ternary parametrizations after initialization"
)
def maple_forcausallm_init(self, config):
    Qwen3MoePreTrainedModel.__init__(self, config)
    self.model = Qwen3MoeModel(config)
    self.vocab_size = config.vocab_size
    self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.router_aux_loss_coef = config.router_aux_loss_coef
    self.num_experts = config.num_experts
    self.num_experts_per_tok = config.num_experts_per_tok
    self.post_init()
    if config.qat_ternary:
        for layer in self.model.layers:
            for projection in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
            ):
                ternarize_parameter(projection)
            ternarize_parameter(layer.mlp.experts, "gate_up_proj")
            ternarize_parameter(layer.mlp.experts, "down_proj")


@config.override_method("Qwen3MoeForCausalLM.forward", description="Add Fast-dLLM v2 corruption and VeOmni fused loss")
def maple_forcausallm_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_router_logits: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    bdlm_decode: bool = False,
    **kwargs: Unpack[TransformersKwargs],
) -> MoeCausalLMOutputWithLogProbs:
    """
    labels (`torch.LongTensor`, *optional*):
        Target token ids; ``-100`` entries are not corrupted or scored.
    cache_position (`torch.LongTensor`, *optional*):
        Absolute positions used by cache implementations.
    bdlm_decode (`bool`, *optional*):
        Use block-causal Fast-dLLM inference attention.
    """
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )
    bdlm_source_length = None
    if bdlm_decode:
        use_cache = False
    if self.training and self.config.training_objective == "fast_dllm_v2":
        if input_ids is None or labels is None or inputs_embeds is not None:
            raise ValueError("Fast-dLLM v2 training requires input_ids and labels, not inputs_embeds.")
        bdlm_source_length = labels.shape[1]
        input_ids, labels, position_ids, attention_mask = prepare_fast_dllm_batch(
            input_ids,
            labels,
            position_ids,
            attention_mask,
            block_size=self.config.bdlm_block_size,
            mask_token_id=self.config.mask_token_id,
            eps=self.config.bdlm_sampling_eps,
        )
        use_cache = False

    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        bdlm_source_length=bdlm_source_length,
        bdlm_decode=bdlm_decode,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state
    if bdlm_source_length is not None:
        hidden_states = hidden_states[:, :bdlm_source_length]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss_output = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            if isinstance(loss_output, tuple):
                loss, _, fused_linear_aux = loss_output
                if fused_linear_aux is not None:
                    logits = None
            else:
                loss = loss_output
    else:
        logits = self.lm_head(hidden_states)

    aux_loss = None
    if output_router_logits:
        if veomni_load_balancing_loss.use_non_eager_impl:
            aux_loss = veomni_load_balancing_loss(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
        else:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
        # Qwen's top-k frequency sums to K; Maple's Switch loss L1-normalizes it.
        aux_loss = aux_loss / self.num_experts_per_tok
        if labels is not None and self.router_aux_loss_coef:
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

    return MoeCausalLMOutputWithLogProbs(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
        fused_linear_aux=fused_linear_aux,
    )


@config.override_method("Qwen3MoeForCausalLM.get_parallel_plan", description="Register Maple EP plan")
def maple_get_parallel_plan(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan(qat_ternary=self.config.qat_ternary)
