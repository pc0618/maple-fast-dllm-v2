# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""VeOmni configuration for DeepGrove Maple."""

from transformers.configuration_utils import PretrainedConfig


class MapleConfig(PretrainedConfig):
    model_type = "maple"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        use_bias=False,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        max_position_embeddings=131072,
        rope_theta=10000.0,
        use_cache=True,
        rope_scaling=None,
        partial_rotary_factor=0.5,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        head_dim=128,
        output_router_logits=False,
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        sliding_window=512,
        layer_types=None,
        nope_on_global_attention=True,
        qat_ternary=True,
        training_objective="fast_dllm_v2",
        bdlm_block_size=32,
        bdlm_sampling_eps=1e-3,
        mask_token_id=151669,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_bias = use_bias
        self.attention_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        self.use_cache = use_cache
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.partial_rotary_factor = partial_rotary_factor
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.output_router_logits = output_router_logits
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
        self.decoder_sparse_step = 1
        self.mlp_only_layers = []
        self.sliding_window = sliding_window
        self.use_sliding_window = sliding_window is not None
        self.layer_types = layer_types or [
            "full_attention" if (index + 1) % 4 == 0 else "sliding_attention" for index in range(num_hidden_layers)
        ]
        self.nope_on_global_attention = nope_on_global_attention
        self.qat_ternary = qat_ternary
        self.training_objective = training_objective
        self.bdlm_block_size = bdlm_block_size
        self.bdlm_sampling_eps = bdlm_sampling_eps
        self.mask_token_id = mask_token_id
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per hidden layer.")
        if self.training_objective not in {"causal", "fast_dllm_v2"}:
            raise ValueError("training_objective must be 'causal' or 'fast_dllm_v2'.")
        if self.bdlm_block_size <= 0 or self.bdlm_sampling_eps <= 0 or self.bdlm_sampling_eps >= 1:
            raise ValueError("Invalid Fast-dLLM v2 block size or sampling epsilon.")
