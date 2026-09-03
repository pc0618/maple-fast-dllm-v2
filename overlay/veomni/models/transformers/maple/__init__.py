# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY
from .configuration_maple import MapleConfig


MODEL_CONFIG_REGISTRY.register("maple", MapleConfig)


@MODELING_REGISTRY.register("maple")
def register_maple_modeling(architecture: str):
    from .checkpoint_tensor_converter import create_maple_checkpoint_tensor_converter
    from .generated.patched_modeling_maple_gpu import Qwen3MoeForCausalLM, Qwen3MoeModel

    for model_class in (Qwen3MoeForCausalLM, Qwen3MoeModel):
        model_class._create_checkpoint_tensor_converter = staticmethod(create_maple_checkpoint_tensor_converter)
    return Qwen3MoeForCausalLM if "ForCausalLM" in architecture else Qwen3MoeModel
