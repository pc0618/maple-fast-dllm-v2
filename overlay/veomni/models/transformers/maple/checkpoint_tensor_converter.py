"""Convert public Maple checkpoints to VeOmni's fused/QAT parameter layout."""

import re
from typing import Dict, List, Optional, Tuple

import torch

from ..._moe_fused_weight_map import PER_EXPERT_SPLIT_TO_FUSED_PATTERN
from ...checkpoint_tensor_loading import ConvertedCheckpointTensor


_EXPERT_PATTERN = PER_EXPERT_SPLIT_TO_FUSED_PATTERN
_ATTENTION_PATTERN = re.compile(r"^(.*\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj))\.weight$")


class MapleCheckpointTensorConverter:
    def __init__(self, num_experts: int, qat_ternary: bool):
        self.num_experts = num_experts
        self.qat_ternary = qat_ternary
        self._expert_buffer: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}
        self._stacked_buffer: Dict[str, Dict[str, torch.Tensor]] = {}

    def _parameter_name(self, name: str) -> str:
        return (
            f"{name.rsplit('.', 1)[0]}.parametrizations.{name.rsplit('.', 1)[1]}.original"
            if self.qat_ternary
            else name
        )

    def can_handle(self, name: str) -> bool:
        return name == "model.word_embeddings.weight" or bool(
            _ATTENTION_PATTERN.match(name) or _EXPERT_PATTERN.match(name)
        )

    def convert(self, name: str, tensor: torch.Tensor) -> Optional[ConvertedCheckpointTensor]:
        if name == "model.word_embeddings.weight":
            return ConvertedCheckpointTensor("model.embed_tokens.weight", tensor)
        attention_match = _ATTENTION_PATTERN.match(name)
        if attention_match:
            return ConvertedCheckpointTensor(self._parameter_name(name), tensor)

        match = _EXPERT_PATTERN.match(name)
        if not match:
            return None
        prefix, expert_id_string, projection = match.groups()
        key = (prefix, projection)
        self._expert_buffer.setdefault(key, {})[int(expert_id_string)] = tensor
        if len(self._expert_buffer[key]) < self.num_experts:
            return None

        stacked = torch.stack([self._expert_buffer[key][index] for index in range(self.num_experts)])
        del self._expert_buffer[key]
        if projection == "down_proj":
            return ConvertedCheckpointTensor(self._parameter_name(f"{prefix}.experts.down_proj"), stacked)

        waiting = self._stacked_buffer.setdefault(prefix, {})
        waiting[projection] = stacked
        if "gate_proj" not in waiting or "up_proj" not in waiting:
            return None
        merged = torch.cat((waiting["gate_proj"], waiting["up_proj"]), dim=1)
        del self._stacked_buffer[prefix]
        return ConvertedCheckpointTensor(self._parameter_name(f"{prefix}.experts.gate_up_proj"), merged)

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        if self._expert_buffer or self._stacked_buffer:
            raise RuntimeError(
                "Maple checkpoint is incomplete: "
                f"experts={[(key, len(value)) for key, value in self._expert_buffer.items()]}, "
                f"projections={[(key, sorted(value)) for key, value in self._stacked_buffer.items()]}"
            )
        return []


def create_maple_checkpoint_tensor_converter(model):
    return MapleCheckpointTensorConverter(model.config.num_experts, model.config.qat_ternary)
