import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from veomni.models.transformers.maple.checkpoint_tensor_converter import MapleCheckpointTensorConverter
from veomni.models.transformers.maple.configuration_maple import MapleConfig
from veomni.models.transformers.maple.modeling_utils import (
    QuantizeTernary,
    ReadOnlyCache,
    create_cached_block_mask,
    create_fast_dllm_mask,
    materialize_ternary_parameters,
    prepare_fast_dllm_batch,
    twn_torch_ref,
)
from veomni.ops.kernels.load_balancing_loss.eager import load_balancing_loss_pytorch
from veomni.trainer.base import freeze_moe_router_parameters


def test_maple_ternary_and_fast_dllm_v2_contract():
    weight = torch.tensor([[1.0, 0.1, -2.0, -0.1]], requires_grad=True)
    expected = torch.tensor([[1.5, 0.0, -1.5, 0.0]])
    torch.testing.assert_close(twn_torch_ref(weight), expected)
    QuantizeTernary.apply(weight).sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))

    input_ids = torch.arange(8).view(1, 8)
    labels = input_ids.clone()
    labels[:, :2] = -100
    position_ids = torch.arange(8).view(1, 8)
    torch.manual_seed(7)
    expanded, targets, positions, valid = prepare_fast_dllm_batch(
        input_ids,
        labels,
        position_ids,
        torch.ones_like(input_ids),
        block_size=4,
        mask_token_id=99,
        eps=1e-3,
    )
    assert expanded.shape == (2, 16)
    assert targets.shape == (2, 8)
    assert torch.equal(expanded[:, 8:], input_ids.expand(2, -1))
    assert torch.equal(targets.ne(-100).sum(0), labels.ne(-100).long().squeeze(0))

    mask = create_fast_dllm_mask(positions, valid, source_length=8, block_size=4, sliding_window=4)

    def allowed(query, key):
        return bool(mask.mask_mod(torch.tensor(0), torch.tensor(0), query, key))

    assert allowed(torch.tensor(0), torch.tensor(3))  # noisy-to-noisy within a block
    assert not allowed(torch.tensor(0), torch.tensor(4))
    assert allowed(torch.tensor(4), torch.tensor(8))  # noisy-to-clean from an earlier block
    assert not allowed(torch.tensor(8), torch.tensor(0))  # no clean-to-noisy edge
    assert allowed(torch.tensor(12), torch.tensor(8))  # clean block-causal edge


def test_maple_readonly_prefix_cache_and_mask():
    source = DynamicCache()
    prefix_keys = torch.randn(1, 1, 3, 4)
    prefix_values = torch.randn_like(prefix_keys)
    source.update(prefix_keys, prefix_values, 0)
    readonly = ReadOnlyCache(source)
    block_keys = torch.randn(1, 1, 2, 4)
    keys, _ = readonly.update(block_keys, block_keys, 0)

    assert source.get_seq_length() == 3
    assert keys.shape[-2] == 5
    replacement = torch.randn_like(block_keys)
    replaced, _ = readonly.update(replacement, replacement, 0)
    torch.testing.assert_close(replaced[..., -2:, :], replacement)
    assert source.get_seq_length() == 3
    mask = create_cached_block_mask(
        torch.tensor([[3, 4]]),
        torch.ones(1, 2, dtype=torch.long),
        past_key_values=readonly,
        layer_idx=0,
        sliding_window=2,
    )

    def allowed(query, key):
        return bool(mask.mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor(query), torch.tensor(key)))

    assert allowed(0, 1)  # absolute positions 3 and 1 are at the sliding-window boundary
    assert not allowed(1, 1)
    assert allowed(0, 4)  # every token in the active block is bidirectionally visible


def test_maple_materializes_ternary_parameters_once():
    linear = torch.nn.Linear(4, 2, bias=False)
    torch.nn.utils.parametrize.register_parametrization(linear, "weight", torch.nn.Identity())
    expected = linear.weight.detach().clone()

    assert materialize_ternary_parameters(linear) == 1
    assert materialize_ternary_parameters(linear) == 0
    torch.testing.assert_close(linear.weight, expected)

def test_maple_config_and_checkpoint_conversion():
    config = MapleConfig(num_hidden_layers=8)
    assert config.router_aux_loss_coef == 0.001
    assert config.layer_types == ["sliding_attention"] * 3 + ["full_attention"] + ["sliding_attention"] * 3 + [
        "full_attention"
    ]

    converter = MapleCheckpointTensorConverter(num_experts=2, qat_ternary=True)
    embedding = converter.convert("model.word_embeddings.weight", torch.ones(3, 2))
    assert embedding.name == "model.embed_tokens.weight"
    attention = converter.convert("model.layers.0.self_attn.q_proj.weight", torch.ones(2, 2))
    assert attention.name.endswith("q_proj.parametrizations.weight.original")

    tensors = {}
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            result = converter.convert(
                f"model.layers.0.mlp.experts.{expert}.{projection}.weight",
                torch.full((2, 3), expert + 1.0),
            )
            if result is not None:
                tensors[result.name] = result.tensor
    assert tensors["model.layers.0.mlp.experts.parametrizations.gate_up_proj.original"].shape == (2, 4, 3)
    assert tensors["model.layers.0.mlp.experts.parametrizations.down_proj.original"].shape == (2, 2, 3)
    assert converter.finalize() == []


def test_maple_switch_loss_l1_normalizes_topk_frequency():
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    probs = logits.softmax(dim=-1)
    top_k = 2
    selected = probs.topk(top_k, dim=-1).indices
    expert_frequency = torch.zeros(probs.size(-1)).scatter_add_(0, selected.flatten(), torch.ones(selected.numel()))
    expected = (
        probs.size(-1) * (F.normalize(probs.sum(0), p=1, dim=0) * F.normalize(expert_frequency, p=1, dim=0)).sum()
    )

    qwen_loss = load_balancing_loss_pytorch((logits,), probs.size(-1), top_k)
    torch.testing.assert_close(qwen_loss / top_k, expected)


def test_freeze_maple_router_only():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.gate = torch.nn.Linear(2, 2, bias=False)
    model.model.layers[0].mlp.experts = torch.nn.Linear(2, 2, bias=False)
    model.model.layers[0].self_attn = torch.nn.Linear(2, 2, bias=False)

    assert freeze_moe_router_parameters(model) == ["model.layers.0.mlp.gate.weight"]
    assert not model.model.layers[0].mlp.gate.weight.requires_grad
    assert model.model.layers[0].mlp.experts.weight.requires_grad
    assert model.model.layers[0].self_attn.weight.requires_grad


def test_maple_partial_rope_uses_fused_kernel_for_rotary_slice(monkeypatch):
    from veomni.models.transformers.maple.generated import patched_modeling_maple_gpu as modeling

    q = torch.zeros(1, 2, 3, 8)
    k = torch.zeros(1, 1, 3, 8)
    cos = torch.ones(1, 3, 4)
    sin = torch.zeros_like(cos)

    monkeypatch.setattr(modeling.veomni_apply_rotary_pos_emb, "_kernel", lambda q, k, cos, sin, **_: (q + 1, k + 1))
    q_out, k_out = modeling.apply_rotary_pos_emb(q, k, cos, sin)

    assert torch.equal(q_out[..., :4], torch.ones_like(q_out[..., :4]))
    assert torch.equal(k_out[..., :4], torch.ones_like(k_out[..., :4]))
    assert torch.equal(q_out[..., 4:], q[..., 4:])
    assert torch.equal(k_out[..., 4:], k[..., 4:])
