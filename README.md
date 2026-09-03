# Maple Fast-dLLM v2 experiments

Experimental code and preliminary results for converting
[`deepgrove/maple-preview`](https://huggingface.co/deepgrove/maple-preview), a
20B-class ternary MoE language model, from autoregressive training to a block
diffusion language model (BDLM) using a Fast-dLLM v2-style objective in
[VeOmni](https://github.com/ByteDance-Seed/VeOmni).

This is research code, not a released checkpoint. The included files are an
overlay for VeOmni commit `f9b19307538b77a5a013755095748400522311b1`.

## Current status

The newest completed run trained for 250,085,376 source tokens on 8 NVIDIA H200
GPUs. It used sequence length 2,048, micro-batch 32 per GPU, global batch 256,
FSDP2 full sharding, expert parallelism off, full activation recomputation,
BF16 compute, ternary weights on read, fused Quack MoE kernels, Triton load
balancing, and chunked cross entropy.

The router was trainable. Its normalized Switch auxiliary loss was weighted by
`0.001`; a perfectly balanced value is approximately `1.0`.

| Final training metric | Value |
|---|---:|
| Steps | 477 |
| Source tokens | 250.085M |
| Foundation loss | 3.249 |
| Switch router loss | 1.010 |
| Average router entropy | 4.816 |
| Source throughput | 65.3K tokens/s |
| Source-normalized Qwen3-MoE MFU | 6.55% |
| Estimated BDLM-compute-adjusted MFU | 21.7% |

Training curves: [Weights & Biases](https://wandb.ai/pranshu-01-c-stanford-university/maple-bdlm/runs/maple-bdlm-switch-aux-250m-mbs32-20260903).

The MFU figures are not interchangeable. VeOmni's Qwen3-MoE counter sees the
original `[B,L]` source batch, while the two-stream objective sends `[2B,2L]`
through the transformer. The more useful estimate accounts for approximately
4x transformer-body work, 2x LM-head work, and Maple's mixture of sliding and
global attention; blindly multiplying the logged MFU by four overstates it.

## Objective

For each source sequence, training builds a noisy stream `x_t` and clean stream
`x_0`, concatenated along sequence length. A complementary corruption mask is
also added along the batch dimension, producing a physical transformer input
of `[2B,2L]`. Across the complementary pair, every eligible target token is
corrupted once. The attention mask preserves causal context encoding while
allowing bidirectional denoising inside each diffusion block.

Maple-specific support retains its hybrid attention layout: global NoPE
attention and partially rotary-embedded sliding-window attention. Attention and
expert projections use ternary weight quantization on read while latent weights,
activations, and accumulation remain BF16.

## Preliminary evaluation

The checked-in JSON includes small, fixed 100-example-per-task evaluations.
The normalized-Switch checkpoint at step 477 produced:

| Task | Type | Base AR | BDLM, 250M | Change |
|---|---|---:|---:|---:|
| HellaSwag | 4-way multiple choice | 0.46 | 0.47 | +0.01 |
| ARC-Easy | Multiple choice | 0.63 | 0.56 | -0.07 |
| PIQA | Binary choice | 0.76 | 0.76 | 0.00 |
| Mean | — | 0.617 | 0.597 | -0.020 |

Aggregate retention was `96.76%`. Both BDLM smoke prompts completed without
residual mask tokens and passed their simple semantic checks. The strict
lm-eval gate remains false because ARC-Easy dropped by 0.07, beyond its 0.05
per-task limit. Evaluation metrics: [Weights & Biases](https://wandb.ai/pranshu-01-c-stanford-university/maple-bdlm/runs/maple-bdlm-switch-aux-250m-evals-20260903).

An earlier checkpoint at step 1,272 (approximately 667M source tokens) predates
the normalized Switch-loss experiment:

| Task | Type | Base AR | BDLM, ~667M | Change |
|---|---|---:|---:|---:|
| HellaSwag | 4-way multiple choice | 0.46 | 0.47 | +0.01 |
| ARC-Easy | Multiple choice | 0.63 | 0.50 | -0.13 |
| PIQA | Binary choice | 0.76 | 0.75 | -0.01 |
| Mean | — | 0.617 | 0.573 | -0.043 |

Its aggregate retention ratio was `92.97%`. BDLM decoding mechanically
completed both smoke prompts without residual mask tokens; one of two simple
semantic checks passed. These samples are diagnostic only. The target is parity
after 1B source tokens, and 100 examples per benchmark are too few for a final
quality claim.

## Repository layout

- `patches/veomni-maple.patch`: modifications to existing VeOmni files.
- `overlay/`: new files copied over the pinned VeOmni revision.
- `overlay/configs/text/`: AR controls and BDLM training configurations.
- `overlay/veomni/models/transformers/maple/`: Maple registration, attention,
  ternary-QAT, MoE, and patchgen implementation.
- `overlay/scripts/maple/`: dataset preparation, profiling, supervision,
  checkpoint export, lm-eval orchestration, quality gate, and BDLM decoding.
- `results/`: raw preliminary evaluation JSON; no model weights or datasets.

## Reproduce

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni
git checkout f9b19307538b77a5a013755095748400522311b1
git apply ../maple-fast-dllm-v2/patches/veomni-maple.patch
rsync -a ../maple-fast-dllm-v2/overlay/ ./
uv sync --extra gpu --dev
source .venv/bin/activate
python scripts/maple/hourly_supervisor.py --self-test
```

See `overlay/scripts/maple/README.md` for data preparation, training, export,
and decoding commands. The supervisor currently contains node-specific paths
and W&B identifiers; change those constants before running elsewhere.

## Scope and caveats

- Checkpoints, datasets, caches, W&B credentials, and machine traces are not
  included.
- The checked-in lm-eval uses `--limit 100`; run full benchmark splits before
  drawing model-quality conclusions.
- The current decoder is correctness-first and does not yet implement the
  inference optimizations needed to demonstrate Fast-dLLM v2 latency claims.
- This work is unaffiliated with the upstream VeOmni, Fast-dLLM, and Maple
  authors.

## License

Apache-2.0, matching the copied VeOmni source files. See `LICENSE`.
