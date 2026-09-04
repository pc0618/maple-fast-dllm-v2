# Maple Fast-dLLM v2 experiment

This fork trains two matched ternary-QAT runs from the same Maple checkpoint and exact Nemotron mixture:

- BDLM: Fast-dLLM v2, block size 32, complementary masks, `[x_t, x_0]`, `eps=1e-3`.
- Control: ordinary autoregressive QAT.

Both full configs run 2,500 updates at global batch 256 and source length 2,048: 1,310,720,000 source-token slots each. Maple keeps global attention NoPE and applies RoPE to the leading 50% of each Q/K head only on sliding-window layers. Attention and expert projections use TWN on read with latent BF16 weights; routers, embeddings, LM head, and norms stay dense.

## H100 setup

Use one 8xH100 80GB node with CUDA 12.8, at least 256GB host RAM, and roughly 1TB free local/shared storage for model, data, optimizer checkpoints, and two retained checkpoints per run.

```bash
uv sync --extra gpu --group transformers-stable
source .venv/bin/activate
huggingface-cli download deepgrove/maple-preview \
  --revision ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07 \
  --local-dir artifacts/maple-preview
```

The Fast-dLLM batch is physically four times the source token count (`2B x 2L`). Phase 1 therefore fixes Ulysses SP at 1, uses EP8 plus FSDP2, activation checkpointing, BF16, FlexAttention, and the fused Triton MoE kernel. Do not enable FlashAttention for the BDLM run: it cannot express the three-part training mask.

## Dataset

Build the 5,120-sequence pilot (10,485,760 source-token slots):

```bash
python scripts/maple/prepare_nemotron.py \
  --output-dir data/maple-nemotron \
  --num-sequences 5120 \
  --shard-size 320
```

The script pins NVIDIA's English-only `Puzzle-KD-Nemotron-Post-Training-Dataset-v2` at commit `7d7a14...`, rejects empty prompts, incomplete conversations, and overlength samples, uses Maple's native chat template with assistant-only labels, aligns packed boundaries to 32 tokens, and writes an exact repeating 20/30/25/25 chat/code/math/stem token-slot mixture. It also adds `<|mask|>` at unused padded-vocabulary id 151669 and saves the tokenizer. Pass `--dataset-dir` to reuse downloaded Arrow shards under `train/` or category-prefixed Parquet shards under `data/`.

For the full runs, rebuild with `--num-sequences 640000`.

## Pilot, then full training

Run both 20-step pilots with five warmup steps:

```bash
bash train.sh tasks/train_text.py configs/text/maple_fast_dllm_v2.yaml \
  --train.max_steps 20 \
  --train.optimizer.lr_warmup_ratio 0.25 \
  --train.checkpoint.output_dir outputs/maple-fast-dllm-v2-pilot \
  --train.checkpoint.save_steps 20

bash train.sh tasks/train_text.py configs/text/maple_ar_control.yaml \
  --train.max_steps 20 \
  --train.optimizer.lr_warmup_ratio 0.25 \
  --train.checkpoint.output_dir outputs/maple-ar-control-pilot \
  --train.checkpoint.save_steps 20
```

Before committing the full allocation, require finite loss/grad norm, no OOM, all 256 experts receiving traffic, and no persistent step-time or loss regression after compilation. Then prepare 640,000 sequences and run:

```bash
bash train.sh tasks/train_text.py configs/text/maple_fast_dllm_v2.yaml
bash train.sh tasks/train_text.py configs/text/maple_ar_control.yaml
```

Each run checkpoints every 500 updates and automatically keeps its newest two DCP checkpoints. Resume by setting `--train.checkpoint.load_path` to a `global_step_*` directory.

## Export and quality gate

Export latent DCP weights into ordinary Maple HF shards. The exporter materializes ternary values and splits fused expert tensors; it deliberately does not produce packed 2-bit weights.

```bash
python scripts/maple/export_ternary_hf.py \
  --checkpoint outputs/maple-fast-dllm-v2/checkpoints/global_step_2500 \
  --output-dir exports/maple-fast-dllm-v2 \
  --model-assets artifacts/maple-preview \
  --tokenizer-assets data/maple-nemotron/tokenizer
```

Evaluate the AR control with ordinary causal generation and the BDLM with block size 32 at a maximum 2,048-token context. Store higher-is-better scores as:

```json
{
  "ar": {"aggregate": 0.0, "benchmarks": {"benchmark": 0.0}},
  "bdlm": {"aggregate": 0.0, "benchmarks": {"benchmark": 0.0}}
}
```

Then run `python scripts/maple/gate_bdlm.py metrics.json`. Decoder work passes only if BDLM reaches at least 95% of the AR aggregate and no benchmark drops more than 5 absolute points.

For zero-shot MMLU, score the single next choice token instead of generating a
reasoning trace:

```bash
python scripts/maple/eval_mmlu_bdlm.py \
  --model artifacts/maple-preview --mode ar \
  --output-json results/base-ar-mmlu.json

python scripts/maple/eval_mmlu_bdlm.py \
  --model exports/maple-fast-dllm-v2 --mode bdlm --block-size 32 \
  --output-json results/bdlm-mmlu.json
```

Both paths apply Maple's chat template and restrict the decision to the four
single-token choices. The BDLM path performs one masked, block-causal forward
pass and reads the shifted logits for that mask.

After the gate passes, the correctness-first, no-KV-cache decoder is:

```bash
python scripts/maple/decode_fast_dllm_v2.py \
  --model exports/maple-fast-dllm-v2 \
  --prompt "Explain why the sky is blue."
```

It uses 32-token blocks, 8-token subblocks, a 0.9 confidence threshold, and guarantees at least one reveal per iteration. Treat results only as 2K-context BDLM evidence; run long-context autoregressive checks separately, and optimize block/KV caching only after the gate passes.
