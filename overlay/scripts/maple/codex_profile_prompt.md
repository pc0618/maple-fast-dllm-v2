Work autonomously on the Maple pure-AR throughput control in /root/VeOmni-maple.

Read AGENTS.md and the required knowledge files first, then use the veomni-profile skill. Do not only propose a plan: run the measurements and implement measured improvements.

Hard invariants:

- Use configs/text/maple_ar_control.yaml only. This run must remain pure causal AR; do not invoke Fast-dLLM/BDLM corruption or doubled [x_t, x_0] streams.
- Keep sequence length 2048, microbatch 16, global batch 128, and gradient accumulation 1 on all 8 H200 GPUs.
- Keep fused_quack MoE, EP=1, FSDP2 full sharding with reshard-after-forward/backward, and FP32 gradient reduction.
- Keep full decoder-layer activation recomputation: gradient_checkpointing.enable=true and early_stop=false.
- Keep qat_ternary=true. QuantizeTernary.forward must remain torch.compile(fullgraph=True) with its straight-through backward.
- Use Liger for compatible elementwise/fused operations. Maple partial RoPE may send only its rotary slice to the Liger full-RoPE kernel; global NoPE layers must remain NoPE. Quack owns the expert clamp/SwiGLU path. Do not force Liger onto the FP32 router softmax or onto an incompatible operation.
- Compute/log MFU with the existing Qwen3-MoE FLOP estimator. Log throughput, peak memory, loss, router entropy by block, and average router entropy to W&B project maple-bdlm under entity pranshu-01-c-stanford-university.
- Do not switch back to fused Triton, disable recomputation, change batch geometry, weaken numerical checks, commit, or push.
- Preserve all unrelated dirty-worktree changes. Never edit generated model files directly; edit the patchgen source and regenerate with the repository workflow.

Known state:

- Quack 0.5.0 and its pinned CUDA 13/CUTLASS dependencies are installed and import successfully on SM90.
- The existing merged gate/up Quack forward+backward parity test passed.
- Quack without recomputation reached backward at microbatch 16 but OOMed while allocating the 1 GiB gate/up weight-gradient output. Full recomputation has now been enabled to recover memory.
- Previous fused-Triton/no-recompute microbatch-16 baseline was about 12.53% MFU and 126.5K tokens/s; do not compare a recompute run to that as if the executed FLOPs were identical.
- The previous trace showed large MoE GEMM and FP32 NCCL reduce-scatter time. Measure again under the required Quack/full-recompute setup.

Measurement loop:

1. Verify the focused Maple and Quack tests, confirm resolved config and actual kernel dispatch, then run a short smoke test. Read the W&B key privately from /root/.bashrc only in the training command; never print it.
2. Run through at least optimizer step 20. Treat completed optimizer steps 10-20 inclusive as the canonical steady-state window for median step time, tokens/s, and MFU; exclude steps 0-9 from those comparison metrics. Capture a rank-0 Torch profiler trace over a narrow subset of that window with at least two steps so profiler overhead does not contaminate the canonical unprofiled measurement. Put traces and notes under /mnt/maple-traces/codex-optimization/ and use unique W&B run names prefixed maple-ar-mbs16-quack-fullrecompute-.
3. Rank CUDA kernels/collectives by self time and memory. Identify the longest actionable operation from evidence, make one minimal change, run the smallest numerical test, and rerun the exact same benchmark.
4. Keep a change only if repeated steady-state measurements improve step time/MFU without instability or excess memory. Revert regressions. Repeat for up to three measured candidate changes, or stop earlier when the remaining leaders are communication/hardware limits or no safe local improvement remains.
5. Write /mnt/maple-traces/codex-optimization/report.md with baseline and candidate run URLs, median steady-state step time, tokens/s, Qwen3-estimated MFU, peak memory, top operations, accepted/rejected changes, tests, and the next bottleneck.

Roofline and communication requirements:

- Establish empirical ceilings on this node where practical: BF16 tensor-core GEMM throughput, HBM bandwidth, and representative 8-GPU NCCL all-gather/reduce-scatter bandwidth. Label vendor-spec numbers as theoretical; do not mix them with measured ceilings.
- For each top actionable compute kernel, estimate FLOPs, bytes moved, arithmetic intensity, roofline-attainable throughput `min(compute_ceiling, HBM_bandwidth * arithmetic_intensity)`, achieved throughput, and percent of that roofline. Classify it as compute-, HBM-bandwidth-, launch/latency-, or communication-bound.
- For collectives, report payload bytes, effective bandwidth, total collective time, exposed critical-path time, and overlap with compute. Do not sum concurrent streams or ranks as wall time. Account explicitly for FSDP full reshard and FP32 gradient reduction.
- Separate the Qwen3 algorithmic/reference MFU used for experiment comparison from executed-work efficiency, which includes full-recompute and ternary quantizer overhead. An accounting change alone is not an optimization.
- Before accepting a change, state its compute/communication tradeoff, effect on overlap, and memory cost. Target the measured limiting regime; do not trade enough memory to risk the required microbatch or violate the hard invariants.
- Include a compact roofline/communication table in report.md and retain the raw traces and commands needed to reproduce every accepted result.

Use a short max_steps override and disable checkpoint saves for profiling. Ensure no stale training process is using the GPUs before each trial.
