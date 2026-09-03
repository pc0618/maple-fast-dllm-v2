# Experiment handoff

- Treat training as complete only after the final DCP `.metadata` file and the
  successful checkpoint-save log marker both exist.
- Wait for training to release all GPUs before export or evaluation.
- Export with `scripts/maple/export_ternary_hf.py`, then run lm-eval on
  `hellaswag,arc_easy,piqa` with `--limit 100 --seed 1234` and compare against
  the checked-in base result.
- Run `scripts/maple/decode_fast_dllm_v2.py` on the same export and report mask
  completion separately from semantic correctness.
- Preserve raw JSON and log base scores, trained scores, per-task drops,
  aggregate retention, and BDLM checks to W&B.
- Intermediate results are diagnostic. At 1B source tokens, quality passes only
  if aggregate retention is at least 95%, no task drops more than 0.05 absolute,
  and BDLM decoding passes its mechanical and semantic checks.
- `scripts/maple/hourly_supervisor.py` is node-specific experimental tooling;
  update its paths, run IDs, and token target before reuse. A reported
  `phase=complete` is not itself proof that the quality gate passed—inspect the
  final report.
