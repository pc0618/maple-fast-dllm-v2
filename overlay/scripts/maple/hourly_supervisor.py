#!/usr/bin/env python3
"""Run one-hour Maple diagnostics, then supervise the full 1B-token run."""

import argparse
import json
import math
import os
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK = Path("/mnt/data/VeOmni-maple/supervisor")
OUTPUT = ROOT / "outputs/maple-fast-dllm-v2-1b-20260903"
BASE_EVAL = Path("/mnt/data/VeOmni-maple/evals/base-diagnostic")
EXPORTS = Path("/mnt/data/VeOmni-maple/evals/exports")
MAX_STEPS = 1908
DIAGNOSTIC_EVERY = 318
WANDB_PATH = "pranshu-01-c-stanford-university/maple-bdlm/maple-bdlm-1b-mbs32-clean-20260903"
EVAL_TASKS = "hellaswag,arc_easy,piqa"
EVAL_PYTHONPATH = (
    "/mnt/data/VeOmni-maple/eval-runtime/transformers-4.57.1:"
    "/root/.cache/uv/archive-v0/Hob_cW_r2Zl7rAXK"
)
METRIC_KEYS = [
    "_step",
    "mfu",
    "tokens_per_second(M)",
    "training/total_loss",
    "training/foundation_loss",
    "training/router_aux_loss",
    "training/grad_norm",
    "moe/router_entropy/avg",
    "moe/max_vio/max",
]


def save_state(**updates):
    path = WORK / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(updates, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    path.write_text(json.dumps(state, indent=2) + "\n")
    return state


def latest_checkpoint(max_step=MAX_STEPS):
    candidates = []
    for path in (OUTPUT / "checkpoints").glob("global_step_*"):
        try:
            step = int(path.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        if step <= max_step and (path / ".metadata").exists():
            candidates.append((step, path))
    return max(candidates, default=(0, None))


def run_logged(command, log_path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n[{time.strftime('%FT%T%z')}] $ {' '.join(map(str, command))}\n")
        log.flush()
        return subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def result_file(directory):
    files = list(directory.glob("**/results_*.json"))
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def eval_scores(directory):
    path = result_file(directory)
    if not path:
        raise FileNotFoundError(f"No lm-eval result under {directory}")
    results = json.loads(path.read_text())["results"]
    scores = {}
    for task, metrics in results.items():
        key = "acc_norm,none" if "acc_norm,none" in metrics else "acc,none"
        scores[task] = float(metrics[key])
    return scores


def run_eval(model, directory):
    if result_file(directory):
        return eval_scores(directory)
    env = os.environ.copy()
    env.update(
        HF_HOME="/mnt/data/huggingface",
        PYTHONPATH=f"{EVAL_PYTHONPATH}:{env.get('PYTHONPATH', '')}",
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
    )
    base = [
        str(ROOT / ".venv/bin/lm-eval"),
        "run",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model},trust_remote_code=True,dtype=bfloat16,parallelize=True",
        "--tasks",
        EVAL_TASKS,
        "--limit",
        "100",
        "--seed",
        "1234",
        "--output_path",
        str(directory),
    ]
    for batch_size in ("8", "1"):
        if run_logged(base + ["--batch_size", batch_size], directory / "run.log", env) == 0:
            return eval_scores(directory)
    raise RuntimeError(f"lm-eval failed; see {directory / 'run.log'}")


def training_command(load_path=None):
    command = [
        str(ROOT / ".venv/bin/torchrun"),
        "--standalone",
        "--nproc_per_node=8",
        "tasks/train_text.py",
        "configs/text/maple_fast_dllm_v2.yaml",
        f"--train.max_steps={MAX_STEPS}",
        "--data.train_sample=61056",
        "--train.micro_batch_size=32",
        "--train.global_batch_size=256",
        "--train.accelerator.dp_shard_size=8",
        "--train.accelerator.ep_size=1",
        "--train.accelerator.fsdp_config.reshard_after_forward=true",
        "--train.accelerator.fsdp_config.reshard_after_backward=true",
        "--train.moe_load_balance_monitor_interval=20",
        f"--train.checkpoint.output_dir={OUTPUT}",
        f"--train.checkpoint.save_steps={DIAGNOSTIC_EVERY}",
        "--train.checkpoint.save_epochs=0",
        "--train.checkpoint.keep_last_n=2",
        "--train.checkpoint.save_hf_weights=false",
        "--train.profile.enable=false",
        "--train.wandb.enable=true",
        "--train.wandb.project=maple-bdlm",
        "--train.wandb.id=maple-bdlm-1b-mbs32-clean-20260903",
        "--train.wandb.name=maple-fast-dllm-v2-maple-20b-1b-mbs32-20260903",
    ]
    if load_path:
        command.append(f"--train.checkpoint.load_path={load_path}")
    return command


def checkpoint_logged(log_path, step):
    if not log_path.exists():
        return False
    with log_path.open(errors="replace") as log:
        log.seek(max(0, log_path.stat().st_size - 32_768))
        return f"global_step_{step} successfully!" in log.read()


def train(until_step=None):
    step, checkpoint = latest_checkpoint()
    env = os.environ.copy()
    env.update(
        WANDB_ENTITY="pranshu-01-c-stanford-university",
        WANDB_RESUME="allow",
        HF_HOME="/mnt/data/huggingface",
        PYTHONUNBUFFERED="1",
    )
    log_path = WORK / f"train-from-{step}.log"
    log = log_path.open("a")
    process = subprocess.Popen(
        training_command(checkpoint),
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    save_state(phase="training", pid=process.pid, from_step=step, target_step=until_step or MAX_STEPS)
    while process.poll() is None:
        current, _ = latest_checkpoint()
        if until_step and current >= until_step and checkpoint_logged(log_path, until_step):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.close()
            return current
        time.sleep(15)
    log.close()
    current, _ = latest_checkpoint()
    if process.returncode or current < (until_step or MAX_STEPS):
        raise RuntimeError(f"training exited {process.returncode} at checkpoint {current}; see {log_path}")
    return current


def train_with_retries(until_step=None):
    for failures in range(3):
        try:
            return train(until_step)
        except RuntimeError as error:
            save_state(phase="retrying", failures=failures + 1, error=str(error))
    raise RuntimeError("training failed three times; see the latest train log")


def wandb_history(min_step, max_step):
    import wandb

    for _ in range(10):
        try:
            run = wandb.Api(timeout=30).run(WANDB_PATH)
            rows = [
                row
                for row in run.scan_history(keys=METRIC_KEYS)
                if min_step <= int(row.get("_step", -1)) <= max_step
            ]
            # The diagnostic launcher stops immediately after the checkpoint
            # barrier, so W&B may not flush that step's final history row.
            if rows and max(int(row["_step"]) for row in rows) >= max_step - 1:
                return rows
        except Exception:
            pass
        time.sleep(30)
    raise RuntimeError(f"W&B history did not reach step {max_step}")


def median(rows, key, start=None, end=None):
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None
        and (start is None or row["_step"] >= start)
        and (end is None or row["_step"] <= end)
    ]
    return statistics.median(values) if values else None


def assess_history(rows):
    scalar_keys = METRIC_KEYS[1:]
    bad = [key for row in rows for key in scalar_keys if row.get(key) is not None and not math.isfinite(float(row[key]))]
    losses = [float(row["training/foundation_loss"]) for row in rows if row.get("training/foundation_loss") is not None]
    early = statistics.median(losses[:20]) if losses else None
    late = statistics.median(losses[-20:]) if losses else None
    entropy = median(rows, "moe/router_entropy/avg")
    max_vio = median(rows, "moe/max_vio/max")
    report = {
        "finite": not bad,
        "loss_early_median": early,
        "loss_late_median": late,
        "mfu_steps_10_20_median": median(rows, "mfu", 10, 20),
        "mfu_all_median": median(rows, "mfu"),
        "tokens_per_second_median_millions": median(rows, "tokens_per_second(M)"),
        "router_entropy_median": entropy,
        "router_max_vio_median": max_vio,
    }
    report["passed"] = bool(
        report["finite"]
        and early is not None
        and late is not None
        and late <= early * 1.10
        and report["mfu_steps_10_20_median"]
        and entropy is not None
        and entropy >= 0.70 * math.log(256)
        and max_vio is not None
        and max_vio < 64
    )
    return report


def compare_eval(base, trained):
    tasks = sorted(set(base) & set(trained))
    if not tasks:
        raise ValueError("No paired lm-eval tasks")
    base_mean = statistics.mean(base[name] for name in tasks)
    trained_mean = statistics.mean(trained[name] for name in tasks)
    drops = {name: base[name] - trained[name] for name in tasks}
    report = {
        "base": base,
        "trained": trained,
        "aggregate_ratio": trained_mean / base_mean if base_mean else 1.0,
        "absolute_drops": drops,
    }
    report["passed"] = report["aggregate_ratio"] >= 0.95 and max(drops.values()) <= 0.05
    return report


def export_and_eval(step, checkpoint):
    destination = EXPORTS / f"global_step_{step}"
    if not (destination / "model.safetensors.index.json").exists():
        code = run_logged(
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/maple/export_ternary_hf.py",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(destination),
                "--model-assets",
                "artifacts/maple-preview",
                "--tokenizer-assets",
                "data/maple-nemotron/tokenizer",
            ],
            WORK / f"export-{step}.log",
        )
        if code:
            raise RuntimeError(f"export failed at step {step}")
    return run_eval(destination, Path("/mnt/data/VeOmni-maple/evals") / f"global_step_{step}")


def summarize_bdlm(samples):
    if len(samples) != 2:
        return {
            "mechanically_passed": False,
            "passed": False,
            "completion_success_rate": 0.0,
            "nonempty_text_rate": 0.0,
            "semantic_checks": {},
            "semantic_pass_rate": 0.0,
            "mean_seconds": None,
            "samples": [],
        }
    passed_samples = [sample for sample in samples if sample["generated_tokens"] and not sample["mask_tokens_remaining"]]
    semantic_checks = {
        "math_7x8": bool(re.search(r"\b56\b", samples[0]["text"])),
        "sky_scattering": "blue" in samples[1]["text"].lower() and "scatter" in samples[1]["text"].lower(),
    }
    mechanically_passed = len(passed_samples) == len(samples)
    return {
        "mechanically_passed": mechanically_passed,
        "passed": mechanically_passed and all(semantic_checks.values()),
        "completion_success_rate": len(passed_samples) / len(samples),
        "nonempty_text_rate": sum(bool(sample["text"].strip()) for sample in samples) / len(samples),
        "semantic_checks": semantic_checks,
        "semantic_pass_rate": sum(semantic_checks.values()) / len(semantic_checks),
        "mean_seconds": statistics.mean(sample["seconds"] for sample in samples),
        "samples": samples,
    }


def run_bdlm_validation(step):
    directory = Path("/mnt/data/VeOmni-maple/evals") / f"global_step_{step}"
    output = directory / "bdlm-validation.json"
    if not output.exists():
        code = run_logged(
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/maple/decode_fast_dllm_v2.py",
                "--model",
                str(EXPORTS / f"global_step_{step}"),
                "--prompt",
                "What is 7 multiplied by 8? Answer briefly.",
                "--prompt",
                "Write one sentence explaining why the sky appears blue.",
                "--max-new-tokens",
                "32",
                "--output-json",
                str(output),
            ],
            WORK / f"bdlm-validation-{step}.log",
            {**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        if code:
            raise RuntimeError(f"BDLM decoding validation failed at step {step}")
    return summarize_bdlm(json.loads(output.read_text()))


def log_eval(step, report, training_report, bdlm_report):
    import wandb

    run = wandb.init(
        entity="pranshu-01-c-stanford-university",
        project="maple-bdlm",
        id="maple-bdlm-1b-evals-20260903",
        name="maple-bdlm-1b-hourly-evals",
        resume="allow",
    )
    metrics = {f"eval/{task}": score for task, score in report["trained"].items()}
    metrics.update({f"eval/base/{task}": score for task, score in report["base"].items()})
    metrics.update({f"eval/drop/{task}": drop for task, drop in report["absolute_drops"].items()})
    metrics.update(
        {
            "eval/aggregate_ratio": report["aggregate_ratio"],
            "eval/passed": int(report["passed"]),
            "diagnostic/training_passed": int(training_report["passed"]),
            "diagnostic/health_passed": int(training_report["passed"] and bdlm_report["mechanically_passed"]),
            "diagnostic/mfu_steps_10_20_median": training_report["mfu_steps_10_20_median"],
            "diagnostic/router_entropy_median": training_report["router_entropy_median"],
            "bdlm/passed": int(bdlm_report["passed"]),
            "bdlm/mechanically_passed": int(bdlm_report["mechanically_passed"]),
            "bdlm/completion_success_rate": bdlm_report["completion_success_rate"],
            "bdlm/nonempty_text_rate": bdlm_report["nonempty_text_rate"],
            "bdlm/semantic_pass_rate": bdlm_report["semantic_pass_rate"],
            "bdlm/mean_seconds": bdlm_report["mean_seconds"],
        }
    )
    run.log(metrics, step=step)
    run.finish()


def self_test():
    rows = [
        {
            "_step": step,
            "mfu": 0.3,
            "tokens_per_second(M)": 0.1,
            "training/total_loss": 10 - step / 100,
            "training/foundation_loss": 10 - step / 100,
            "training/router_aux_loss": 8.0,
            "training/grad_norm": 100.0,
            "moe/router_entropy/avg": 4.8,
            "moe/max_vio/max": 12.0,
        }
        for step in range(1, 22)
    ]
    assert assess_history(rows)["passed"]
    assert compare_eval({"a": 0.5}, {"a": 0.49})["passed"]
    assert not compare_eval({"a": 0.5}, {"a": 0.4})["passed"]
    assert summarize_bdlm(
        [
            {"generated_tokens": 2, "mask_tokens_remaining": 0, "text": "56", "seconds": 1},
            {"generated_tokens": 2, "mask_tokens_remaining": 0, "text": "blue light scatters", "seconds": 1},
        ]
    )["passed"]
    degraded = summarize_bdlm(
        [
            {"generated_tokens": 2, "mask_tokens_remaining": 0, "text": "wrong", "seconds": 1},
            {"generated_tokens": 2, "mask_tokens_remaining": 0, "text": "rough", "seconds": 1},
        ]
    )
    assert degraded["mechanically_passed"] and not degraded["passed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ok")
        return

    os.chdir(ROOT)
    WORK.mkdir(parents=True, exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)
    base = run_eval("artifacts/maple-preview", BASE_EVAL)
    save_state(phase="baseline_complete", base_eval=base)

    satisfied = False
    for target in range(DIAGNOSTIC_EVERY, MAX_STEPS, DIAGNOSTIC_EVERY):
        report_path = WORK / f"diagnostic-{target}.json"
        if report_path.exists():
            previous = json.loads(report_path.read_text())
            if previous.get("health_passed"):
                satisfied = True
                break
        step, checkpoint = latest_checkpoint(target)
        if step < target:
            try:
                step = train_with_retries(target)
            except RuntimeError as error:
                save_state(phase="blocked", reason=str(error), failures=3)
                return
            _, checkpoint = latest_checkpoint(target)
        trained = export_and_eval(step, checkpoint)
        bdlm_report = run_bdlm_validation(step)
        training_report = assess_history(wandb_history(1, step))
        eval_report = compare_eval(base, trained)
        report = {"step": step, "training": training_report, "evaluation": eval_report, "bdlm": bdlm_report}
        report["health_passed"] = training_report["passed"] and bdlm_report["mechanically_passed"]
        report["quality_passed"] = eval_report["passed"] and bdlm_report["passed"]
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        log_eval(step, eval_report, training_report, bdlm_report)
        save_state(phase="diagnosed", last_diagnostic=report)
        if report["health_passed"]:
            satisfied = True
            break

    if not satisfied:
        save_state(phase="blocked", reason="No diagnostic interval passed")
        return

    while latest_checkpoint()[0] < MAX_STEPS:
        try:
            train_with_retries()
        except RuntimeError as error:
            save_state(phase="blocked", reason=str(error), failures=3)
            return
    step, checkpoint = latest_checkpoint()
    final_scores = export_and_eval(step, checkpoint)
    final_bdlm = run_bdlm_validation(step)
    final_eval = compare_eval(base, final_scores)
    final_training = assess_history(wandb_history(1, step))
    log_eval(step, final_eval, final_training, final_bdlm)
    final_report = {"evaluation": final_eval, "bdlm": final_bdlm}
    (WORK / "final-eval.json").write_text(json.dumps(final_report, indent=2) + "\n")
    save_state(phase="complete", final_step=step, final_eval=final_eval, final_bdlm=final_bdlm)


if __name__ == "__main__":
    main()
