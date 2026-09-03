#!/usr/bin/env python3
"""Enforce the agreed AR-vs-BDLM quality gate from an evaluation JSON file."""

import argparse
import json
from pathlib import Path


def evaluate_gate(metrics):
    ar, bdlm = metrics["ar"], metrics["bdlm"]
    aggregate_ratio = bdlm["aggregate"] / ar["aggregate"]
    benchmarks = sorted(set(ar["benchmarks"]) | set(bdlm["benchmarks"]))
    missing = [name for name in benchmarks if name not in ar["benchmarks"] or name not in bdlm["benchmarks"]]
    if missing:
        raise ValueError(f"Missing paired benchmark scores: {missing}")
    drops = {name: ar["benchmarks"][name] - bdlm["benchmarks"][name] for name in benchmarks}
    passed = aggregate_ratio >= 0.95 and all(drop <= 5.0 for drop in drops.values())
    return {"passed": passed, "aggregate_ratio": aggregate_ratio, "absolute_drops": drops}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args()
    result = evaluate_gate(json.loads(args.metrics.read_text()))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
