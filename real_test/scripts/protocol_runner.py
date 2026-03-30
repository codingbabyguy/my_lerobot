#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ask(text: str, choices: list[str]) -> str:
    choice_text = "/".join(choices)
    while True:
        v = input(f"{text} [{choice_text}]: ").strip().lower()
        if v in choices:
            return v
        print(f"Invalid input: {v}")


def _append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=True) + "\n")


def _run_stage(stage_name: str, rounds: int, log_path: Path) -> list[dict]:
    print(f"\n=== {stage_name} ({rounds} rounds) ===")
    records: list[dict] = []

    for i in range(1, rounds + 1):
        print(f"\nRound {i}/{rounds}")
        result = _ask("Result", ["success", "fail", "estop"])
        note = input("Note (optional): ").strip()
        rec = {
            "timestamp": time.time(),
            "stage": stage_name,
            "round": i,
            "result": result,
            "note": note,
        }
        _append_jsonl(log_path, rec)
        records.append(rec)

    return records


def _summary(records: list[dict]) -> dict:
    total = len(records)
    success = sum(r["result"] == "success" for r in records)
    fail = sum(r["result"] == "fail" for r in records)
    estop = sum(r["result"] == "estop" for r in records)
    return {
        "total": total,
        "success": success,
        "fail": fail,
        "estop": estop,
        "success_rate": (success / total) if total > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Real test protocol runner")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json",
    )
    args = parser.parse_args()

    cfg = _load_json(args.config)
    protocol_cfg = cfg["real_test_protocol"]
    log_path = Path(protocol_cfg["log_jsonl"])

    print("Before starting, verify safety conditions.")
    safety_ok = _ask("Emergency stop reachable and enabled", ["y", "n"])
    if safety_ok != "y":
        print("Abort due to safety check failure.")
        return

    empty_rounds = int(protocol_cfg["empty_load_rounds"])
    simple_rounds = int(protocol_cfg["simplified_task_rounds"])

    empty_records = _run_stage("empty_load", empty_rounds, log_path)
    empty_sum = _summary(empty_records)
    print(f"\n[SUMMARY] empty_load: {empty_sum}")

    # Gate: do not continue if estop happened or success is very low.
    if empty_sum["estop"] > 0 or empty_sum["success_rate"] < 0.8:
        print("\nGate not passed. Stop before simplified task stage.")
        return

    go_next = _ask("Proceed to simplified task stage", ["y", "n"])
    if go_next != "y":
        print("Stopped by operator after empty_load stage.")
        return

    simple_records = _run_stage("simplified_task", simple_rounds, log_path)
    simple_sum = _summary(simple_records)
    print(f"\n[SUMMARY] simplified_task: {simple_sum}")

    final = {
        "timestamp": time.time(),
        "stage": "final_summary",
        "empty_load": empty_sum,
        "simplified_task": simple_sum,
    }
    _append_jsonl(log_path, final)
    print(f"\n[DONE] Protocol log saved to: {log_path}")


if __name__ == "__main__":
    main()
