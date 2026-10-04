"""
evaluation/run_eval.py - Run the evaluation set through the guarded agent
=========================================================================
Collects raw results only; scoring is a separate step (score.py), so scoring
can change without paying to rerun the agent.

    python -m evaluation.run_eval                        # full run
    python -m evaluation.run_eval --only 1 16 33         # a few questions
    python -m evaluation.run_eval --groups A E           # some groups
    python -m evaluation.run_eval --resume 20261004-1200 # continue a stopped run

Output: output/eval/<run_id>/
    results.jsonl   one line per question per repeat, saved as each finishes
    meta.json       models, git commit, date, and fingerprints of the inputs
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

EVAL_SET = Path("evaluation/eval_set.jsonl")
OUT_ROOT = Path("output/eval")

# Repeated 3 times to measure run-to-run variation. Covers every group.
REPEAT_SUBSET = [1, 10, 12, 16, 19, 24, 26, 29, 37, 47]
WARMUP_QUESTION = "How many recognitions are in each category?"


def load_cases() -> list[dict]:
    with open(EVAL_SET, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    except OSError:
        return "missing"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def git_commit() -> str:
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    # Uncommitted changes mean the commit alone does not describe the code that ran
    return commit + ("+dirty" if _git("status", "--porcelain") else "")


def write_meta(run_dir: Path, args) -> None:
    from src.rag import agent_graph as ag
    from src.rag.guardrails import input_guard as ig
    from src.rag.schema import load_schema, schema_fingerprint

    data_path = Path(load_schema()["dataset"]["path"])
    meta = {
        "run_id": run_dir.name,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": "langgraph_agent_guarded",
        "agent_model": ag.GROQ_MODEL,
        "scope_model": ig.settings()["scope"]["model"],
        "injection_model": ig.settings()["injection"]["model"],
        "git_commit": git_commit(),
        "fingerprints": {
            "data": _file_hash(data_path),
            "schema": schema_fingerprint(),
            "eval_set": _file_hash(EVAL_SET),
            "guardrails_config": _file_hash(Path("config/guardrails.yaml")),
        },
        "repeats": args.repeats,
        "repeat_subset": REPEAT_SUBSET,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def done_keys(results_path: Path) -> set[tuple[int, int]]:
    """(question id, repeat) pairs already saved, so a resumed run skips them."""
    if not results_path.exists():
        return set()
    with open(results_path, encoding="utf-8") as f:
        return {(r["id"], r["repeat"]) for r in map(json.loads, f) if r}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation set")
    parser.add_argument("--only", type=int, nargs="+", help="question ids to run")
    parser.add_argument("--groups", nargs="+", help="groups to run, e.g. A E")
    parser.add_argument("--repeats", type=int, default=3,
                        help="runs per question in the repeat subset (others run once)")
    parser.add_argument("--resume", help="run id to continue")
    args = parser.parse_args()

    cases = load_cases()
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    if args.groups:
        cases = [c for c in cases if c["group"] in args.groups]

    run_id = args.resume or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    finished = done_keys(results_path)

    jobs = [
        (c, r) for c in cases
        for r in range(args.repeats if c["id"] in REPEAT_SUBSET else 1)
        if (c["id"], r) not in finished
    ]
    print(f"Run {run_id}: {len(jobs)} to do, {len(finished)} already saved")
    if not jobs:
        return

    # Tag every LLM call in the cost log with this run's id, so the scorer can
    # total the FULL cost (guard models included). Must be set before the
    # agent (and so the cost tracker) is imported.
    os.environ["RUN_ID"] = run_id

    # Import here: loading the agent loads models, which the --help path doesn't need
    from src.rag.agent_graph import chat_agent
    if not (run_dir / "meta.json").exists():
        write_meta(run_dir, args)

    print("Warm-up (not saved): loading models...")
    chat_agent(WARMUP_QUESTION, session_id=f"eval-{run_id}-warmup")

    total_cost = 0.0
    start = time.perf_counter()
    for n, (case, repeat) in enumerate(jobs, 1):
        result = chat_agent(case["question"], session_id=f"eval-{run_id}-{case['id']}-{repeat}")
        record = {
            "id": case["id"],
            "group": case["group"],
            "repeat": repeat,
            "question": case["question"],
            "answer": result.get("answer", ""),
            "tools_used": result.get("tools_used", []),
            "tool_calls": result.get("tool_calls", []),
            "input_label": result.get("input_label", ""),
            "guard": result.get("guard", ""),
            "pii_redacted": result.get("pii_redacted", 0),
            "contexts": result.get("contexts", []),
            "timings": result.get("timings", {}),
            "processing_s": result.get("processing_s", result.get("latency_s")),
            "wait_s": result.get("wait_s", 0),
            "tokens": result.get("tokens", {}),
            "cost_usd": result.get("cost_usd", 0),
            "error": result.get("error", ""),
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        total_cost += record["cost_usd"] or 0
        status = "ERROR" if record["error"] else f"{record['input_label']}/{record['guard']}"
        print(f"  [{n}/{len(jobs)}] Q{case['id']} r{repeat} {status:<32} "
              f"{record['processing_s']}s  ${total_cost:.4f} so far")

    minutes = (time.perf_counter() - start) / 60
    print(f"\nDone in {minutes:.1f} min. Results: {results_path}")
    print(f"Next: python -m evaluation.score {run_id}")


if __name__ == "__main__":
    main()
