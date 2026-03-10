"""
QuanBench Evaluate
===================
Supports three modes:

  Benchmark mode  — run against QuanBench-44/117 tasks with a canonical oracle
  Custom mode     — natural language → Qiskit code → self-evaluation
  Both            — run benchmark tasks AND custom tasks in one report

Usage:
  python evaluate.py                                          # all 44 benchmark tasks
  python evaluate.py --tasks 01 10 11 12 --max-iter 5        # specific tasks
  python evaluate.py --custom "Grover search for |101>"      # custom NL task
  python evaluate.py --custom-file my_tasks.txt              # batch custom
  python evaluate.py --tasks 01 --custom "QFT on 4 qubits" --both
"""

import argparse
import json
import sys
import time
from collections import defaultdict

from agent import run_benchmark_agent, run_custom_agent, AgentResult


def load_tasks(jsonl_path: str, task_ids=None):
    tasks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    if task_ids:
        tasks = [t for t in tasks if t["task_id"] in task_ids]
    return tasks


def _section_report(label: str, subset: list):
    if not subset:
        return
    passed = [r for r in subset if r.passed]
    total  = len(subset)
    pct    = len(passed) / total * 100 if total else 0

    print(f"\n  {'─'*56}")
    print(f"  {label}")
    print(f"  {'─'*56}")
    print(f"  Tasks: {total}  │  Passed: {len(passed)}  ({pct:.1f}%)")

    if passed:
        iter_dist = defaultdict(int)
        for r in passed:
            iter_dist[r.iterations] += 1
        dist_str = "  ".join(f"iter{k}={v}" for k, v in sorted(iter_dist.items()))
        print(f"  Pass by iter: {dist_str}")

    fail_types = defaultdict(int)
    for r in subset:
        for h in r.attempt_history:
            fail_types[h["failure_type"]] += 1
    if fail_types:
        top = sorted(fail_types.items(), key=lambda x: -x[1])[:4]
        print(f"  Failure types: " + "  ".join(f"{k}({v})" for k, v in top))

    print()
    for r in subset:
        print(f"    {r.summary()}")


def print_report(results: list, elapsed: float):
    bench  = [r for r in results if r.mode == "benchmark"]
    custom = [r for r in results if r.mode == "custom"]

    print("\n" + "="*60)
    print("  QUANBENCH AGENTIC EVALUATION REPORT")
    print("="*60)
    print(f"  Total tasks : {len(results)}")
    avg = f"  ({elapsed/len(results):.1f}s avg)" if results else ""
    print(f"  Total time  : {elapsed:.1f}s{avg}")

    _section_report("BENCHMARK MODE", bench)
    _section_report("CUSTOM MODE (natural language)", custom)
    print("="*60 + "\n")


def save_results(results: list, out_path: str):
    data = [{
        "task_id":         r.task_id,
        "mode":            r.mode,
        "passed":          r.passed,
        "iterations":      r.iterations,
        "final_code":      r.final_code,
        "test_result":     r.final_test_result,
        "self_eval_tests": r.self_eval_tests,
        "attempt_history": r.attempt_history,
        "reflection_log":  r.reflection_log,
    } for r in results]
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved → {out_path}")


def main():
    p = argparse.ArgumentParser(description="QuanBench Agentic Evaluator")
    p.add_argument("--benchmark",    default="QuanBench44.jsonl")
    p.add_argument("--tasks",        nargs="*", default=None)
    p.add_argument("--max-iter",     type=int, default=5)
    p.add_argument("--output",       default="results.json")
    p.add_argument("--verbose",      action="store_true", default=True)
    p.add_argument("--custom",       type=str, default=None)
    p.add_argument("--custom-file",  type=str, default=None)
    p.add_argument("--both",         action="store_true", default=False)
    args = p.parse_args()

    results = []
    t0 = time.time()

    run_bench  = not (args.custom or args.custom_file) or args.both
    run_custom = bool(args.custom or args.custom_file)

    if run_bench:
        try:
            tasks = load_tasks(args.benchmark, args.tasks)
            print(f"\nBenchmark mode: {len(tasks)} task(s), max {args.max_iter} iter")
            for task in tasks:
                results.append(run_benchmark_agent(
                    task, max_iterations=args.max_iter, verbose=args.verbose))
        except FileNotFoundError:
            print(f"Benchmark file not found: {args.benchmark}")
            if not run_custom:
                sys.exit(1)

    if run_custom:
        nl_tasks = []
        if args.custom:
            nl_tasks.append(args.custom)
        if args.custom_file:
            with open(args.custom_file) as f:
                nl_tasks += [ln.strip() for ln in f if ln.strip()]
        print(f"\nCustom mode: {len(nl_tasks)} task(s), max {args.max_iter} iter")
        for nl in nl_tasks:
            results.append(run_custom_agent(
                nl, max_iterations=args.max_iter, verbose=args.verbose))

    if not results:
        print("No tasks ran.")
        sys.exit(1)

    print_report(results, time.time() - t0)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
