# QuanBench Agentic Evaluator

An agentic system that generates, tests, and self-repairs Qiskit quantum circuits.
Built on the QuanBench-44 benchmark (arXiv:2510.16779) with two operating modes.

---

## What it does

**Benchmark mode** — evaluates LLMs on the QuanBench-44/117 task suite using a
multi-iteration repair loop guided by a quantum simulator oracle.

**Custom mode (Phase 2)** — takes plain English descriptions of quantum algorithms,
generates verified Qiskit code, and self-evaluates it without any canonical solution.

---

## Architecture

```
Natural language  ──►  Task Formulator
                              │
Benchmark task  ─────────────┤
                              ▼
                        Generator Agent
                              │
                              ▼
                      Reflection Pre-check
                      (cheap syntax/API scan)
                              │
                         ┌────┴────┐
                         │  Fixed? │──yes──►  loop continues
                         └────┬────┘
                              │ no
                              ▼
                    Simulator-Oracle / Self-Evaluator
                              │
                         ┌────┴────┐
                         │  PASS?  │──yes──► done ✅
                         └────┬────┘
                              │ fail
                              ▼
                     Failure Classifier
                     (6 typed buckets)
                              │
                     Circuit Inspector
                     (unitary fidelity,
                      ASCII diagram diff)
                              │
                     Repair Agent
                     (patch or rethink)
                              │
                     back to Simulator
                     (max N iterations)
```
## Results

| Mode | Tasks | Passed | Pass Rate |
|------|-------|--------|-----------|
| Benchmark (QuanBench-44) | 16 | 11 | **68.8%** |
| Paper baseline (no repair loop) | — | — | <40% |
| Custom NL task (Phase 2) | 1 | 1 | **100%** |

Repair loop improves Pass@1 (12.5%) → Pass@5 (68.8%) — a **+56.3pp gain**.
94% of failures are `CIRCUIT_STRUCTURE`: circuits compile and run but produce
wrong quantum distributions. Syntactic correctness ≠ semantic correctness.

### Files

| File | Role |
|------|------|
| `harness.py` | Test execution engine. Provides all QuanBench helper functions (`run_circuit`, `compute_KL`, `compute_KL_noexecute`, `check_phase`, `is_gate_count_subset`). Includes Qiskit 2.x compatibility shims. |
| `classifier.py` | 6-bucket failure classifier: `OUTDATED_API`, `SYNTAX_ERROR`, `MISSING_ENTRYPOINT`, `RUNTIME_ERROR`, `CIRCUIT_STRUCTURE`, `ALGORITHM_LOGIC`. Routes each failure to the right repair strategy. |
| `inspector.py` | Extracts circuit structure (gate counts, depth, ASCII diagram). Computes unitary process fidelity vs canonical for circuits ≤8 qubits. Injected into every repair prompt. |
| `self_eval.py` | Phase 2 oracle. When there is no canonical solution, asks the LLM to write its own behavioral test assertions, then runs them. Includes hard structural checks (algorithm priors) that can't be overridden. |
| `agent.py` | Orchestrator. Runs benchmark and custom pipelines. Contains all prompt builders, reflection logic, convergence memory, and strategy switching. |
| `evaluate.py` | CLI entry point. Handles benchmark/custom/both modes, reporting, and JSON output. |

---

## Deployment Steps

### Step 1 — Clone or copy the files

```bash
mkdir quanbench_agent && cd quanbench_agent
# Copy all 6 .py files + requirements.txt + .env.example here
```

### Step 2 — Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `qiskit>=2.0.0` — quantum circuit framework
- `qiskit-aer>=0.17.0` — simulator backend
- `qiskit-algorithms>=0.3.0` — VQE/QAOA support
- `qiskit-optimization>=0.6.0` — QUBO/knapsack tasks
- `anthropic>=0.40.0` — Claude API client
- `numpy>=1.26.0`

### Step 4 — Set your API key

```bash
cp .env.example .env
# Edit .env and paste your key, then:
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 5 — Download the benchmark data

```bash
# Download QuanBench-44 from the paper's repo:
curl -O https://raw.githubusercontent.com/GuoXiaoYu1125/Quanbench/main/QuanBench44.jsonl
# Optional: download the 117-task version
curl -O https://raw.githubusercontent.com/GuoXiaoYu1125/Quanbench/main/Quanbench117.jsonl
```

### Step 6 — Run

```bash
# Quick sanity check on 3 known-good tasks
python evaluate.py --tasks 01 10 11 --max-iter 3

# Full benchmark (44 tasks, ~45–90 min depending on pass rate)
python evaluate.py --max-iter 5

# Phase 2: natural language → verified circuit
python evaluate.py --custom "Implement Grover search targeting |101> on 3 qubits"
python evaluate.py --custom "Build a 4-qubit Quantum Fourier Transform circuit"
python evaluate.py --custom "Create a quantum ML feature map with ZZ entanglement for 2 features"

# Both modes in one run
python evaluate.py --tasks 01 10 11 --custom "GHZ state on 4 qubits" --both

# Batch custom tasks
python evaluate.py --custom-file my_tasks.txt
```

---

## Output

Every run produces `results.json` (configurable via `--output`) with:

```json
{
  "task_id": "01",
  "mode": "benchmark",
  "passed": true,
  "iterations": 2,
  "final_code": "...",
  "test_result": {"passed": true, "num_passed": 2, "num_total": 2, ...},
  "self_eval_tests": "",
  "attempt_history": [{"iteration": 1, "failure_type": "CIRCUIT_STRUCTURE", ...}],
  "reflection_log": [{"iteration": 1, "needs_fix": false}]
}
```

---

## Tunable parameters

**In `evaluate.py` (CLI flags):**

| Flag | Default | Description |
|------|---------|-------------|
| `--max-iter` | 5 | Max repair iterations per task |
| `--benchmark` | `QuanBench44.jsonl` | Benchmark file path |
| `--output` | `results.json` | Results output path |

**In `agent.py`:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `claude-sonnet-4-20250514` | LLM model for all calls |

**In `harness.py`:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_SHOTS` | `10_000` | Simulator shots per circuit run |

**In `inspector.py`:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_UNITARY_QUBITS` | `8` | Max qubits for unitary diff (O(4^n) cost) |

---

## Benchmark compatibility

QuanBench-44 was published targeting Qiskit 0.x. This harness includes shims
so it runs on Qiskit 2.x:

- `qiskit.primitives.Sampler` → `qiskit_aer.primitives.Sampler`
- `qiskit.primitives.Estimator` → `qiskit_aer.primitives.Estimator`
- `qiskit.execute()` → `transpile() + backend.run()`

Of the 44 canonical solutions, 30 pass with these shims. The remaining 14 have
deeper Qiskit 2.x incompatibilities in the canonical solution itself (not in
your generated code) — they are valid evaluation targets but their ground truth
cannot be verified automatically with Qiskit 2.x.

The recommended stable task set for evaluation is tasks:
`01 06 07 08 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 36 37`

---

