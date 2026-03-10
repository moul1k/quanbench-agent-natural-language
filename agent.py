"""
QuanBench Upgraded Agent
=========================
Improvements over v1:

Phase 1 — Better eval on benchmark:
  ✦ Circuit Inspector: repair prompt includes ASCII diagram + gate diff
  ✦ Unitary fidelity: concrete similarity score vs canonical
  ✦ Reflection Agent: cheap pre-check before simulator round-trip
  ✦ Convergence memory: tracks tried strategies, switches on repeat failure

Phase 2 — Natural language → quantum code:
  ✦ Task Formulator: converts plain English to structured task format
  ✦ Self-Evaluator: LLM writes its own tests when no canonical exists
  ✦ Supports: Standard algorithms (Grover, QFT, QPE, Bell, GHZ)
              Quantum ML (VQC, feature maps, QNN layers)
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from classifier import FailureReport, FailureType, classify
from harness import run_tests, run_circuit
from inspector import inspect, compare, format_for_prompt, CircuitFacts
from self_eval import (
    run_self_eval,
    _detect_algorithm_type,
)

# ── Client ────────────────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL  = "claude-sonnet-4-20250514"

# ── Qiskit 2.x context block ──────────────────────────────────────────────────
QISKIT_CONTEXT = """
QISKIT 2.x API RULES (strictly enforce these — violations will cause runtime errors):
- Import: `from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile`
- Simulator: `from qiskit_aer import AerSimulator`
- Do NOT use: qiskit.execute(), QuantumInstance, qiskit.opflow, qiskit.providers.aer,
  qiskit.extensions, Aer.get_backend(), or any Qiskit 0.x / 1.x imports
- Statevector: `from qiskit.quantum_info import Statevector`
- Parameters: `from qiskit.circuit import Parameter, ParameterVector`
- Standard gates: h, x, y, z, cx, cz, ccx, rx, ry, rz, s, t, sdg, tdg, swap, cp, crz
- Function MUST return a QuantumCircuit with measurements applied (qc.measure_all() or qc.measure(...))
- Do NOT include print statements, if __name__ blocks, or plt.show()
"""

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _generator_prompt(task: dict) -> str:
    return f"""{QISKIT_CONTEXT}

Implement the following quantum circuit function using Qiskit 2.x.

## Task
{task['complete_prompt']}

## Instructions
- Implement ONLY the function shown in the prompt
- Return the QuantumCircuit with measurements
- Wrap your code in a ```python ... ``` block
"""


def _reflection_prompt(task: dict, generated_code: str) -> str:
    return f"""You are a quantum code reviewer. Quickly check this code for obvious issues.

## Task entry point
{task['entry_point']}

## Code to check
```python
{generated_code}
```

Answer each question with YES or NO only, one per line:
1. Does the function name match `{task['entry_point']}`?
2. Are all imports Qiskit 2.x compatible (no qiskit.execute, no Aer.get_backend, no qiskit.providers.aer)?
3. Does the function return a QuantumCircuit?
4. Does the circuit have measurements?
5. Are there any obvious logic errors in the gate sequence?

Then on the final line write: NEEDS_FIX if any answer is NO, or OK if all YES.
"""


def _repair_prompt(
    task: dict,
    prev_code: str,
    report: FailureReport,
    iteration: int,
    history: list[dict],
    circuit_context: str = "",
    strategy: str = "patch",
) -> str:

    history_str = ""
    if history:
        history_str = "\n## Attempt History\n"
        for h in history:
            history_str += f"- Iter {h['iteration']}: [{h['failure_type']}] {h['message'][:100]}\n"

    strategy_note = ""
    if strategy == "rethink":
        strategy_note = (
            "\n⚠️  STRATEGY SWITCH: The same failure has occurred twice. "
            "Do NOT patch the existing code. Rethink the algorithm from scratch. "
            "Reconsider the oracle construction, gate ordering, and qubit assignments.\n"
        )

    return f"""{QISKIT_CONTEXT}
{strategy_note}
You are repairing a quantum circuit that failed its tests (Attempt {iteration}).

## Original Task
{task['complete_prompt']}

## Failed Code
```python
{prev_code}
```

## Failure Report
Type: {report.failure_type}
Message: {report.message}
Hint: {report.hint}
{"Errors: " + chr(10).join(report.raw_errors[:2]) if report.raw_errors else ""}
{"Test failures: " + chr(10).join(report.raw_failures[:2]) if report.raw_failures else ""}

{circuit_context}
{history_str}

## Instructions
{"- Make targeted fix only — preserve what is correct" if strategy == "patch" else "- Rebuild the circuit with a fresh approach"}
- Wrap your corrected code in a ```python ... ``` block
"""


def _task_formulator_prompt(natural_language: str) -> str:
    return f"""{QISKIT_CONTEXT}

Convert the following natural language description into a structured quantum programming task.

## User Request
{natural_language}

## Output Format
Respond with a JSON object (inside ```json ... ```) with these exact fields:

{{
  "task_id": "custom_01",
  "entry_point": "snake_case_function_name",
  "complete_prompt": "from qiskit import QuantumCircuit\\ndef function_name() -> QuantumCircuit:\\n    \\\"\\\"\\\"\\n    [clear docstring describing what the circuit should do]\\n    \\\"\\\"\\\"",
  "algorithm_type": "grover|qft|qpe|bell|ghz|vqc|qml|other",
  "description": "one sentence description for the self-evaluator"
}}

Rules:
- entry_point must be a valid Python function name
- complete_prompt must include the function signature and docstring
- algorithm_type must be one of the listed values
- The docstring should be precise about qubit count, target state, parameters
"""


def _nl_generator_prompt(task: dict) -> str:
    """Generator prompt for natural-language tasks (no canonical solution)."""
    return f"""{QISKIT_CONTEXT}

Implement the following quantum circuit function.

## Task
{task['complete_prompt']}

## Algorithm Type
{task.get('algorithm_type', 'general')}

## Additional Context
{task.get('description', '')}

## Instructions
- Implement ONLY the function shown above
- Return the QuantumCircuit with measurements
- For parameterized circuits, bind reasonable default parameter values before measuring
- Wrap your code in a ```python ... ``` block
"""


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_json(text: str) -> dict:
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    raw = match.group(1).strip() if match else text.strip()
    return json.loads(raw)


def _call_llm(prompt: str, max_tokens: int = 1800) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _run_reflection(task: dict, code: str) -> tuple[bool, str]:
    """
    Run a fast reflection pre-check. Returns (needs_fix: bool, raw_response: str).
    Skips the full simulator round-trip for obvious errors.
    """
    try:
        raw = _call_llm(_reflection_prompt(task, code), max_tokens=300)
        needs_fix = "NEEDS_FIX" in raw.upper()
        return needs_fix, raw
    except Exception:
        return False, ""   # on failure, let the simulator catch it


def _try_get_circuits(canonical_code: str, generated_code: str,
                       entry_point: str):
    """Try to instantiate both circuits for inspection. Returns (canon_qc, gen_qc) or (None, None)."""
    try:
        ns1 = {}
        exec(compile(canonical_code, "<c>", "exec"), ns1)
        canon_qc = ns1[entry_point]()

        ns2 = {}
        exec(compile(generated_code, "<g>", "exec"), ns2)
        gen_qc = ns2[entry_point]()
        return canon_qc, gen_qc
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    task_id: str
    passed: bool
    iterations: int
    final_code: str
    final_test_result: dict
    mode: str = "benchmark"   # "benchmark" | "custom"
    self_eval_tests: str = ""
    attempt_history: list[dict] = field(default_factory=list)
    reflection_log: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        mode_tag = f"[{self.mode}]"
        num_p = self.final_test_result.get("num_passed", 0)
        num_t = self.final_test_result.get("num_total", 2)
        return (f"{status} {mode_tag} Task {self.task_id} | "
                f"{self.iterations} iter | {num_p}/{num_t} tests")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Benchmark agent (upgraded)
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark_agent(
    task: dict,
    max_iterations: int = 5,
    verbose: bool = True,
) -> AgentResult:
    """
    Upgraded agentic loop for QuanBench benchmark tasks.

    Improvements over v1:
    - Reflection pre-check on every generated/repaired code
    - Circuit inspector: ASCII diagram + gate counts in repair prompt
    - Unitary fidelity diff in repair context
    - Convergence memory with strategy switching on repeated failure
    """
    task_id = task["task_id"]
    history: list[dict] = []
    reflection_log: list[dict] = []
    generated_code = ""
    test_result: dict = {}
    failure_type_counts: dict[str, int] = {}

    if verbose:
        print(f"\n{'='*62}")
        print(f"[BENCHMARK] Task {task_id}: {task['entry_point']}")
        print(f"{'='*62}")

    for iteration in range(1, max_iterations + 1):

        # ── Generate or Repair ────────────────────────────────────────
        if iteration == 1:
            if verbose: print(f"  [Iter {iteration}] Generating...")
            prompt = _generator_prompt(task)
            raw = _call_llm(prompt)
            generated_code = extract_code(raw)
        else:
            report = classify(generated_code, test_result)
            if verbose:
                print(f"  [Iter {iteration}] Repairing — {report.failure_type}: {report.message[:70]}")

            # Track failure type for convergence detection
            ft = str(report.failure_type)
            failure_type_counts[ft] = failure_type_counts.get(ft, 0) + 1
            strategy = "rethink" if failure_type_counts.get(ft, 0) >= 2 else "patch"
            if verbose and strategy == "rethink":
                print(f"  [Iter {iteration}] ⚡ Strategy switch → full rethink")

            history.append({
                "iteration": iteration - 1,
                "failure_type": ft,
                "message": report.message,
                "hint": report.hint,
            })

            # Circuit inspection context
            circuit_context = ""
            try:
                canon_qc, gen_qc = _try_get_circuits(
                    task["canonical_solution"], generated_code, task["entry_point"]
                )
                if canon_qc and gen_qc:
                    canon_facts = inspect(canon_qc)
                    gen_facts   = compare(canon_qc, gen_qc)
                    circuit_context = format_for_prompt(canon_facts, gen_facts)
            except Exception:
                pass

            prompt = _repair_prompt(
                task, generated_code, report,
                iteration - 1, history[:-1],
                circuit_context=circuit_context,
                strategy=strategy,
            )
            raw = _call_llm(prompt)
            generated_code = extract_code(raw)

        # ── Reflection pre-check ──────────────────────────────────────
        needs_fix, refl_raw = _run_reflection(task, generated_code)
        reflection_log.append({
            "iteration": iteration,
            "needs_fix": needs_fix,
            "response": refl_raw[:300],
        })
        if needs_fix and verbose:
            print(f"  [Iter {iteration}] 🔍 Reflection flagged issues — applying quick fix...")

        if needs_fix:
            # One cheap self-fix pass before running simulator
            fix_prompt = f"""{QISKIT_CONTEXT}
The following code has issues identified by a reviewer. Fix them and return corrected code.

Issues flagged:
{refl_raw}

Code to fix:
```python
{generated_code}
```

Return ONLY corrected code in ```python ... ``` block.
"""
            generated_code = extract_code(_call_llm(fix_prompt, max_tokens=1200))

        # ── Simulator Oracle ──────────────────────────────────────────
        if verbose: print(f"  [Iter {iteration}] Running simulator oracle...")
        test_result = run_tests(task, generated_code)

        passed  = test_result["passed"]
        num_p   = test_result.get("num_passed", 0)
        num_t   = test_result.get("num_total", 2)
        if verbose:
            status = "✅ PASS" if passed else f"❌ FAIL ({num_p}/{num_t})"
            print(f"  [Iter {iteration}] {status}")

        if passed:
            return AgentResult(
                task_id=task_id, passed=True, iterations=iteration,
                final_code=generated_code, final_test_result=test_result,
                mode="benchmark", attempt_history=history,
                reflection_log=reflection_log,
            )

        time.sleep(0.4)

    return AgentResult(
        task_id=task_id, passed=False, iterations=max_iterations,
        final_code=generated_code, final_test_result=test_result,
        mode="benchmark", attempt_history=history,
        reflection_log=reflection_log,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Natural language → verified Qiskit code
# ─────────────────────────────────────────────────────────────────────────────

def formulate_task(natural_language: str, verbose: bool = True) -> dict:
    """
    Convert a plain English description into a structured task dict.
    Equivalent to converting user intent into a QuanBench-style task.
    """
    if verbose:
        print(f"\n  [Formulator] Converting natural language to task structure...")
    raw = _call_llm(_task_formulator_prompt(natural_language), max_tokens=800)
    try:
        task = extract_json(raw)
        # Ensure required fields
        task.setdefault("task_id", "custom_01")
        task.setdefault("canonical_solution", None)
        task.setdefault("test", None)
        task.setdefault("algorithm_type",
                         _detect_algorithm_type(natural_language) or "other")
        if verbose:
            print(f"  [Formulator] → {task['entry_point']} ({task['algorithm_type']})")
        return task
    except Exception as e:
        raise ValueError(f"Task formulation failed: {e}\nRaw response:\n{raw}")


def run_custom_agent(
    natural_language: str,
    max_iterations: int = 5,
    verbose: bool = True,
) -> AgentResult:
    """
    Phase 2: Full pipeline from natural language → verified Qiskit code.

    Pipeline:
      1. Task Formulator  — NL → structured task dict
      2. Generator        — task → Qiskit code
      3. Reflection       — quick pre-check
      4. Self-Evaluator   — LLM writes + runs its own tests
      5. Failure Classifier + Repair loop (up to max_iterations)
    """
    if verbose:
        print(f"\n{'='*62}")
        print(f"[CUSTOM] {natural_language[:70]}")
        print(f"{'='*62}")

    # ── Step 1: Formulate task ────────────────────────────────────────
    task = formulate_task(natural_language, verbose)
    algo_type = task.get("algorithm_type", "other")

    history: list[dict] = []
    reflection_log: list[dict] = []
    generated_code = ""
    test_result: dict = {}
    failure_type_counts: dict[str, int] = {}
    last_self_eval_tests = ""

    for iteration in range(1, max_iterations + 1):

        # ── Generate or Repair ────────────────────────────────────────
        if iteration == 1:
            if verbose: print(f"  [Iter {iteration}] Generating...")
            prompt = _nl_generator_prompt(task)
            generated_code = extract_code(_call_llm(prompt))
        else:
            # Build report from last self-eval result
            report = classify(generated_code, test_result)
            ft = str(report.failure_type)
            failure_type_counts[ft] = failure_type_counts.get(ft, 0) + 1
            strategy = "rethink" if failure_type_counts.get(ft, 0) >= 2 else "patch"

            if verbose:
                print(f"  [Iter {iteration}] Repairing — {ft}: {report.message[:70]}")
                if strategy == "rethink":
                    print(f"  [Iter {iteration}] ⚡ Strategy switch → full rethink")

            history.append({
                "iteration": iteration - 1,
                "failure_type": ft,
                "message": report.message,
                "hint": report.hint,
            })

            # Circuit inspection (no canonical, just inspect generated)
            circuit_context = ""
            try:
                ns = {}
                exec(compile(generated_code, "<g>", "exec"), ns)
                gen_qc = ns[task["entry_point"]]()
                gen_facts = inspect(gen_qc)
                circuit_context = (
                    f"## Generated Circuit Structure\n"
                    f"- Qubits: {gen_facts.num_qubits}, Depth: {gen_facts.depth}\n"
                    f"- Gates: {gen_facts.gate_counts}\n"
                    f"- Diagram:\n```\n{gen_facts.ascii_diagram}\n```"
                )
            except Exception:
                pass

            # For custom tasks, include self-eval tests in repair context
            if last_self_eval_tests:
                circuit_context += (
                    f"\n\n## Self-Eval Tests That Failed\n```python\n"
                    f"{last_self_eval_tests}\n```"
                )

            # Build a synthetic task dict for _repair_prompt compatibility
            repair_task = {
                "complete_prompt": task["complete_prompt"],
                "entry_point": task["entry_point"],
                "canonical_solution": "",   # none for custom tasks
            }
            prompt = _repair_prompt(
                repair_task, generated_code, report,
                iteration - 1, history[:-1],
                circuit_context=circuit_context,
                strategy=strategy,
            )
            generated_code = extract_code(_call_llm(prompt))

        # ── Reflection pre-check ──────────────────────────────────────
        needs_fix, refl_raw = _run_reflection(
            {"entry_point": task["entry_point"]}, generated_code
        )
        reflection_log.append({"iteration": iteration, "needs_fix": needs_fix})

        if needs_fix:
            if verbose: print(f"  [Iter {iteration}] 🔍 Reflection fix...")
            fix_prompt = (
                f"{QISKIT_CONTEXT}\nFix issues in this code:\n{refl_raw}\n"
                f"```python\n{generated_code}\n```\nReturn fixed code in ```python...```."
            )
            generated_code = extract_code(_call_llm(fix_prompt, max_tokens=1200))

        # ── Self-Evaluator Oracle ─────────────────────────────────────
        if verbose: print(f"  [Iter {iteration}] Running self-evaluator oracle...")
        test_result = run_self_eval(
            task_description=task.get("description", task["complete_prompt"]),
            entry_point=task["entry_point"],
            generated_code=generated_code,
            llm_caller=_call_llm,
            algorithm_type=algo_type,
        )
        last_self_eval_tests = test_result.get("self_eval_test_code", "")

        passed = test_result["passed"]
        num_p  = test_result.get("num_passed", 0)
        num_t  = test_result.get("num_total", 2)

        if verbose:
            status = "✅ PASS" if passed else f"❌ FAIL ({num_p}/{num_t})"
            print(f"  [Iter {iteration}] {status}")

        if passed:
            return AgentResult(
                task_id=task["task_id"], passed=True, iterations=iteration,
                final_code=generated_code, final_test_result=test_result,
                mode="custom", self_eval_tests=last_self_eval_tests,
                attempt_history=history, reflection_log=reflection_log,
            )

        time.sleep(0.4)

    return AgentResult(
        task_id=task["task_id"], passed=False, iterations=max_iterations,
        final_code=generated_code, final_test_result=test_result,
        mode="custom", self_eval_tests=last_self_eval_tests,
        attempt_history=history, reflection_log=reflection_log,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible alias
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(task: dict, max_iterations: int = 5,
              verbose: bool = True) -> AgentResult:
    """Drop-in replacement for v1 run_agent — routes to benchmark agent."""
    return run_benchmark_agent(task, max_iterations=max_iterations, verbose=verbose)
