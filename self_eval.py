"""
Self-Evaluator
==============
For Phase 2 (natural language → quantum code), there is no canonical solution
to compare against. This module:

1. Takes a natural language task description and generated Qiskit code
2. Uses an LLM to write test assertions appropriate for that task
3. Executes those tests against the generated circuit
4. Returns a structured result compatible with harness.run_tests()

The self-evaluator writes two classes of tests:
  - Behavioral: measurement outcome distributions match expected quantum behavior
  - Structural: circuit contains the expected gate primitives

For Standard Algorithms (Grover, QFT, QPE, etc.) it also applies
known theoretical checks (e.g. "Grover on n qubits should peak at marked state").
"""

import re
import textwrap
import numpy as np
import unittest
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, Operator

SIM = AerSimulator()

# ─────────────────────────────────────────────────────────────────────────────
# Known theoretical properties for standard algorithms
# These are injected as hard constraints alongside LLM-written tests
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHM_PRIORS = {
    "grover": {
        "description": "Grover's search should amplify the marked state to appear with high probability (>50% for small n).",
        "min_gates": {"h": 1},
        "notes": "Must contain Hadamard gates for superposition and an oracle+diffusion structure."
    },
    "qft": {
        "description": "QFT should produce uniform superposition from |0> with specific phase relationships.",
        "min_gates": {"h": 1},
        "notes": "Must contain Hadamard and controlled-phase (cp or cz) gates."
    },
    "qpe": {
        "description": "QPE should encode eigenvalue phase in ancilla register with high probability.",
        "min_gates": {"h": 1},
        "notes": "Requires QFT inverse sub-circuit."
    },
    "bell": {
        "description": "Bell state circuit should produce exactly two outcomes with equal probability.",
        "min_gates": {"h": 1, "cx": 1},
        "notes": "H then CNOT is the canonical Bell state construction."
    },
    "ghz": {
        "description": "GHZ state: should produce |000...0> and |111...1> with ~equal probability.",
        "min_gates": {"h": 1, "cx": 1},
        "notes": "H on first qubit, then chain of CNOTs."
    },
    "vqc": {
        "description": "Variational quantum circuit: parameterized circuit with rotation gates and entanglers.",
        "min_gates": {"rx": 1},
        "notes": "Should contain parameterized rotation gates (rx, ry, rz) and entangling gates."
    },
    "qml": {
        "description": "Quantum ML circuit: feature map or ansatz structure with data encoding.",
        "min_gates": {"rx": 1},
        "notes": "Should have data-encoding rotations and entanglement layers."
    },
}


def _detect_algorithm_type(description: str) -> Optional[str]:
    """Heuristically detect algorithm type from natural language description."""
    desc_lower = description.lower()
    if "grover" in desc_lower:
        return "grover"
    if "fourier" in desc_lower or "qft" in desc_lower:
        return "qft"
    if "phase estimation" in desc_lower or "qpe" in desc_lower:
        return "qpe"
    if "bell state" in desc_lower or "bell pair" in desc_lower:
        return "bell"
    if "ghz" in desc_lower:
        return "ghz"
    if "variational" in desc_lower or "vqc" in desc_lower or "ansatz" in desc_lower:
        return "vqc"
    if "machine learning" in desc_lower or "quantum ml" in desc_lower or \
       "feature map" in desc_lower or "qnn" in desc_lower:
        return "qml"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM-written test generation
# ─────────────────────────────────────────────────────────────────────────────

SELF_EVAL_SYSTEM = """You are a quantum computing expert writing test assertions
for a generated Qiskit circuit. Your tests must be executable Python using only
the helper functions and imports listed below.

Available helpers (already in scope, do NOT import them):
  run_circuit(qc) -> dict          # runs circuit, returns {bitstring: count} with 10000 shots
  is_gate_count_subset(req, actual) -> bool  # checks gate counts
  np                               # numpy
  QuantumCircuit                   # qiskit QuantumCircuit

The generated circuit function is called: generated_circuit = cir_generated()
Do NOT call cir_solution() — there is no canonical solution.

Write a Python class TestSelfEval(unittest.TestCase) with 2–4 test methods.
Each test should check ONE specific property. Tests must be deterministic and
robust to small statistical fluctuations (use delta tolerances on counts).

Focus on:
1. Measurement outcome distribution (which states dominate, rough ratios)
2. Gate structure (presence of key gates via is_gate_count_subset)
3. No crash / valid circuit returned

Output ONLY the test class and nothing else. No imports, no if __name__ == '__main__'.
"""


def generate_self_eval_tests(
    task_description: str,
    entry_point: str,
    generated_code: str,
    llm_caller,           # callable(prompt: str) -> str
    algorithm_type: Optional[str] = None,
) -> str:
    """
    Ask an LLM to write test assertions for a natural language task.
    Returns a string containing a TestSelfEval class.
    """
    algo_hint = ""
    if algorithm_type and algorithm_type in ALGORITHM_PRIORS:
        p = ALGORITHM_PRIORS[algorithm_type]
        algo_hint = f"\nAlgorithm context: {p['description']}\nNotes: {p['notes']}\n"

    prompt = f"""{SELF_EVAL_SYSTEM}

## Task Description
{task_description}
{algo_hint}

## Generated Circuit Code
```python
{generated_code}
```

## Entry Point
The function is named: {entry_point}

Write TestSelfEval(unittest.TestCase) now:"""

    raw = llm_caller(prompt)

    # Extract code block
    match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no fences, assume entire response is code
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded structural checks for known algorithms
# These run REGARDLESS of what the LLM writes — they're non-negotiable
# ─────────────────────────────────────────────────────────────────────────────

def _structural_checks(qc: QuantumCircuit, algorithm_type: Optional[str]) -> list[str]:
    """Return list of failure messages for hard structural violations."""
    failures = []
    gate_counts = dict(qc.count_ops())

    if not qc.num_clbits:
        failures.append("Circuit has no classical bits — measurements cannot be read.")

    if "measure" not in gate_counts:
        failures.append("Circuit has no measurements.")

    if algorithm_type and algorithm_type in ALGORITHM_PRIORS:
        required = ALGORITHM_PRIORS[algorithm_type].get("min_gates", {})
        for gate, count in required.items():
            if gate_counts.get(gate, 0) < count:
                failures.append(
                    f"Algorithm '{algorithm_type}' requires at least {count} '{gate}' gate(s), "
                    f"but found {gate_counts.get(gate, 0)}."
                )

    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Main self-evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_self_eval(
    task_description: str,
    entry_point: str,
    generated_code: str,
    llm_caller,
    algorithm_type: Optional[str] = None,
) -> dict:
    """
    Run self-evaluation on generated code for a natural language task.

    Returns the same structure as harness.run_tests() for compatibility
    with the existing agent loop.
    """
    from harness import run_circuit, is_gate_count_subset, apply_qiskit_shims
    apply_qiskit_shims()

    # ── Step 1: Load the generated function ──────────────────────────
    ns = {
        "QuantumCircuit": QuantumCircuit,
        "np": np,
        "run_circuit": run_circuit,
        "is_gate_count_subset": is_gate_count_subset,
        "unittest": unittest,
    }
    try:
        exec(compile(generated_code, "<generated>", "exec"), ns)
        generated_fn = ns.get(entry_point)
        if generated_fn is None:
            return {
                "task_id": "custom",
                "passed": False,
                "num_passed": 0,
                "num_total": 1,
                "failures": [],
                "errors": [f"Entry point '{entry_point}' not found in generated code."],
            }
        ns["cir_generated"] = generated_fn
    except Exception as e:
        return {
            "task_id": "custom",
            "passed": False,
            "num_passed": 0,
            "num_total": 1,
            "failures": [],
            "errors": [f"Code failed to load: {e}"],
        }

    # ── Step 2: Instantiate circuit for structural checks ─────────────
    try:
        qc = generated_fn()
        struct_failures = _structural_checks(qc, algorithm_type)
    except Exception as e:
        return {
            "task_id": "custom",
            "passed": False,
            "num_passed": 0,
            "num_total": 1,
            "failures": [],
            "errors": [f"Circuit instantiation failed: {e}"],
        }

    if struct_failures:
        return {
            "task_id": "custom",
            "passed": False,
            "num_passed": 0,
            "num_total": len(struct_failures),
            "failures": struct_failures,
            "errors": [],
        }

    # ── Step 3: LLM writes and runs behavioral tests ──────────────────
    test_class_code = generate_self_eval_tests(
        task_description, entry_point, generated_code,
        llm_caller, algorithm_type
    )

    full_test_code = textwrap.dedent(f"""
{test_class_code}

suite = unittest.TestLoader().loadTestsFromTestCase(TestSelfEval)
_test_result = unittest.TestResult()
suite.run(_test_result)
""")

    try:
        exec(compile(full_test_code, "<self_eval>", "exec"), ns)
    except Exception as e:
        return {
            "task_id": "custom",
            "passed": False,
            "num_passed": 0,
            "num_total": 1,
            "failures": [],
            "errors": [f"Self-eval test code crashed: {e}"],
        }

    test_result = ns.get("_test_result")
    if test_result is None:
        return {"task_id": "custom", "passed": True,
                "num_passed": 1, "num_total": 1, "failures": [], "errors": []}

    failures = [str(f[1]) for f in test_result.failures]
    errors   = [str(e[1]) for e in test_result.errors]
    num_total  = test_result.testsRun
    num_passed = num_total - len(failures) - len(errors)

    return {
        "task_id": "custom",
        "passed": len(failures) == 0 and len(errors) == 0,
        "num_passed": num_passed,
        "num_total": num_total,
        "failures": failures,
        "errors": errors,
        "self_eval_test_code": test_class_code,  # expose for debugging
    }
