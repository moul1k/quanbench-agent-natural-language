"""
QuanBench Agentic Evaluation Harness
=====================================
Test harness that provides all helper functions expected by QuanBench-44 test cases.
Compatible with Qiskit 2.x / qiskit-aer 0.17+
"""

import numpy as np
import unittest
import textwrap
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# ── Qiskit 2.x compatibility shims ──────────────────────────────────────────
# Some canonical solutions use qiskit.primitives.Sampler (removed in Qiskit 2.x).
# We inject the Aer-backed version so those tasks still run.
def apply_qiskit_shims():
    """Inject qiskit.primitives.Sampler and Estimator shims if not present (Qiskit 2.x removed them)."""
    import qiskit.primitives as _prim
    try:
        from qiskit.primitives import Sampler  # noqa: F401
    except ImportError:
        from qiskit_aer.primitives import Sampler as AerSampler
        _prim.Sampler = AerSampler

    try:
        from qiskit.primitives import Estimator  # noqa: F401
    except ImportError:
        from qiskit_aer.primitives import Estimator as AerEstimator
        _prim.Estimator = AerEstimator

    # Qiskit 2.x removed qiskit.execute() — shim it
    try:
        import qiskit as _qiskit
        if not hasattr(_qiskit, 'execute'):
            def _execute_shim(circuit, backend, **kwargs):
                from qiskit import transpile as _transpile
                shots = kwargs.get('shots', 1024)
                tqc = _transpile(circuit, backend)
                return backend.run(tqc, shots=shots)
            _qiskit.execute = _execute_shim
    except Exception:
        pass

apply_qiskit_shims()

# ── Global simulator ────────────────────────────────────────────────────────
SIM = AerSimulator()
DEFAULT_SHOTS = 10_000


# ── Core circuit runners ─────────────────────────────────────────────────────

def run_circuit(qc: QuantumCircuit, shots: int = DEFAULT_SHOTS) -> dict:
    """Run a circuit on AerSimulator, return raw counts dict."""
    tqc = transpile(qc, SIM)
    job = SIM.run(tqc, shots=shots)
    return job.result().get_counts()


def _normalize(counts: dict, shots: int) -> dict:
    """Normalize counts → probabilities, adding epsilon."""
    eps = 1e-10
    total = sum(counts.values()) or shots
    return {k: v / total + eps for k, v in counts.items()}


# ── KL-divergence helpers ─────────────────────────────────────────────────────

def compute_KL(qc1: QuantumCircuit, qc2: QuantumCircuit,
               shots: int = DEFAULT_SHOTS) -> float:
    """
    KL divergence D(P || Q) where P = canonical circuit, Q = generated circuit.
    Circuits must have measurements.
    """
    c1 = run_circuit(qc1, shots)
    c2 = run_circuit(qc2, shots)
    all_keys = sorted(set(c1) | set(c2))
    eps = 1e-10
    p = np.array([c1.get(k, 0) / shots for k in all_keys]) + eps
    q = np.array([c2.get(k, 0) / shots for k in all_keys]) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_KL_noexecute(oracle_result: dict, generated_result: dict) -> float:
    """
    KL divergence when oracle_result is already a counts dict (pre-computed),
    and generated_result is also a counts dict.
    Used by tasks where oracle provides raw counts (e.g. task 03).
    """
    all_keys = sorted(set(oracle_result) | set(generated_result))
    eps = 1e-10
    total_p = sum(oracle_result.values()) or 1
    total_q = sum(generated_result.values()) or 1
    p = np.array([oracle_result.get(k, 0) / total_p for k in all_keys]) + eps
    q = np.array([generated_result.get(k, 0) / total_q for k in all_keys]) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


# ── Gate analysis helpers ─────────────────────────────────────────────────────

def is_gate_count_subset(required: dict, actual: dict) -> bool:
    """
    Check that all gates in `required` appear in `actual` with at least
    the required count. Used by test_statical_assert in many tasks.

    Example: is_gate_count_subset({"h": 2, "cswap": 1}, circuit.count_ops())
    """
    for gate, count in required.items():
        if actual.get(gate, 0) < count:
            return False
    return True

def check_phase(qc: QuantumCircuit, bitstring: str) -> float:
    """
    Return the phase (in radians) of the amplitude for a given bitstring.
    Removes final measurements before computing statevector.
    Qiskit uses little-endian bit ordering.
    """
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    sv = Statevector(qc_no_meas)
    # Convert bitstring to index (Qiskit little-endian: reverse the bitstring)
    idx = int(bitstring[::-1], 2)
    amp = sv.data[idx]
    return float(np.angle(amp))


# ── Test runner ───────────────────────────────────────────────────────────────

def build_test_namespace(canonical_solution_code: str,
                          generated_code: str,
                          entry_point: str) -> dict:
    """
    Build the execution namespace for a test case by:
    1. Executing canonical solution → wraps it as cir_solution()
    2. Executing generated code    → wraps it as cir_generated()
    3. Injecting all helper functions
    """
    ns = {
        # Qiskit imports available to test code
        "QuantumCircuit": QuantumCircuit,
        "np": np,
        # Helper functions
        "run_circuit": run_circuit,
        "compute_KL": compute_KL,
        "compute_KL_noexecute": compute_KL_noexecute,
        "check_phase": check_phase,
        "is_gate_count_subset": is_gate_count_subset,
    }

    # Execute canonical solution
    exec(compile(canonical_solution_code, "<canonical>", "exec"), ns)
    canonical_fn = ns[entry_point]
    ns["cir_solution"] = canonical_fn

    # Execute generated code
    try:
        exec(compile(generated_code, "<generated>", "exec"), ns)
        generated_fn = ns.get(entry_point)
        if generated_fn is None:
            raise NameError(f"Entry point '{entry_point}' not found in generated code")
        ns["cir_generated"] = generated_fn
    except Exception as e:
        # Wrap failure so test will report it cleanly
        def _failing_generated():
            raise RuntimeError(f"Generated code failed to load: {e}")
        ns["cir_generated"] = _failing_generated

    return ns


def run_tests(task: dict, generated_code: str) -> dict:
    """
    Run the QuanBench test suite for a single task.

    Returns:
        {
            "task_id": str,
            "passed": bool,
            "num_passed": int,
            "num_total": int,
            "failures": list[str],
            "errors": list[str],
        }
    """
    task_id = task["task_id"]
    entry_point = task["entry_point"]
    canonical = task["canonical_solution"]
    test_code = task["test"]

    # Build namespace
    ns = build_test_namespace(canonical, generated_code, entry_point)

    # Ensure test code has a suite runner (tasks 01, 02, 09 lack one)
    suite_runner = textwrap.dedent("""
        suite = unittest.TestLoader().loadTestsFromTestCase(TestKLDivergence)
        _test_result = unittest.TestResult()
        suite.run(_test_result)
    """)
    if "suite" not in test_code:
        test_code = test_code + suite_runner

    # Inject unittest
    ns["unittest"] = unittest

    try:
        exec(compile(test_code, f"<test_{task_id}>", "exec"), ns)
    except AssertionError as e:
        return {
            "task_id": task_id,
            "passed": False,
            "num_passed": 0,
            "num_total": 2,
            "failures": [str(e)],
            "errors": [],
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "passed": False,
            "num_passed": 0,
            "num_total": 2,
            "failures": [],
            "errors": [str(e)],
        }

    # Read unittest result from namespace
    test_result = ns.get("_test_result")
    if test_result is None:
        # Older tasks that raise directly — if we got here, they passed
        return {"task_id": task_id, "passed": True,
                "num_passed": 2, "num_total": 2,
                "failures": [], "errors": []}

    failures = [str(f[1]) for f in test_result.failures]
    errors   = [str(e[1]) for e in test_result.errors]
    num_total  = test_result.testsRun
    num_passed = num_total - len(failures) - len(errors)

    return {
        "task_id":   task_id,
        "passed":    len(failures) == 0 and len(errors) == 0,
        "num_passed": num_passed,
        "num_total":  num_total,
        "failures":   failures,
        "errors":     errors,
    }
