"""
Failure Classifier
==================
Categorizes why a generated quantum circuit failed, to route repair strategy.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re


class FailureType(str, Enum):
    OUTDATED_API     = "OUTDATED_API"       # deprecated qiskit imports/methods
    SYNTAX_ERROR     = "SYNTAX_ERROR"       # code doesn't parse
    RUNTIME_ERROR    = "RUNTIME_ERROR"      # code runs but crashes
    CIRCUIT_STRUCTURE = "CIRCUIT_STRUCTURE" # circuit builds but structure wrong
    ALGORITHM_LOGIC  = "ALGORITHM_LOGIC"    # circuit runs, semantically wrong
    MISSING_ENTRYPOINT = "MISSING_ENTRYPOINT"  # function name not found
    UNKNOWN          = "UNKNOWN"


# Patterns that indicate outdated Qiskit API usage
_OUTDATED_PATTERNS = [
    r"from qiskit\.providers\.aer",
    r"from qiskit\.test\.mock",
    r"qiskit\.execute\(",
    r"from qiskit import execute",
    r"backend\.run\(",
    r"Aer\.get_backend",
    r"from qiskit\.visualization",
    r"QuantumInstance",
    r"from qiskit\.algorithms",
    r"qiskit\.opflow",
    r"from qiskit\.utils import QuantumInstance",
    r"job\.result\(\)\.get_statevector",   # old statevector API
]

_OUTDATED_RE = re.compile("|".join(_OUTDATED_PATTERNS))


@dataclass
class FailureReport:
    failure_type: FailureType
    message: str
    hint: str = ""
    raw_errors: list[str] = field(default_factory=list)
    raw_failures: list[str] = field(default_factory=list)

    def __str__(self):
        return f"[{self.failure_type}] {self.message}\nHint: {self.hint}"


def classify(
    generated_code: str,
    test_result: dict,
) -> FailureReport:
    """
    Given the generated code and the test result dict from harness.run_tests(),
    return a FailureReport with the most likely root cause.
    """
    errors   = test_result.get("errors", [])
    failures = test_result.get("failures", [])
    all_messages = "\n".join(errors + failures)

    # ── 1. Syntax error ───────────────────────────────────────────────
    try:
        compile(generated_code, "<check>", "exec")
    except SyntaxError as e:
        return FailureReport(
            FailureType.SYNTAX_ERROR,
            f"SyntaxError at line {e.lineno}: {e.msg}",
            hint="Fix the Python syntax error before re-running.",
            raw_errors=errors,
        )

    # ── 2. Outdated API (check code pattern BEFORE error messages) ────
    if _OUTDATED_RE.search(generated_code):
        match = _OUTDATED_RE.search(generated_code)
        return FailureReport(
            FailureType.OUTDATED_API,
            f"Detected outdated Qiskit API usage: '{match.group()}'",
            hint=(
                "Use Qiskit 2.x APIs only. "
                "Replace qiskit.execute() with transpile() + backend.run(). "
                "Use qiskit_aer.AerSimulator, not Aer.get_backend(). "
                "Do not import from qiskit.providers.aer or qiskit.algorithms."
            ),
            raw_errors=errors,
        )

    # ── 3b. Missing entry point ───────────────────────────────────────
    if "not found in generated code" in all_messages or (
        "NameError" in all_messages and "entry_point" not in all_messages
    ):
        return FailureReport(
            FailureType.MISSING_ENTRYPOINT,
            "The required function name is missing from the generated code.",
            hint="Make sure the function name exactly matches the entry_point in the task.",
            raw_errors=errors,
        )

    # ── 4. Import / module errors (old API at runtime) ────────────────
    if any(kw in all_messages for kw in
           ["ModuleNotFoundError", "ImportError", "cannot import name"]):
        return FailureReport(
            FailureType.OUTDATED_API,
            "Import error — likely using a module removed in Qiskit 2.x.",
            hint=(
                "Do not import from qiskit.extensions, qiskit.providers.aer, "
                "or any deprecated sub-package. "
                "Check that all gate names match the Qiskit 2.x Gate API."
            ),
            raw_errors=errors,
        )

    # ── 5. Runtime error (circuit crashes during execution) ───────────
    if errors:
        first = errors[0]
        if any(kw in first for kw in
               ["CircuitError", "QiskitError", "TypeError", "AttributeError",
                "ValueError", "IndexError"]):
            return FailureReport(
                FailureType.RUNTIME_ERROR,
                f"Circuit raised a runtime error: {first[:300]}",
                hint=(
                    "Check qubit/clbit indices, gate argument counts, "
                    "and that measurement registers match circuit size."
                ),
                raw_errors=errors,
            )
        return FailureReport(
            FailureType.RUNTIME_ERROR,
            f"Unexpected runtime error: {first[:300]}",
            hint="Debug the stack trace above.",
            raw_errors=errors,
        )

    # ── 6. Test assertion failures → distinguish structure vs logic ────
    if failures:
        first = failures[0]

        # KL > threshold → distribution is wrong (algorithm logic)
        if "KL" in first or "AssertionError" in first:
            if "most_common" in first or "assertEqual" in first:
                return FailureReport(
                    FailureType.ALGORITHM_LOGIC,
                    "Circuit produces wrong measurement outcome distribution.",
                    hint=(
                        "The circuit runs but the output state is wrong. "
                        "Re-check the oracle construction, diffusion operator, "
                        "or state preparation logic."
                    ),
                    raw_failures=failures,
                )
            return FailureReport(
                FailureType.CIRCUIT_STRUCTURE,
                "Circuit's output distribution differs significantly from canonical (high KL divergence).",
                hint=(
                    "The circuit topology may be correct but gate order or "
                    "parameter values differ. Compare against the expected unitary."
                ),
                raw_failures=failures,
            )

        if "phase" in first.lower():
            return FailureReport(
                FailureType.ALGORITHM_LOGIC,
                "Incorrect relative phase in the generated state.",
                hint=(
                    "Pay careful attention to phase kickback and Z/S/T gate placement. "
                    "The amplitude magnitude may be correct but the phase is wrong."
                ),
                raw_failures=failures,
            )

    # ── 7. Fallback ───────────────────────────────────────────────────
    return FailureReport(
        FailureType.UNKNOWN,
        "Test failed for an unclassified reason.",
        hint="Review the full error messages and compare generated vs canonical circuits.",
        raw_errors=errors,
        raw_failures=failures,
    )
