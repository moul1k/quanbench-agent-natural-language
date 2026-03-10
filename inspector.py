"""
Circuit Inspector
==================
Extracts structured, human-readable facts about a QuantumCircuit and
computes unitary fidelity between two circuits.

Used by the Repair Agent to give the LLM concrete structural context
rather than just a raw error message.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector


MAX_UNITARY_QUBITS = 8   # unitary diff is O(4^n) — skip for large circuits


@dataclass
class CircuitFacts:
    num_qubits: int
    num_clbits: int
    depth: int
    gate_counts: dict
    total_gates: int
    has_measurements: bool
    ascii_diagram: str
    unitary_fidelity: Optional[float] = None   # vs canonical, if computable
    unitary_diff_hint: str = ""                # human-readable diff


def inspect(qc: QuantumCircuit) -> CircuitFacts:
    """Extract structured facts from a single circuit."""
    gate_counts = dict(qc.count_ops())
    has_meas = "measure" in gate_counts
    total = sum(v for k, v in gate_counts.items() if k != "barrier")

    # ASCII diagram (truncated if huge)
    try:
        diagram = str(qc.draw("text"))
        if len(diagram) > 2000:
            diagram = diagram[:2000] + "\n... (truncated)"
    except Exception:
        diagram = "(could not render)"

    return CircuitFacts(
        num_qubits=qc.num_qubits,
        num_clbits=qc.num_clbits,
        depth=qc.depth(),
        gate_counts=gate_counts,
        total_gates=total,
        has_measurements=has_meas,
        ascii_diagram=diagram,
    )


def compare(canonical: QuantumCircuit, generated: QuantumCircuit) -> CircuitFacts:
    """
    Inspect the generated circuit and compute unitary fidelity vs canonical.
    Returns a CircuitFacts with unitary comparison filled in where possible.
    """
    facts = inspect(generated)

    # Remove measurements for unitary comparison
    try:
        qc_ref = canonical.remove_final_measurements(inplace=False)
        qc_gen = generated.remove_final_measurements(inplace=False)
    except Exception:
        facts.unitary_diff_hint = "Could not remove measurements for unitary comparison."
        return facts

    n = qc_ref.num_qubits
    if n != qc_gen.num_qubits:
        facts.unitary_diff_hint = (
            f"Qubit count mismatch: canonical has {n} qubits, "
            f"generated has {qc_gen.num_qubits}."
        )
        return facts

    if n > MAX_UNITARY_QUBITS:
        facts.unitary_diff_hint = (
            f"Circuit too large ({n} qubits) for unitary comparison. "
            "Use KL divergence only."
        )
        return facts

    try:
        U_ref = Operator(qc_ref).data
        U_gen = Operator(qc_gen).data
        dim = 2 ** n
        # Frobenius-based process fidelity: |Tr(U_ref† U_gen)| / dim
        fidelity = float(abs(np.trace(U_ref.conj().T @ U_gen)) / dim)
        facts.unitary_fidelity = round(fidelity, 4)

        if fidelity > 0.99:
            hint = "Unitary fidelity is near-perfect (>0.99). Failure is likely in measurement setup or classical register."
        elif fidelity > 0.75:
            hint = (
                f"Unitary fidelity is {fidelity:.3f} — circuits are partially correct. "
                "One or more gates are placed incorrectly or have wrong parameters."
            )
        elif fidelity > 0.3:
            hint = (
                f"Unitary fidelity is {fidelity:.3f} — significant structural mismatch. "
                "Key gates (oracle, diffusion, phase kickback) are likely wrong."
            )
        else:
            hint = (
                f"Unitary fidelity is {fidelity:.3f} — circuits implement fundamentally different operations. "
                "Reconsider the algorithm structure from scratch."
            )

        facts.unitary_diff_hint = hint

    except Exception as e:
        facts.unitary_diff_hint = f"Unitary comparison failed: {e}"

    return facts


def format_for_prompt(canonical_facts: CircuitFacts,
                       generated_facts: CircuitFacts) -> str:
    """Format circuit comparison as a compact prompt block."""
    lines = ["## Circuit Inspection"]

    lines.append("\n### Canonical circuit")
    lines.append(f"- Qubits: {canonical_facts.num_qubits}, Depth: {canonical_facts.depth}")
    lines.append(f"- Gates: {canonical_facts.gate_counts}")
    lines.append(f"- Diagram:\n```\n{canonical_facts.ascii_diagram}\n```")

    lines.append("\n### Generated circuit")
    lines.append(f"- Qubits: {generated_facts.num_qubits}, Depth: {generated_facts.depth}")
    lines.append(f"- Gates: {generated_facts.gate_counts}")
    lines.append(f"- Has measurements: {generated_facts.has_measurements}")
    if generated_facts.unitary_fidelity is not None:
        lines.append(f"- Unitary fidelity vs canonical: {generated_facts.unitary_fidelity}")
    if generated_facts.unitary_diff_hint:
        lines.append(f"- Diagnosis: {generated_facts.unitary_diff_hint}")
    lines.append(f"- Diagram:\n```\n{generated_facts.ascii_diagram}\n```")

    return "\n".join(lines)
