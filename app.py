"""
QuanBench Live Demo
====================
Streamlit app: natural language → Qiskit circuit → visualization + evaluation.

Run locally:
    streamlit run app.py

Deploy on Streamlit Cloud:
    Push to GitHub, connect repo at share.streamlit.io
"""

import os
import io
import time
import textwrap
import traceback
from datetime import datetime, timedelta
from collections import defaultdict

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import anthropic

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuanBench · Quantum Circuit Generator",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
}
.main { background: #04070f; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.metric-card {
    background: #0c1424;
    border: 1px solid #162035;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.pass-badge {
    background: #4ade8022;
    color: #4ade80;
    border: 1px solid #4ade8044;
    border-radius: 999px;
    padding: 3px 14px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}
.fail-badge {
    background: #f8717122;
    color: #f87171;
    border: 1px solid #f8717144;
    border-radius: 999px;
    padding: 3px 14px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}
.info-box {
    background: #22d3ee0a;
    border: 1px solid #22d3ee25;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
}
.warn-box {
    background: #fbbf2408;
    border: 1px solid #fbbf2430;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
}
.stTextArea textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    background: #080d1a !important;
    border: 1px solid #162035 !important;
    color: #cbd5e1 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #22d3ee, #a78bfa);
    color: #04070f;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 1px;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}
code {
    background: #0c1424 !important;
    color: #22d3ee !important;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── rate limiter ───────────────────────────────────────────────────────────────
# Simple in-memory rate limiter: max N runs per hour globally
RATE_LIMIT_MAX  = 20       # max runs per hour globally
RATE_LIMIT_SECS = 3600     # 1 hour window

if "rate_log" not in st.session_state:
    st.session_state.rate_log = []   # list of timestamps

def _check_rate_limit() -> tuple[bool, int]:
    """Returns (allowed, remaining_runs)."""
    now = datetime.utcnow()
    cutoff = now - timedelta(seconds=RATE_LIMIT_SECS)
    st.session_state.rate_log = [
        t for t in st.session_state.rate_log if t > cutoff
    ]
    used = len(st.session_state.rate_log)
    remaining = RATE_LIMIT_MAX - used
    return remaining > 0, remaining

def _record_run():
    st.session_state.rate_log.append(datetime.utcnow())


# ── Qiskit helpers ─────────────────────────────────────────────────────────────
SIM = AerSimulator()
SHOTS = 4096

def run_circuit(qc: QuantumCircuit, shots: int = SHOTS) -> dict:
    t = transpile(qc, SIM)
    return SIM.run(t, shots=shots).result().get_counts()

def circuit_png(qc: QuantumCircuit) -> bytes:
    fig = qc.draw("mpl", style="bw", fold=40)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130,
                facecolor="#080d1a")
    plt.close("all")
    buf.seek(0)
    return buf.read()

def histogram_png(counts: dict, title: str = "") -> bytes:
    # Sort by state, take top 16 for readability
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])[:16]
    states = [s.replace(" ", "") for s, _ in sorted_items]
    vals   = [v for _, v in sorted_items]
    total  = sum(vals)
    probs  = [v / total for v in vals]

    fig, ax = plt.subplots(figsize=(max(6, len(states) * 0.7), 3.5))
    fig.patch.set_facecolor("#080d1a")
    ax.set_facecolor("#0c1424")

    colors = ["#22d3ee" if p == max(probs) else "#1e3050" for p in probs]
    bars = ax.bar(states, probs, color=colors, edgecolor="#162035",
                  linewidth=0.5, zorder=3)

    # Label top bar
    max_idx = probs.index(max(probs))
    ax.text(max_idx, probs[max_idx] + 0.01,
            f"{probs[max_idx]:.1%}", ha="center", va="bottom",
            color="#22d3ee", fontsize=9, fontweight="bold",
            fontfamily="monospace")

    ax.set_xlabel("Measurement outcome", color="#475569", fontsize=10)
    ax.set_ylabel("Probability", color="#475569", fontsize=10)
    ax.set_title(title, color="#cbd5e1", fontsize=11, pad=10,
                 fontfamily="monospace")
    ax.tick_params(colors="#475569", labelsize=8)
    ax.spines[:].set_color("#162035")
    ax.grid(axis="y", color="#162035", linewidth=0.5, zorder=0)
    plt.xticks(rotation=45, ha="right")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130,
                facecolor="#080d1a")
    plt.close("all")
    buf.seek(0)
    return buf.read()


# ── LLM agent (minimal inline version) ────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
MODEL   = "claude-sonnet-4-20250514"

QISKIT_RULES = """
QISKIT 2.x RULES (strictly enforce):
- from qiskit import QuantumCircuit, transpile
- from qiskit_aer import AerSimulator
- DO NOT use: qiskit.execute(), Aer.get_backend(), QuantumInstance, qiskit.opflow
- Function MUST return a QuantumCircuit with measurements applied
- No print statements, no plt.show(), no if __name__ blocks
"""

def _llm(prompt: str, max_tokens: int = 1800) -> str:
    client = anthropic.Anthropic(api_key=API_KEY)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

def _extract_code(text: str) -> str:
    import re
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    return text.strip()

def _extract_json(text: str) -> dict:
    import re, json
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    raw = m.group(1).strip() if m else text.strip()
    return json.loads(raw)

def formulate_task(nl: str) -> dict:
    prompt = f"""{QISKIT_RULES}

Convert this natural language description into a structured quantum task.

User request: {nl}

Respond with JSON in ```json ... ```:
{{
  "entry_point": "snake_case_function_name",
  "complete_prompt": "from qiskit import QuantumCircuit\\ndef function_name() -> QuantumCircuit:\\n    \\\"\\\"\\\"docstring\\\"\\\"\\\"",
  "algorithm_type": "grover|qft|qpe|bell|ghz|vqc|other",
  "description": "one sentence description"
}}
"""
    return _extract_json(_llm(prompt, 600))

def generate_code(task: dict) -> str:
    prompt = f"""{QISKIT_RULES}

Implement this quantum circuit function using Qiskit 2.x.

{task['complete_prompt']}

Return ONLY the function in ```python ... ```.
"""
    return _extract_code(_llm(prompt))

def repair_code(task: dict, code: str, error: str, iteration: int) -> str:
    prompt = f"""{QISKIT_RULES}

Fix this quantum circuit (attempt {iteration}).

Task: {task['complete_prompt']}

Failed code:
```python
{code}
```

Error/failure: {error[:500]}

Return ONLY fixed code in ```python ... ```.
"""
    return _extract_code(_llm(prompt))

def try_execute(code: str, entry_point: str) -> tuple[QuantumCircuit | None, str]:
    """Try to exec code and return (circuit, error_message)."""
    ns = {"QuantumCircuit": QuantumCircuit}
    try:
        exec(compile(code, "<gen>", "exec"), ns)
        fn = ns.get(entry_point)
        if fn is None:
            return None, f"Function '{entry_point}' not found in generated code."
        qc = fn()
        if not isinstance(qc, QuantumCircuit):
            return None, "Function did not return a QuantumCircuit."
        if "measure" not in dict(qc.count_ops()):
            return None, "Circuit has no measurements — add qc.measure_all()."
        return qc, ""
    except Exception as e:
        return None, traceback.format_exc(limit=4)


# ── session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("result", None),
    ("history", []),
    ("running", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚛ QuanBench")
    st.markdown("*Natural language → verified Qiskit circuit*")
    st.divider()

    st.markdown("**How it works**")
    for step, desc in [
        ("1 · Formulate", "NL → structured task + entry point"),
        ("2 · Generate",  "LLM writes Qiskit 2.x code"),
        ("3 · Execute",   "Qiskit-Aer simulator runs circuit"),
        ("4 · Repair",    "Failure classifier + repair loop"),
        ("5 · Visualize", "Circuit diagram + measurement histogram"),
    ]:
        st.markdown(f"**`{step}`** {desc}")

    st.divider()
    st.markdown("**Try these prompts**")
    examples = [
        "Bell state on 2 qubits",
        "GHZ state on 4 qubits",
        "QFT on 3 qubits",
        "Grover search for |101> on 3 qubits",
        "Quantum teleportation circuit",
        "3-qubit phase estimation",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state["prefill"] = ex

    st.divider()
    allowed, remaining = _check_rate_limit()
    st.markdown(f"**Rate limit:** {remaining}/{RATE_LIMIT_MAX} runs remaining this hour")
    if not allowed:
        st.error("Rate limit reached. Try again in an hour.")

    st.divider()
    st.markdown(
        "Built on [QuanBench-44](https://arxiv.org/abs/2510.16779) · "
        "[GitHub](https://github.com) · "
        "Qiskit 2.x + claude-sonnet-4"
    )


# ── main ───────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:4px'>Quantum Circuit Generator</h1>"
    "<p style='color:#475569;font-size:13px;margin-top:0'>Type any quantum algorithm in plain English. "
    "The agent generates, runs, and visualizes the Qiskit circuit.</p>",
    unsafe_allow_html=True,
)

# Input box — pick up prefill from sidebar buttons
prefill = st.session_state.pop("prefill", "")
nl_input = st.text_area(
    "Describe your quantum circuit",
    value=prefill,
    height=80,
    placeholder='e.g. "Implement Grover search targeting |101> on a 3-qubit register"',
    label_visibility="collapsed",
)

col_btn, col_opts = st.columns([1, 2])
with col_btn:
    run_btn = st.button("⚛  Generate Circuit", disabled=st.session_state.running)
with col_opts:
    max_iter = st.slider("Max repair iterations", 1, 5, 3, label_visibility="visible")


# ── RUN ────────────────────────────────────────────────────────────────────────
if run_btn and nl_input.strip():
    allowed, _ = _check_rate_limit()
    if not allowed:
        st.error("Rate limit reached. Try again in an hour.")
        st.stop()

    if not API_KEY:
        st.error("ANTHROPIC_API_KEY not set. Add it to your .env or Streamlit secrets.")
        st.stop()

    _record_run()
    st.session_state.running = True

    with st.status("Running agent...", expanded=True) as status:
        t0 = time.time()
        result = {
            "nl": nl_input.strip(),
            "task": None,
            "code": "",
            "qc": None,
            "counts": None,
            "passed": False,
            "iterations": 0,
            "error": "",
            "circuit_png": None,
            "hist_png": None,
            "gate_counts": {},
            "depth": 0,
        }

        try:
            # Step 1: formulate
            st.write("**Step 1** · Formulating task structure...")
            task = formulate_task(nl_input.strip())
            result["task"] = task
            st.write(f"  → `{task['entry_point']}()` · type: `{task['algorithm_type']}`")

            # Step 2+: generate → execute → repair loop
            code = ""
            qc   = None
            err  = ""

            for iteration in range(1, max_iter + 1):
                result["iterations"] = iteration

                if iteration == 1:
                    st.write(f"**Step 2** · Generating Qiskit code...")
                    code = generate_code(task)
                else:
                    st.write(f"**Step {iteration+1}** · Repairing (iter {iteration})...")
                    code = repair_code(task, code, err, iteration)

                result["code"] = code
                qc, err = try_execute(code, task["entry_point"])

                if qc is not None:
                    result["passed"] = True
                    result["qc"] = qc
                    st.write(f"  → ✅ Circuit executes ({qc.num_qubits} qubits, depth {qc.depth()})")
                    break
                else:
                    st.write(f"  → ❌ {err[:120].strip()}...")

            if not result["passed"]:
                result["error"] = err
                status.update(label="Agent finished — circuit could not be verified", state="error")
            else:
                # Step: simulate
                st.write("**Simulating** · Running on Qiskit-Aer...")
                counts = run_circuit(qc)
                result["counts"]     = counts
                result["gate_counts"] = dict(qc.count_ops())
                result["depth"]      = qc.depth()

                st.write("**Rendering** · Generating diagrams...")
                result["circuit_png"] = circuit_png(qc)
                result["hist_png"]    = histogram_png(
                    counts,
                    title=f"{task['entry_point']}() · {SHOTS} shots"
                )
                elapsed = time.time() - t0
                status.update(
                    label=f"✅ Done in {elapsed:.1f}s — {iteration} iteration(s)",
                    state="complete"
                )

        except Exception as e:
            result["error"] = traceback.format_exc(limit=6)
            status.update(label=f"Error: {e}", state="error")

    st.session_state.result   = result
    st.session_state.running  = False
    st.session_state.history.append({
        "nl":       result["nl"],
        "passed":   result["passed"],
        "iter":     result["iterations"],
        "entry_pt": result["task"]["entry_point"] if result["task"] else "—",
        "ts":       datetime.utcnow().strftime("%H:%M:%S"),
    })


# ── RESULTS ────────────────────────────────────────────────────────────────────
r = st.session_state.result
if r:
    st.divider()

    # Header row
    col1, col2, col3, col4 = st.columns(4)
    badge = '<span class="pass-badge">✓ PASS</span>' if r["passed"] else '<span class="fail-badge">✗ FAIL</span>'
    col1.markdown(f"**Result** {badge}", unsafe_allow_html=True)
    col2.metric("Iterations", r["iterations"])
    if r["passed"]:
        col3.metric("Qubits", r["qc"].num_qubits)
        col4.metric("Circuit depth", r["depth"])

    st.divider()

    if r["passed"]:
        tab_circ, tab_hist, tab_code, tab_stats = st.tabs([
            "⚛ Circuit Diagram", "📊 Measurement Histogram",
            "📄 Generated Code",  "📈 Gate Statistics"
        ])

        with tab_circ:
            st.markdown(
                f"<div class='info-box'>Circuit for <code>{r['task']['entry_point']}()</code> · "
                f"{r['qc'].num_qubits} qubits · depth {r['depth']}</div>",
                unsafe_allow_html=True
            )
            st.image(r["circuit_png"], use_container_width=True)

        with tab_hist:
            total = sum(r["counts"].values())
            top_state, top_count = max(r["counts"].items(), key=lambda x: x[1])
            top_prob = top_count / total
            st.markdown(
                f"<div class='info-box'>Most probable state: "
                f"<code>|{top_state.replace(' ','')}⟩</code> "
                f"with probability <b>{top_prob:.1%}</b> "
                f"({SHOTS} shots · {len(r['counts'])} unique outcomes)</div>",
                unsafe_allow_html=True
            )
            st.image(r["hist_png"], use_container_width=True)

        with tab_code:
            st.markdown(
                f"<div class='info-box'>Generated by claude-sonnet-4 · "
                f"verified by Qiskit-Aer simulator</div>",
                unsafe_allow_html=True
            )
            st.code(r["code"], language="python")

        with tab_stats:
            gc = r["gate_counts"]
            if gc:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Gate counts**")
                    for gate, count in sorted(gc.items(), key=lambda x: -x[1]):
                        pct = count / max(gc.values())
                        st.progress(pct, text=f"`{gate}` × {count}")
                with col_b:
                    st.markdown("**Circuit summary**")
                    st.metric("Total gates",   sum(v for k,v in gc.items() if k != "barrier"))
                    st.metric("Circuit depth", r["depth"])
                    st.metric("Qubits",        r["qc"].num_qubits)
                    st.metric("Classical bits",r["qc"].num_clbits)
                    st.metric("Unique outcomes", len(r["counts"]))
                    top_s = top_state.replace(" ","")
                    st.metric("Top state", f"|{top_s}⟩ ({top_prob:.1%})")

    else:
        st.error("The agent could not produce a valid circuit within the iteration limit.")
        if r["error"]:
            with st.expander("Error details"):
                st.code(r["error"])
        if r["code"]:
            with st.expander("Last generated code"):
                st.code(r["code"], language="python")
        st.markdown(
            "<div class='warn-box'>💡 Try rephrasing the prompt to be more specific, "
            "or increase the max repair iterations.</div>",
            unsafe_allow_html=True
        )


# ── HISTORY ────────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    st.markdown("### Session history")
    for h in reversed(st.session_state.history[-8:]):
        badge = "✅" if h["passed"] else "❌"
        st.markdown(
            f"`{h['ts']}` {badge} **{h['entry_pt']}** · "
            f"{h['iter']} iter · _{h['nl'][:60]}_"
        )
