"""
QuanBench Live Demo
====================
Streamlit app: natural language -> Qiskit circuit -> visualization + evaluation.

Run locally:
    streamlit run app.py
"""

import os
import io
import re
import json
import time
import traceback
from datetime import datetime, timedelta

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# ── API key — read from secrets OR env, never at module load ──────────────────
def get_api_key() -> str:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.pass-badge {
    background:#4ade8022; color:#4ade80; border:1px solid #4ade8044;
    border-radius:999px; padding:3px 14px; font-size:12px; font-weight:700; letter-spacing:1px;
}
.fail-badge {
    background:#f8717122; color:#f87171; border:1px solid #f8717144;
    border-radius:999px; padding:3px 14px; font-size:12px; font-weight:700; letter-spacing:1px;
}
.info-box {
    background:#22d3ee0a; border:1px solid #22d3ee25;
    border-radius:10px; padding:14px 18px; margin:10px 0;
}
.warn-box {
    background:#fbbf2408; border:1px solid #fbbf2430;
    border-radius:10px; padding:14px 18px; margin:10px 0;
}
.stTextArea textarea {
    font-family:'JetBrains Mono', monospace !important; font-size:13px !important;
    background:#080d1a !important; border:1px solid #162035 !important; color:#cbd5e1 !important;
}
.stButton > button {
    background:linear-gradient(135deg, #22d3ee, #a78bfa); color:#04070f;
    font-family:'JetBrains Mono', monospace; font-weight:700; font-size:13px;
    letter-spacing:1px; border:none; border-radius:8px; padding:10px 28px; width:100%;
}
</style>
""", unsafe_allow_html=True)

# ── rate limiter — cross-session via cache_resource ──────────────────────────
RATE_LIMIT_MAX  = 20
RATE_LIMIT_SECS = 3600

@st.cache_resource
def _get_global_rate_store():
    """Shared across all sessions on this Streamlit instance."""
    return {"log": []}

def _check_rate_limit():
    store = _get_global_rate_store()
    now = datetime.utcnow()
    cutoff = now - timedelta(seconds=RATE_LIMIT_SECS)
    store["log"] = [t for t in store["log"] if t > cutoff]
    remaining = RATE_LIMIT_MAX - len(store["log"])
    return remaining > 0, remaining

def _record_run():
    _get_global_rate_store()["log"].append(datetime.utcnow())

# ── Qiskit helpers ─────────────────────────────────────────────────────────────
SHOTS = 4096

def run_circuit(qc: QuantumCircuit, shots: int = SHOTS) -> dict:
    sim = AerSimulator()
    t = transpile(qc, sim)
    return sim.run(t, shots=shots).result().get_counts()

def circuit_png(qc: QuantumCircuit) -> bytes:
    fig = None
    try:
        fig = qc.draw("mpl", style="bw", fold=40)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor="#080d1a")
        buf.seek(0)
        return buf.read()
    finally:
        plt.close("all")

def histogram_png(counts: dict, title: str = "") -> bytes:
    if not counts:
        # Return a minimal placeholder image
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("#080d1a")
        ax.set_facecolor("#0c1424")
        ax.text(0.5, 0.5, "No measurement results", ha="center", va="center",
                color="#475569", fontsize=12, transform=ax.transAxes)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor="#080d1a")
        plt.close("all")
        buf.seek(0)
        return buf.read()
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])[:16]
    states = [s.replace(" ", "") for s, _ in sorted_items]
    vals   = [v for _, v in sorted_items]
    total  = sum(vals)
    probs  = [v / total for v in vals]

    fig, ax = plt.subplots(figsize=(max(6, len(states) * 0.7), 3.5))
    fig.patch.set_facecolor("#080d1a")
    ax.set_facecolor("#0c1424")

    colors = ["#22d3ee" if p == max(probs) else "#1e3050" for p in probs]
    ax.bar(states, probs, color=colors, edgecolor="#162035", linewidth=0.5, zorder=3)

    max_idx = probs.index(max(probs))
    ax.text(max_idx, probs[max_idx] + 0.01, f"{probs[max_idx]:.1%}",
            ha="center", va="bottom", color="#22d3ee", fontsize=9,
            fontweight="bold", fontfamily="monospace")

    ax.set_xlabel("Measurement outcome", color="#475569", fontsize=10)
    ax.set_ylabel("Probability",         color="#475569", fontsize=10)
    ax.set_title(title, color="#cbd5e1", fontsize=11, pad=10, fontfamily="monospace")
    ax.tick_params(colors="#475569", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#162035")
    ax.grid(axis="y", color="#162035", linewidth=0.5, zorder=0)
    plt.xticks(rotation=45, ha="right")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor="#080d1a")
    plt.close("all")
    buf.seek(0)
    return buf.read()

# ── LLM helpers ───────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-20250514"

QISKIT_RULES = """
QISKIT 2.x RULES (strictly enforce):
- from qiskit import QuantumCircuit, transpile
- from qiskit_aer import AerSimulator
- DO NOT use: qiskit.execute(), Aer.get_backend(), QuantumInstance, qiskit.opflow
- Function MUST return a QuantumCircuit with measurements applied (qc.measure_all())
- No print statements, no plt.show(), no if __name__ blocks
"""

def _llm(prompt: str, max_tokens: int = 1800) -> str:
    api_key = get_api_key()
    client = anthropic.Anthropic(api_key=api_key, timeout=45.0)
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except anthropic.RateLimitError:
        raise RuntimeError("Anthropic API rate limit reached. Please wait a minute and try again.")
    except anthropic.APITimeoutError:
        raise RuntimeError("Request timed out. The model is busy — please try again.")
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Anthropic API error ({e.status_code}): {e.message}")
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

def _extract_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    return text.strip()

def _extract_json(text: str) -> dict:
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    raw = m.group(1).strip() if m else text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(
            "LLM returned malformed JSON. Try rephrasing your prompt."
        )

def formulate_task(nl: str) -> dict:
    prompt = f"""{QISKIT_RULES}

Convert this natural language description into a structured quantum task.

User request: {nl}

Respond with JSON inside ```json ... ```:
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

Return ONLY the function inside ```python ... ```.
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

Error: {error[:500]}

Return ONLY fixed code inside ```python ... ```.
"""
    return _extract_code(_llm(prompt))

# Imports that generated code should never use
_BLOCKED_IMPORTS = ["os", "sys", "subprocess", "socket", "shutil",
                    "pathlib", "glob", "requests", "urllib", "http",
                    "importlib", "builtins", "eval", "exec"]

def _check_safe_code(code: str) -> str | None:
    """Returns an error string if code contains dangerous patterns, else None."""
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return None  # let exec() catch it with a better message
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, _ast.Import) else [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                if root in _BLOCKED_IMPORTS:
                    return f"Blocked import: '{name}' is not permitted in generated code."
    return None

def try_execute(code: str, entry_point: str):
    """Returns (qc, error_str). qc is None on failure."""
    safety_err = _check_safe_code(code)
    if safety_err:
        return None, safety_err
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
            return None, "Circuit has no measurements — needs qc.measure_all()."
        return qc, ""
    except Exception:
        return None, traceback.format_exc(limit=4)

# ── run the full pipeline, return a plain-dict result (no QC objects) ─────────
def run_pipeline(nl: str, max_iter: int) -> dict:
    """
    Runs the full generate→execute→repair loop.
    Returns a serializable dict — NO QuantumCircuit objects stored.
    PNG images are stored as bytes.
    """
    result = {
        "nl": nl, "task": None, "code": "", "passed": False,
        "iterations": 0, "error": "", "circuit_png": None,
        "hist_png": None, "gate_counts": {}, "depth": 0,
        "num_qubits": 0, "counts": None, "top_state": "",
        "top_prob": 0.0, "num_clbits": 0, "steps": [],
    }

    def log(msg):
        result["steps"].append(msg)

    try:
        log("Step 1 · Formulating task structure...")
        task = formulate_task(nl)
        result["task"] = task
        log(f"→ {task['entry_point']}()  type: {task['algorithm_type']}")

        code = ""
        qc   = None
        err  = ""

        for iteration in range(1, max_iter + 1):
            result["iterations"] = iteration

            if iteration == 1:
                log(f"Step 2 · Generating Qiskit code...")
                code = generate_code(task)
            else:
                log(f"Step {iteration + 1} · Repairing (iter {iteration})...")
                code = repair_code(task, code, err, iteration)

            result["code"] = code
            qc, err = try_execute(code, task["entry_point"])

            if qc is not None:
                result["passed"] = True
                log(f"✅ Circuit executes ({qc.num_qubits} qubits, depth {qc.depth()})")
                break
            else:
                log(f"❌ {err[:120].strip()}")

        if not result["passed"]:
            result["error"] = err
            return result

        # Simulate
        log("Simulating on Qiskit-Aer...")
        counts = run_circuit(qc)
        result["counts"]     = counts
        result["gate_counts"] = dict(qc.count_ops())
        result["depth"]      = qc.depth()
        result["num_qubits"] = qc.num_qubits
        result["num_clbits"] = qc.num_clbits

        top_state, top_count = max(counts.items(), key=lambda x: x[1])
        result["top_state"] = top_state.replace(" ", "")
        result["top_prob"]  = round(top_count / sum(counts.values()), 4)

        # Render images
        log("Rendering diagrams...")
        result["circuit_png"] = circuit_png(qc)
        result["hist_png"]    = histogram_png(
            counts, title=f"{task['entry_point']}() · {SHOTS} shots"
        )
        log("Done ✅")

    except Exception:
        result["error"] = traceback.format_exc(limit=6)

    return result

# ── session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("result",  None),
    ("history", []),
    ("running", False),
    ("prefill", ""),
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
            st.session_state.prefill = ex
            st.session_state.nl_input_box = ex
            st.rerun()

    st.divider()
    _, remaining = _check_rate_limit()
    st.markdown(f"**Rate limit:** {remaining}/{RATE_LIMIT_MAX} runs/hour")

    st.divider()
    st.markdown(
        "Built on [QuanBench-44](https://arxiv.org/abs/2510.16779) · "
        "Qiskit 2.x + claude-sonnet-4"
    )

# ── main UI ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:4px'>Quantum Circuit Generator</h1>"
    "<p style='color:#475569;font-size:13px;margin-top:0'>"
    "Type any quantum algorithm in plain English. "
    "The agent generates, runs, and visualizes the Qiskit circuit.</p>",
    unsafe_allow_html=True,
)

# Capture input — use session state key so value survives reruns
nl_input = st.text_area(
    "prompt",
    height=80,
    placeholder='e.g. "Grover search targeting |101> on a 3-qubit register"',
    label_visibility="collapsed",
    key="nl_input_box",
)

col_btn, col_opts = st.columns([1, 2])
with col_btn:
    run_btn = st.button("⚛  Generate Circuit", disabled=st.session_state.running)
with col_opts:
    max_iter = st.slider("Max repair iterations", 1, 5, 3)

# ── RUN ────────────────────────────────────────────────────────────────────────
if run_btn:
    prompt_text = st.session_state.get("nl_input_box", "").strip()

    if not prompt_text:
        st.warning("Please enter a circuit description first.")
        st.stop()

    allowed, _ = _check_rate_limit()
    if not allowed:
        st.error("Rate limit reached. Try again in an hour.")
        st.stop()

    api_key = get_api_key()
    if not api_key:
        st.error("ANTHROPIC_API_KEY not set. Add it to Streamlit Cloud secrets.")
        st.stop()

    _record_run()
    # Clear prefill so next run starts clean
    st.session_state.prefill = ""
    st.session_state.running = True
    st.session_state.result  = None

    with st.status("Running agent...", expanded=True) as status:
        t0 = time.time()
        result = run_pipeline(prompt_text, max_iter)
        elapsed = time.time() - t0

        for step_msg in result["steps"]:
            st.write(step_msg)

        if result["passed"]:
            status.update(
                label=f"✅ Done in {elapsed:.1f}s — {result['iterations']} iteration(s)",
                state="complete",
            )
        else:
            status.update(label="Agent finished — circuit could not be verified", state="error")

    st.session_state.result  = result
    st.session_state.running = False

    # Add to history
    st.session_state.history.append({
        "nl":       prompt_text,
        "passed":   result["passed"],
        "iter":     result["iterations"],
        "entry_pt": result["task"]["entry_point"] if result["task"] else "—",
        "ts":       datetime.utcnow().strftime("%H:%M:%S"),
    })
    # Cap history to prevent unbounded memory growth
    st.session_state.history = st.session_state.history[-50:]

# ── RESULTS ────────────────────────────────────────────────────────────────────
r = st.session_state.result
if r:
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    badge = '<span class="pass-badge">✓ PASS</span>' if r["passed"] else '<span class="fail-badge">✗ FAIL</span>'
    col1.markdown(f"**Result** {badge}", unsafe_allow_html=True)
    col2.metric("Iterations", r["iterations"])
    if r["passed"]:
        col3.metric("Qubits",        r["num_qubits"])
        col4.metric("Circuit depth", r["depth"])

    st.divider()

    if r["passed"]:
        tab_circ, tab_hist, tab_code, tab_stats = st.tabs([
            "⚛ Circuit Diagram", "📊 Measurement Histogram",
            "📄 Generated Code",  "📈 Gate Statistics",
        ])

        with tab_circ:
            entry = r["task"]["entry_point"] if r["task"] else ""
            st.markdown(
                f"<div class='info-box'>Circuit for <code>{entry}()</code> · "
                f"{r['num_qubits']} qubits · depth {r['depth']}</div>",
                unsafe_allow_html=True,
            )
            st.image(r["circuit_png"], use_container_width=True)

        with tab_hist:
            st.markdown(
                f"<div class='info-box'>Most probable state: "
                f"<code>|{r['top_state']}⟩</code> · "
                f"probability <b>{r['top_prob']:.1%}</b> · "
                f"{SHOTS} shots · {len(r['counts'])} unique outcomes</div>",
                unsafe_allow_html=True,
            )
            st.image(r["hist_png"], use_container_width=True)

        with tab_code:
            st.markdown(
                "<div class='info-box'>Generated by claude-sonnet-4 · "
                "verified by Qiskit-Aer simulator</div>",
                unsafe_allow_html=True,
            )
            st.code(r["code"], language="python")

        with tab_stats:
            gc = r["gate_counts"]
            if gc:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Gate counts**")
                    max_count = max(gc.values())
                    for gate, count in sorted(gc.items(), key=lambda x: -x[1]):
                        st.progress(count / max_count, text=f"`{gate}` × {count}")
                with col_b:
                    st.markdown("**Circuit summary**")
                    st.metric("Total gates",     sum(v for k, v in gc.items() if k != "barrier"))
                    st.metric("Circuit depth",   r["depth"])
                    st.metric("Qubits",          r["num_qubits"])
                    st.metric("Classical bits",  r["num_clbits"])
                    st.metric("Unique outcomes", len(r["counts"]))
                    st.metric("Top state",       f"|{r['top_state']}⟩ ({r['top_prob']:.1%})")
    else:
        st.error("The agent could not produce a valid circuit within the iteration limit.")
        if r["error"]:
            with st.expander("Error details"):
                st.code(r["error"])
        if r["code"]:
            with st.expander("Last generated code"):
                st.code(r["code"], language="python")
        st.markdown(
            "<div class='warn-box'>💡 Try rephrasing the prompt more specifically, "
            "or increase the max repair iterations.</div>",
            unsafe_allow_html=True,
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
