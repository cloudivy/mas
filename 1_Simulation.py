import time
import streamlit as st
from core.agents import execute_node, NODES, TASK_SPEC
from core.metrics import compute_asi
from core.embeddings import get_embedding

TOTAL_STEPS = 16

st.header("🔬 LangGraph Simulation — Research Synthesis")
st.caption(
    f"4-node directed cyclic graph · {TOTAL_STEPS} interaction steps · "
    "ASI measured at every step"
)

with st.expander("📋 Task Specification", expanded=False):
    st.code(TASK_SPEC, language=None)

# ── Guard ─────────────────────────────────────────────────────────────────────
if not st.session_state.api_key:
    st.warning("⚠️ Enter your OpenAI API key in the sidebar to begin.")
    st.stop()

# ── Graph topology display ────────────────────────────────────────────────────
st.subheader("Graph Topology")
node_cols = st.columns(4)
node_containers = {}
for i, node in enumerate(NODES):
    with node_cols[i]:
        node_containers[node["id"]] = st.empty()
        node_containers[node["id"]].markdown(
            f"""<div style="background:#0a1018;border:1px solid #1a2535;border-radius:8px;
            padding:12px;text-align:center;">
            <span style="color:{node['color']};font-weight:700;font-family:monospace">
            {node['label']}</span><br/>
            <span style="font-size:10px;color:#556">{node['role']}</span>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown(
    "**Flow:** Retriever → Summarizer → Critic → Compositor → *(cycles back)*"
)
st.divider()

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl_col1, ctrl_col2 = st.columns([1, 3])

with ctrl_col1:
    run_btn = st.button(
        "▶ Run Simulation",
        type="primary",
        disabled=st.session_state.simulation_done,
        use_container_width=True,
    )
    if st.session_state.simulation_done:
        if st.button("🔄 Reset & Rerun", use_container_width=True):
            for k in ["asi_history", "mitigation_history", "baseline_embeddings",
                      "node_outputs", "graph_history"]:
                st.session_state[k] = [] if isinstance(st.session_state[k], list) else {}
            st.session_state.simulation_done = False
            st.session_state.mitigation_done = False
            st.rerun()

with ctrl_col2:
    progress_bar = st.progress(0, text="Ready — press Run Simulation to start")

log_container = st.empty()

# ── Simulation loop ───────────────────────────────────────────────────────────
if run_btn:
    local_history  = []
    local_baselines = {}
    local_prev     = {}
    curr_outputs   = {}
    asi_records    = []
    log_lines      = []

    for s in range(TOTAL_STEPS):
        node = NODES[s % len(NODES)]

        # Highlight the active node
        for nid, container in node_containers.items():
            n = next(x for x in NODES if x["id"] == nid)
            active = nid == node["id"]
            border = f"2px solid {n['color']}" if active else "1px solid #1a2535"
            bg     = "#1a2535" if active else "#0a1018"
            container.markdown(
                f"""<div style="background:{bg};border:{border};border-radius:8px;
                padding:12px;text-align:center;transition:all 0.3s;">
                <span style="color:{n['color']};font-weight:700;font-family:monospace">
                {n['label']}</span><br/>
                <span style="font-size:10px;color:#556">{n['role']}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        progress_bar.progress(
            s / TOTAL_STEPS,
            text=f"Step {s+1}/{TOTAL_STEPS} · {node['label']} running…",
        )

        # Execute node
        try:
            output = execute_node(
                st.session_state.api_key, node, s,
                local_history, TOTAL_STEPS, apply_mitigation=False,
            )
        except Exception as e:
            st.error(f"API error at step {s+1}: {e}")
            st.stop()

        curr_outputs[node["id"]] = output
        local_history.append({
            "step": s + 1,
            "node_id": node["id"],
            "node_label": node["label"],
            "output": output,
        })

        # Store baseline embedding (first visit per node)
        if node["id"] not in local_baselines:
            try:
                local_baselines[node["id"]] = get_embedding(
                    st.session_state.api_key, output
                )
            except Exception as e:
                st.warning(f"Embedding error at step {s+1}: {e}")

        # Compute ASI
        try:
            result = compute_asi(
                st.session_state.api_key, s,
                dict(curr_outputs), local_baselines, local_prev,
            )
        except Exception as e:
            result = {"asi": 0.75, "dims": {}}
            st.warning(f"ASI computation error at step {s+1}: {e}")

        record = {
            "step": s + 1,
            "node_id": node["id"],
            "node_label": node["label"],
            "output": output,
            "asi": result["asi"],
            **{f"dim_{k}": v for k, v in result["dims"].items()},
        }
        asi_records.append(record)
        local_prev[node["id"]] = output

        asi_val = result["asi"]
        status  = "🟢" if asi_val > 0.85 else "🟡" if asi_val > 0.70 else "🔴"
        log_lines.append(
            f"{status} Step {s+1:02d} · {node['label']:12s} · "
            f"ASI={asi_val:.3f}  "
            f"RC={result['dims'].get('response_consistency', 0):.2f}  "
            f"RS={result['dims'].get('reasoning_stability', 0):.2f}  "
            f"IA={result['dims'].get('inter_agent_agreement', 0):.2f}"
        )
        log_container.code("\n".join(log_lines[-14:]), language=None)
        time.sleep(0.1)

    # Persist results to session state
    st.session_state.asi_history        = asi_records
    st.session_state.baseline_embeddings = local_baselines
    st.session_state.node_outputs       = curr_outputs
    st.session_state.graph_history      = local_history
    st.session_state.simulation_done    = True

    progress_bar.progress(1.0, text="✅ Simulation complete!")
    st.success("Simulation complete — navigate to 📊 ASI Dashboard to explore results.")
    st.balloons()

# ── Summary if already run ────────────────────────────────────────────────────
if st.session_state.simulation_done and not run_btn:
    records = st.session_state.asi_history
    final   = records[-1]["asi"]
    onset   = next((r["step"] for r in records if r["asi"] < 0.85), None)

    c1, c2, c3 = st.columns(3)
    c1.metric("Final ASI",    f"{final:.3f}",
              delta=f"{final - records[0]['asi']:+.3f}")
    c2.metric("Drift Onset",  f"Step {onset}" if onset else "None — stable")
    c3.metric(
        "Status",
        "🟢 STABLE" if final > 0.85 else "🟡 DEGRADING" if final > 0.70 else "🔴 DRIFTED",
    )
    st.info("To rerun, press **Reset & Rerun** above.")
