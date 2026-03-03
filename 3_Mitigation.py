import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.agents import execute_node, NODES
from core.metrics import compute_asi
from core.embeddings import get_embedding

TOTAL_STEPS = 16

st.header("Mitigation Run")
st.caption("Episodic Memory Consolidation + Behavioral Anchoring vs baseline")

if not st.session_state.simulation_done:
    st.info("Complete the baseline simulation on the Simulation page first.")
    st.stop()

with st.expander("Mitigation Strategies Applied", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Episodic Memory Consolidation (EMC)**")
        st.markdown(
            "Each node only receives the last 4 steps of graph state history, "
            "preventing context window pollution from early interactions."
        )
    with col2:
        st.markdown("**Behavioral Anchoring (BA)**")
        st.markdown(
            "The original task specification is re-injected into every node "
            "system prompt at every step, re-grounding agents to the original objective."
        )

st.divider()

ctrl1, ctrl2 = st.columns([1, 3])
with ctrl1:
    run_btn = st.button(
        "Run Mitigation",
        type="primary",
        disabled=st.session_state.mitigation_done,
        use_container_width=True,
    )
    if st.session_state.mitigation_done:
        if st.button("Re-run Mitigation", use_container_width=True):
            st.session_state.mitigation_history = []
            st.session_state.mitigation_done = False
            st.rerun()
with ctrl2:
    progress_bar = st.progress(0, text="Ready")

log_container = st.empty()

if run_btn:
    log_lines = []
    local_history = []
    local_baselines = {}
    local_prev = {}
    curr_outputs = {}
    asi_records = []

    progress_bar.progress(0, text="Starting mitigation run...")

    for s in range(TOTAL_STEPS):
        node = NODES[s % len(NODES)]
        progress_bar.progress(
            s / TOTAL_STEPS,
            text=f"Step {s+1}/{TOTAL_STEPS} - {node['label']}"
        )

        try:
            output = execute_node(
                st.session_state.api_key, node, s,
                local_history, TOTAL_STEPS, apply_mitigation=True
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

        if node["id"] not in local_baselines:
            try:
                local_baselines[node["id"]] = get_embedding(
                    st.session_state.api_key, output
                )
            except Exception:
                pass

        try:
            result = compute_asi(
                st.session_state.api_key, s,
                dict(curr_outputs), local_baselines, local_prev
            )
        except Exception:
            result = {"asi": 0.75, "dims": {}}

        asi_records.append({
            "step": s + 1,
            "node_id": node["id"],
            "node_label": node["label"],
            "output": output,
            "asi": result["asi"],
            **{f"dim_{k}": v for k, v in result["dims"].items()},
        })
        local_prev[node["id"]] = output

        asi_val = result["asi"]
        status = "STABLE" if asi_val > 0.85 else "DEGRADING" if asi_val > 0.70 else "DRIFTED"
        log_lines.append(
            f"[{status}] Step {s+1:02d} - {node['label']:12s} - ASI={asi_val:.3f}"
        )
        log_container.code("\n".join(log_lines[-12:]), language=None)
        time.sleep(0.1)

    st.session_state.mitigation_history = asi_records
    st.session_state.mitigation_done = True
    progress_bar.progress(1.0, text="Mitigation run complete!")
    st.success("Mitigation run complete - see comparison below and on the ASI Dashboard.")

if st.session_state.mitigation_done:
    st.divider()
    st.subheader("Baseline vs Mitigation - ASI Comparison")

    df_base = pd.DataFrame(st.session_state.asi_history)
    df_mit = pd.DataFrame(st.session_state.mitigation_history)

    fig = go.Figure()
    fig.add_hline(y=0.85, line_dash="dot", line_color="rgba(76,175,82,0.4)", line_width=1)
    fig.add_hline(y=0.70, line_dash="dot", line_color="rgba(230,192,64,0.4)", line_width=1)
    fig.add_trace(go.Scatter(
        x=df_base["step"], y=df_base["asi"],
        name="Baseline (no mitigation)",
        mode="lines+markers",
        line=dict(color="#4FC3F7", width=2.5),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=df_mit["step"], y=df_mit["asi"],
        name="With Mitigation (EMC + BA)",
        mode="lines+markers",
        line=dict(color="#e6c040", width=2.5, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a1018",
        plot_bgcolor="#070b10",
        height=360,
        xaxis_title="Step",
        yaxis_title="ASI",
        yaxis_range=[0.4, 1.05],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    base_final = df_base["asi"].iloc[-1]
    mit_final = df_mit["asi"].iloc[-1]
    base_avg = df_base["asi"].mean()
    mit_avg = df_mit["asi"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline Final ASI", f"{base_final:.3f}")
    m2.metric("Mitigation Final ASI", f"{mit_final:.3f}",
              delta=f"{mit_final - base_final:+.3f}")
    m3.metric("Baseline Avg ASI", f"{base_avg:.3f}")
    m4.metric("Mitigation Avg ASI", f"{mit_avg:.3f}",
              delta=f"{mit_avg - base_avg:+.3f}")

    st.divider()
    st.subheader("Step-by-step comparison")
    compare_df = pd.DataFrame({
        "Step": df_base["step"],
        "Node": df_base["node_label"],
        "ASI Baseline": df_base["asi"].round(4),
        "ASI Mitigation": df_mit["asi"].round(4),
        "Delta": (df_mit["asi"] - df_base["asi"]).round(4),
    })
    st.dataframe(
        compare_df.style.background_gradient(
            subset=["Delta"], cmap="RdYlGn", vmin=-0.2, vmax=0.2
        ),
        use_container_width=True,
        height=400,
    )
