import json
import pandas as pd
import streamlit as st
from core.agents import NODES, TASK_SPEC
from core.metrics import ASI_DIMS

st.header("Results and Export")
st.caption("Summary statistics, full step log, and data download")

if not st.session_state.simulation_done:
    st.info("Run the simulation first.")
    st.stop()

df = pd.DataFrame(st.session_state.asi_history)
df_mit = pd.DataFrame(st.session_state.mitigation_history) if st.session_state.mitigation_history else None

st.subheader("Summary Statistics")
base_stats = {
    "Run": "Baseline",
    "Final ASI": round(df["asi"].iloc[-1], 4),
    "Mean ASI": round(df["asi"].mean(), 4),
    "Min ASI": round(df["asi"].min(), 4),
    "Drift Onset (step)": int(df[df["asi"] < 0.85]["step"].min()) if (df["asi"] < 0.85).any() else "None",
    "Status": "STABLE" if df["asi"].iloc[-1] > 0.85 else "DEGRADING" if df["asi"].iloc[-1] > 0.70 else "DRIFTED",
}
rows = [base_stats]
if df_mit is not None:
    mit_stats = {
        "Run": "With Mitigation (EMC + BA)",
        "Final ASI": round(df_mit["asi"].iloc[-1], 4),
        "Mean ASI": round(df_mit["asi"].mean(), 4),
        "Min ASI": round(df_mit["asi"].min(), 4),
        "Drift Onset (step)": int(df_mit[df_mit["asi"] < 0.85]["step"].min()) if (df_mit["asi"] < 0.85).any() else "None",
        "Status": "STABLE" if df_mit["asi"].iloc[-1] > 0.85 else "DEGRADING" if df_mit["asi"].iloc[-1] > 0.70 else "DRIFTED",
    }
    rows.append(mit_stats)
st.dataframe(pd.DataFrame(rows).set_index("Run"), use_container_width=True)

st.divider()
st.subheader("Full Step Log")
display_cols = ["step", "node_label", "asi"] + [f"dim_{d['key']}" for d in ASI_DIMS]
available = [c for c in display_cols if c in df.columns]
rename_map = {f"dim_{d['key']}": d["label"] for d in ASI_DIMS}
st.dataframe(
    df[available].rename(columns=rename_map).style.background_gradient(subset=["asi"], cmap="RdYlGn", vmin=0.5, vmax=1.0),
    use_container_width=True, height=420)

st.divider()
st.subheader("Export Data")
dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button("Download Baseline CSV", df.to_csv(index=False).encode("utf-8"), "asi_baseline.csv", "text/csv", use_container_width=True)
with dl2:
    if df_mit is not None:
        st.download_button("Download Mitigation CSV", df_mit.to_csv(index=False).encode("utf-8"), "asi_mitigation.csv", "text/csv", use_container_width=True)
    else:
        st.button("Mitigation CSV (run mitigation first)", disabled=True, use_container_width=True)
with dl3:
    export_payload = {"task": TASK_SPEC, "nodes": [n["id"] for n in NODES], "asi_dims": [d["key"] for d in ASI_DIMS], "baseline": st.session_state.asi_history, "mitigation": st.session_state.mitigation_history}
    st.download_button("Download Full JSON", json.dumps(export_payload, indent=2).encode("utf-8"), "drift_experiment.json", "application/json", use_container_width=True)

st.divider()
if st.session_state.graph_history:
    st.subheader("Full Graph State History")
    for entry in st.session_state.graph_history:
        with st.expander(f"Step {entry['step']} - {entry['node_label']}"):
            st.write(entry["output"])
