# Context Drift Observatory

A Streamlit app for investigating **context drift in LLM-based multi-agent systems**, using a LangGraph Research Synthesis pipeline as the experimental use case.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/context-drift-observatory
cd context-drift-observatory
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # paste your OPENAI_API_KEY
streamlit run app.py
```

Open `http://localhost:8501` — enter your key in the sidebar, press **Run Simulation**.

## What It Does

Runs a 4-node cyclic LangGraph pipeline over 16 steps on a research synthesis task.  
Measures the **Agent Stability Index (ASI)** at every step across 4 sub-dimensions:

| Dimension | Method |
|---|---|
| Response Consistency | Cosine similarity to baseline embedding |
| Reasoning Stability | Cosine similarity vs previous step |
| Inter-Agent Agreement | Jaccard term overlap across nodes |
| Task Adherence | gpt-4o-mini LLM-judge score |

Reruns with **Episodic Memory Consolidation + Behavioral Anchoring** to compare mitigation.

## Pages

| Page | Purpose |
|---|---|
| Run Simulation | Baseline LangGraph simulation (16 steps) |
| Mitigation Run | Rerun with EMC + Behavioral Anchoring |
| ASI Dashboard | Trajectory chart, heatmap, dimension breakdown, output viewer |
| Results and Export | Summary stats, step log, CSV and JSON download |

## ASI Thresholds

| Range | Status |
|---|---|
| ASI >= 0.85 | Stable |
| 0.70-0.85 | Degrading |
| ASI < 0.70 | Drifted |

## Project Structure

```
context-drift-observatory/
├── app.py                 Entrypoint + shared sidebar
├── pages/
│   ├── 1_Simulation.py    Baseline simulation
│   ├── 2_ASI_Dashboard.py ASI visualisation
│   ├── 3_Mitigation.py    Mitigation run
│   └── 4_Results.py       Export
├── core/
│   ├── agents.py          Node execution logic
│   ├── metrics.py         ASI computation
│   └── embeddings.py      Embedding helpers
├── .streamlit/config.toml Dark theme
├── requirements.txt
└── .env.example
```

## API Cost

Full run (baseline + mitigation, 32 steps): approx $0.05-0.10 with gpt-4o-mini.

## License

MIT

<!--placeholder_end-->
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
