import streamlit as st

st.set_page_config(
    page_title="Context Drift Observatory",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

defaults = {
    "api_key": "",
    "asi_history": [],
    "mitigation_history": [],
    "baseline_embeddings": {},
    "node_outputs": {},
    "graph_history": [],
    "simulation_done": False,
    "mitigation_done": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

pages = {
    "Experiment": [
        st.Page("pages/1_Simulation.py",    title="Run Simulation",    icon="🔬"),
        st.Page("pages/3_Mitigation.py",    title="Mitigation Run",    icon="⚓"),
    ],
    "Analysis": [
        st.Page("pages/2_ASI_Dashboard.py", title="ASI Dashboard",    icon="📊"),
        st.Page("pages/4_Results.py",       title="Results & Export", icon="📋"),
    ],
}

pg = st.navigation(pages)

with st.sidebar:
    st.markdown("## 🔭 Context Drift Observatory")
    st.caption("LangGraph · Research Synthesis · ASI Monitor")
    st.divider()

    key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-...",
        help="Used for gpt-4o-mini completions and text-embedding-3-small",
    )
    if key_input:
        st.session_state.api_key = key_input

    st.divider()
    if st.session_state.simulation_done:
        st.success(f"✅ Simulation complete — {len(st.session_state.asi_history)} steps")
    else:
        st.info("Run the simulation to begin")
    if st.session_state.mitigation_done:
        st.success("✅ Mitigation run complete")
    st.divider()
    st.caption("Research: context drift in LLM-based multi-agent systems")

pg.run()
