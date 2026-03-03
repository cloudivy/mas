import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from core.metrics import ASI_DIMS
from core.agents import NODES

st.header("ASI Dashboard")
st.caption("Agent Stability Index - trajectory, dimensions, per-node heatmap, output viewer")

if not st.session_state.simulation_done:
    st.info("Run the simulation first on the Simulation page.")
    st.stop()

records = st.session_state.asi_history
df = pd.DataFrame(records)
mit_records = st.session_state.mitigation_history
df_mit = pd.DataFrame(mit_records) if mit_records else None

final_asi = df["asi"].iloc[-1]
min_asi = df["asi"].min()
onset = df[df["asi"] < 0.85]["step"].min() if (df["asi"] < 0.85).any() else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Final ASI", f"{final_asi:.3f}", f"{final_asi - df['asi'].iloc[0]:+.3f}")
c2.metric("Minimum ASI", f"{min_asi:.3f}")
c3.metric("Drift Onset", f"Step {int(onset)}" if onset else "None")
c4.metric("Status", "STABLE" if final_asi > 0.85 else "DEGRADING" if final_asi > 0.70 else "DRIFTED")

st.divider()
st.subheader("ASI Trajectory")

fig = go.Figure()
fig.add_hrect(y0=0.85, y1=1.05, fillcolor="rgba(76,175,82,0.06)", line_width=0, annotation_text="Stable >= 0.85", annotation_position="top right")
fig.add_hrect(y0=0.70, y1=0.85, fillcolor="rgba(230,192,64,0.06)", line_width=0, annotation_text="Degrading 0.70-0.85", annotation_position="top right")
fig.add_hrect(y0=0.0, y1=0.70, fillcolor="rgba(224,85,85,0.06)", line_width=0, annotation_text="Drifted < 0.70", annotation_position="top right")
fig.add_hline(y=0.85, line_dash="dot", line_color="rgba(76,175,82,0.4)", line_width=1)
fig.add_hline(y=0.70, line_dash="dot", line_color="rgba(230,192,64,0.4)", line_width=1)

for dim in ASI_DIMS:
    col_key = f"dim_{dim['key']}"
    if col_key in df.columns:
        vals = df[col_key].ffill()
        fig.add_trace(go.Scatter(x=df['step'], y=vals, name=dim['label'],
            line=dict(color=dim['color'], width=1, dash='dot'), opacity=0.35, mode='lines'))

if df_mit is not None and not df_mit.empty:
    fig.add_trace(go.Scatter(x=df_mit['step'], y=df_mit['asi'], name='ASI (w/ Mitigation)',
        mode='lines+markers', line=dict(color='#e6c040', width=2, dash='dash'),
        marker=dict(size=5, color='#e6c040')))

dot_colors = ['#4caf82' if v > 0.85 else '#e6c040' if v > 0.70 else '#e05555' for v in df['asi']]
fig.add_trace(go.Scatter(x=df['step'], y=df['asi'], name='ASI (baseline)',
    mode='lines+markers', line=dict(color='#4FC3F7', width=3),
    marker=dict(size=7, color=dot_colors, line=dict(color='#070b10', width=1.5))))

fig.update_layout(template='plotly_dark', paper_bgcolor='#0a1018', plot_bgcolor='#070b10',
    height=380, xaxis_title='Interaction Step', yaxis_title='ASI Score', yaxis_range=[0.4, 1.05],
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(l=40, r=20, t=30, b=40))
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("ASI Sub-Dimension Breakdown - Latest Step")
latest = df.iloc[-1]
dim_cols = st.columns(4)
for i, dim in enumerate(ASI_DIMS):
    col_key = f"dim_{dim['key']}"
    val = latest.get(col_key)
    if val is None or pd.isna(val):
        val = 0.75
    pct = round(float(val) * 100)
    with dim_cols[i]:
        st.metric(dim['label'], f"{float(val):.3f}")
        st.progress(pct / 100, text=f"{pct}%")
        st.caption(dim['desc'])

st.divider()
st.subheader("Per-Node ASI Heatmap")
node_ids = [n['id'] for n in NODES]
node_labels = [n['label'] for n in NODES]
pivot = (df.groupby(['step', 'node_id'])['asi'].mean().reset_index()
          .pivot(index='step', columns='node_id', values='asi').reindex(columns=node_ids))
pivot.columns = node_labels
fig2 = px.imshow(pivot.T, color_continuous_scale=['#e05555', '#e6c040', '#4caf82'],
    zmin=0.5, zmax=1.0, aspect='auto', labels=dict(x='Step', y='Node', color='ASI'))
fig2.update_layout(template='plotly_dark', paper_bgcolor='#0a1018', plot_bgcolor='#070b10',
    height=210, margin=dict(l=40, r=20, t=20, b=40),
    coloraxis_colorbar=dict(thickness=12, len=0.8))
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Sub-Dimension Trajectories")
fig3 = go.Figure()
for dim in ASI_DIMS:
    col_key = f"dim_{dim['key']}"
    if col_key in df.columns:
        vals = df[col_key].ffill()
        fig3.add_trace(go.Scatter(x=df['step'], y=vals, name=dim['label'],
            mode='lines+markers', line=dict(color=dim['color'], width=2), marker=dict(size=5)))
fig3.update_layout(template='plotly_dark', paper_bgcolor='#0a1018', plot_bgcolor='#070b10',
    height=300, xaxis_title='Step', yaxis_title='Score', yaxis_range=[0.3, 1.05],
    legend=dict(orientation='h', yanchor='bottom', y=1.02), margin=dict(l=40, r=20, t=30, b=40))
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.subheader("Node Output Viewer")
selected_step = st.slider("Select step", 1, len(df), len(df))
row = df[df['step'] == selected_step].iloc[0]
v1, v2 = st.columns([1, 3])
with v1:
    st.markdown(f"**Node:** `{row['node_label']}`")
    st.markdown(f"**ASI:** `{row['asi']:.3f}`")
    status = "STABLE" if row['asi'] > 0.85 else "DEGRADING" if row['asi'] > 0.70 else "DRIFTED"
    st.markdown(f"**Status:** {status}")
    for dim in ASI_DIMS:
        ck = f"dim_{dim['key']}"
        v = row.get(ck)
        if v is not None and not pd.isna(v):
            st.markdown(f"**{dim['label'][:12]}:** `{float(v):.3f}`")
with v2:
    st.text_area("Output text", value=str(row.get('output', '')), height=180, disabled=True)
