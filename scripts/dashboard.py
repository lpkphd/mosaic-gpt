import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

SCRIPTS_DIR = Path(__file__).parent
STATUS_PATH = SCRIPTS_DIR / "status.json"
JOURNAL_PATH = SCRIPTS_DIR / "journal.jsonl"

st.set_page_config(page_title="MOSAIC-GPT", layout="wide")

STATUS_COLORS = {
    "healthy": "#2ecc71",
    "stalled": "#f1c40f",
    "crashed": "#e74c3c",
    "down": "#e74c3c",
    "loading": "#3498db",
    "unknown": "#95a5a6",
}

SEVERITY_COLORS = {
    "CRASH": "#e74c3c",
    "AUTO_RESTART": "#f39c12",
    "OUTBID": "#e67e22",
    "REPLACE": "#3498db",
    "HEALTHY": "#2ecc71",
    "STALL": "#f1c40f",
    "INFO": "#95a5a6",
    "COMPLETED": "#2ecc71",
}


def load_status():
    if not STATUS_PATH.exists():
        return None
    try:
        with open(STATUS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def load_journal():
    if not JOURNAL_PATH.exists():
        return []
    events = []
    try:
        with open(JOURNAL_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except IOError:
        return []
    return events


def status_badge(status):
    color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.85em;font-weight:600;">{status.upper()}</span>'


def format_eta(eta_hrs):
    if eta_hrs is None:
        return "N/A"
    if eta_hrs < 1:
        return f"{eta_hrs * 60:.0f}m"
    if eta_hrs < 24:
        return f"{eta_hrs:.1f}h"
    days = eta_hrs / 24
    return f"{days:.1f}d"


def format_cost(cost):
    if cost is None:
        return "N/A"
    return f"${cost:.2f}"


def bar_color(pct, yellow_thresh=75, red_thresh=90):
    if pct >= red_thresh:
        return "#e74c3c"
    if pct >= yellow_thresh:
        return "#f1c40f"
    return "#2ecc71"


def render_usage_bar(label, pct, yellow_thresh=75, red_thresh=90):
    color = bar_color(pct, yellow_thresh, red_thresh)
    st.markdown(
        f"""
        <div style="margin-bottom:4px;">
            <span style="font-size:0.85em;">{label}: {pct:.0f}%</span>
            <div style="background:#333;border-radius:4px;height:12px;width:100%;">
                <div style="background:{color};border-radius:4px;height:12px;width:{min(pct, 100):.1f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


status = load_status()

if status is None:
    st.title("MOSAIC-GPT Training Dashboard")
    st.warning("Waiting for monitor... (no status.json found)")
    time.sleep(10)
    st.rerun()

instances = status.get("instances", [])
timestamp = status.get("timestamp", "unknown")
total_burn = status.get("total_cost_hr", 0)

st.markdown("## MOSAIC-GPT Training Dashboard")

header_cols = st.columns([3, 1])
with header_cols[0]:
    st.caption(f"Last updated: {timestamp}")
with header_cols[1]:
    st.metric("Burn Rate", f"${total_burn:.2f}/hr")

st.divider()

# --- Training Progress ---
if instances:
    cols = st.columns(len(instances))
    for col, inst in zip(cols, instances):
        with col:
            name = inst.get("model_name", "unknown")
            s = inst.get("status", "unknown")
            step = inst.get("step", 0)
            max_steps = inst.get("max_steps", 50000)
            ppl = inst.get("ppl")
            loss = inst.get("loss")
            tok_s = inst.get("tok_s")
            eta_hrs = inst.get("eta_hrs")
            cost_total = inst.get("cost_total")
            cost_hr = inst.get("cost_per_hr")
            pct = inst.get("pct_complete", 0)

            projected_total = None
            if cost_hr and eta_hrs and cost_total is not None:
                projected_total = cost_total + cost_hr * eta_hrs

            st.markdown(
                f"### {name} {status_badge(s)}", unsafe_allow_html=True
            )
            st.progress(min(pct / 100, 1.0))
            st.caption(f"Step {step:,} / {max_steps:,} ({pct:.1f}%)")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("PPL", f"{ppl:.1f}" if ppl is not None else "N/A")
                st.metric("Tokens/s", f"{tok_s:,.0f}" if tok_s else "N/A")
                st.metric("Cost", format_cost(cost_total))
            with m2:
                st.metric("Loss", f"{loss:.3f}" if loss is not None else "N/A")
                st.metric("ETA", format_eta(eta_hrs))
                st.metric("Projected", format_cost(projected_total))

st.divider()

# --- Health Panel ---
st.markdown("### Health")
if instances:
    health_cols = st.columns(len(instances))
    for col, inst in zip(health_cols, instances):
        with col:
            name = inst.get("model_name", "unknown")
            s = inst.get("status", "unknown")
            gpu = inst.get("gpu_util", 0)
            disk = inst.get("disk_pct", 0)
            hb = inst.get("heartbeat_age_s")

            st.markdown(f"**{name}** {status_badge(s)}", unsafe_allow_html=True)
            render_usage_bar("GPU", gpu, yellow_thresh=50, red_thresh=30)
            render_usage_bar("Disk", disk, yellow_thresh=75, red_thresh=90)

            if hb is not None:
                hb_color = "#2ecc71" if hb < 120 else "#f1c40f" if hb < 300 else "#e74c3c"
                st.markdown(
                    f'<span style="font-size:0.85em;">Heartbeat: <span style="color:{hb_color};font-weight:600;">{hb}s ago</span></span>',
                    unsafe_allow_html=True,
                )

st.divider()

# --- Cost Panel ---
st.markdown("### Cost")
cost_cols = st.columns(len(instances) + 1) if instances else [st.columns(1)[0]]

if instances:
    total_spend = sum(inst.get("cost_total", 0) or 0 for inst in instances)
    total_projected = sum(
        (inst.get("cost_total", 0) or 0) + (inst.get("cost_per_hr", 0) or 0) * (inst.get("eta_hrs", 0) or 0)
        for inst in instances
    )

    with cost_cols[0]:
        st.metric("Total Spend", format_cost(total_spend))
        st.metric("Projected Total", format_cost(total_projected))

    for i, inst in enumerate(instances):
        with cost_cols[i + 1]:
            st.metric(
                inst.get("model_name", "unknown"),
                format_cost(inst.get("cost_per_hr")),
                delta=f"{format_cost(inst.get('cost_total'))} spent",
                delta_color="off",
            )

st.divider()

# --- PPL Progress Chart ---
st.markdown("### Perplexity Over Steps")

if instances:
    fig = go.Figure()
    for inst in instances:
        name = inst.get("model_name", "unknown")
        step = inst.get("step", 0)
        ppl = inst.get("ppl")
        if step and ppl is not None:
            fig.add_trace(
                go.Scatter(
                    x=[step],
                    y=[ppl],
                    mode="markers+text",
                    name=name,
                    text=[f"{ppl:.1f}"],
                    textposition="top center",
                    marker=dict(size=12),
                )
            )
    fig.update_layout(
        xaxis_title="Step",
        yaxis_title="Perplexity",
        yaxis_type="log",
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No instance data available for chart.")

st.divider()

# --- Journal Panel ---
st.markdown("### Event Journal")

journal = load_journal()
if journal:
    model_names = sorted(set(e.get("model", "") for e in journal if e.get("model")))
    severity_types = sorted(set(e.get("type", "") for e in journal if e.get("type")))

    filter_cols = st.columns(2)
    with filter_cols[0]:
        model_filter = st.multiselect("Filter by model", model_names, default=model_names)
    with filter_cols[1]:
        severity_filter = st.multiselect("Filter by severity", severity_types, default=severity_types)

    filtered = [
        e for e in reversed(journal)
        if e.get("model", "") in model_filter and e.get("type", "") in severity_filter
    ]

    for event in filtered[:50]:
        ts = event.get("ts", "")
        etype = event.get("type", "INFO")
        model = event.get("model", "")
        msg = event.get("message", "")
        color = SEVERITY_COLORS.get(etype, "#95a5a6")
        st.markdown(
            f'<div style="padding:4px 0;border-bottom:1px solid #333;font-size:0.88em;">'
            f'<span style="color:#888;">{ts}</span> '
            f'<span style="background:{color};color:#fff;padding:1px 6px;border-radius:8px;font-size:0.82em;">{etype}</span> '
            f'<strong>{model}</strong> {msg}</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("No journal events yet.")

st.divider()

# --- Instance Details ---
st.markdown("### Instance Details")

if instances:
    for inst in instances:
        iid = inst.get("instance_id", "N/A")
        label = inst.get("label", "")
        model = inst.get("model_name", "unknown")
        uptime = inst.get("uptime_hrs")

        with st.expander(f"{model} -- {label} (ID: {iid})"):
            d1, d2 = st.columns(2)
            with d1:
                st.text(f"Instance ID: {iid}")
                st.text(f"Label:       {label}")
                st.text(f"SSH:         ssh root@<host> -p <port>")
            with d2:
                st.text(f"Uptime:      {uptime:.1f}h" if uptime else "Uptime: N/A")
                st.text(f"GPU Util:    {inst.get('gpu_util', 'N/A')}%")
                st.text(f"Status:      {inst.get('status', 'unknown')}")

# --- Auto-refresh ---
time.sleep(10)
st.rerun()
