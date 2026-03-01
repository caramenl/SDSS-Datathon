import streamlit as st
import requests
import math

API_URL = "http://127.0.0.1:8000/predict"

# ---------- Page config ----------
st.set_page_config(
    page_title="Airfare Affordability Flight Deck",
    page_icon="✈️",
    layout="wide"
)

# ---------- Small helpers ----------
def call_api(payload):
    r = requests.post(API_URL, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

def pct_change(a, b):
    # from a to b
    return ((b - a) / a * 100) if a else 0

def fmt_money(x):
    return f"${x:,.2f}"

# ---------- Header ----------
st.title("✈️ Airfare Affordability Flight Deck")
st.caption(
    "An educational simulator that predicts airfares and demonstrates how competition, "
    "low-cost carriers, and hub structure shape affordability for students and families."
)

st.info(
    "How to use: enter a route’s distance + demand + market structure → get a predicted fare → "
    "run scenarios that simulate competition entry or removing hub premiums."
)

# ---------- Sidebar: model explanation ----------
with st.sidebar:
    st.header("How the model works")
    st.write(
        "**Model type:** Gradient Boosting (tree-based)\n\n"
        "**Target:** Predicts *log(fare)* then converts back to dollars.\n\n"
        "**Why log?** Fares are right-skewed; logs stabilize variance and improve predictions.\n\n"
        "**Core idea:** Distance sets baseline cost, but **market structure** (dominance, LCC presence, hubs) "
        "creates systematic markups/discounts."
    )
    st.markdown("---")
    st.subheader("What each input means")
    st.write(
        "- **Distance (miles):** cost proxy; longer flights generally cost more.\n"
        "- **Passengers:** demand proxy; denser routes can be cheaper due to scale.\n"
        "- **Dominant share (large_ms):** market power; higher values → higher prices.\n"
        "- **Low-cost share (lf_ms):** competition pressure; higher values → lower prices.\n"
        "- **Hub intensity:** network power; hub-to-hub routes often carry premiums.\n"
        "- **Year:** captures period-wide shifts (e.g., post-pandemic trends)."
    )

# ---------- Inputs ----------
st.subheader("Route Inputs (Your flight plan)")

c1, c2, c3 = st.columns([1.1, 1.1, 1.0])

with c1:
    st.markdown("### Route fundamentals")
    nsmiles = st.number_input("Distance (miles)", min_value=1.0, value=1000.0, step=25.0)
    passengers = st.number_input("Passengers (route demand proxy)", min_value=0.0, value=5000.0, step=100.0)
    year = st.selectbox("Year", [2021, 2022, 2023, 2024, 2025], index=3)

with c2:
    st.markdown("### Market structure")
    large_ms = st.slider("Dominant carrier share (large_ms)", 0.0, 1.0, 0.60, 0.01)
    lf_ms = st.slider("Low-cost carrier share (lf_ms)", 0.0, 1.0, 0.20, 0.01)
    hub_intensity = st.selectbox("Hub intensity (0=none, 1=one hub, 2=hub-to-hub)", [0, 1, 2], index=1)

with c3:
    st.markdown("### Quick checks")
    st.write("**Derived transforms (for transparency):**")
    log_distance = math.log(max(nsmiles, 1e-6))
    log_passengers = math.log(passengers + 1.0)
    st.code(
        f"log_distance = ln({nsmiles:.0f}) = {log_distance:.4f}\n"
        f"log_passengers = ln({passengers:.0f} + 1) = {log_passengers:.4f}",
        language="text"
    )
    st.write("**Interpretation tip:** log inputs make effects comparable across small/large routes.")

payload = {
    "nsmiles": float(nsmiles),
    "passengers": float(passengers),
    "large_ms": float(large_ms),
    "lf_ms": float(lf_ms),
    "hub_intensity": int(hub_intensity),
    "Year": int(year),
}

# ---------- Prediction ----------
st.markdown("---")
st.subheader("Prediction (Estimated fare)")

pred_col, edu_col = st.columns([1.0, 1.2])

with pred_col:
    if st.button("🧾 Predict fare", use_container_width=True):
        try:
            out = call_api(payload)
            st.session_state["base_out"] = out
        except Exception as e:
            st.error(f"API error: {e}")

    base_out = st.session_state.get("base_out")
    if base_out:
        st.metric("Predicted fare", fmt_money(base_out["predicted_fare"]))
        st.caption(f"log_fare = {base_out['predicted_log_fare']:.4f}")

with edu_col:
    st.markdown("### What this prediction represents")
    st.write(
        "This predicted fare is the model’s estimate **given distance, demand, competition, and hub exposure**. "
        "It’s useful for comparing *structural affordability*: what changes when competition increases or a route "
        "is less hub-dependent."
    )
    st.write(
        "**Key idea:** If two routes have similar distance and demand but different market structure, the model "
        "should predict systematically different fares — that’s the affordability barrier story."
    )

# ---------- Scenarios (Best Insights) ----------
st.markdown("---")
st.subheader("Scenario Simulator (Counterfactual affordability)")

st.caption(
    "These scenarios hold distance and demand fixed and change **market structure** to estimate how fares would respond."
)

scenario_row1 = st.columns(3)
scenario_row2 = st.columns(3)

def ensure_base():
    base_out = st.session_state.get("base_out")
    if not base_out:
        # Auto-run base prediction if user didn't click
        out = call_api(payload)
        st.session_state["base_out"] = out
    return st.session_state["base_out"]

# 1) LCC scenario
with scenario_row1[0]:
    st.markdown("#### Add low-cost competition")
    st.caption("Set lf_ms → 0.40 (holding others fixed)")
    if st.button("Run LCC scenario", use_container_width=True):
        try:
            base = ensure_base()
            alt = dict(payload); alt["lf_ms"] = 0.40
            alt_out = call_api(alt)

            base_f = base["predicted_fare"]
            alt_f = alt_out["predicted_fare"]
            savings = base_f - alt_f

            st.metric("Current", fmt_money(base_f))
            st.metric("With lf_ms=0.40", fmt_money(alt_f))
            st.metric("Savings", fmt_money(savings), f"{-pct_change(base_f, alt_f):.1f}%")
        except Exception as e:
            st.error(f"Scenario error: {e}")

# 2) Dominated -> competitive
with scenario_row1[1]:
    st.markdown("#### Dominated → competitive market")
    st.caption("Set large_ms → 0.40 and lf_ms → 0.40")
    if st.button("Run competition entry", use_container_width=True):
        try:
            base = ensure_base()
            alt = dict(payload); alt["large_ms"] = 0.40; alt["lf_ms"] = 0.40
            alt_out = call_api(alt)

            base_f = base["predicted_fare"]
            alt_f = alt_out["predicted_fare"]
            gap = base_f - alt_f

            st.metric("Dominated market", fmt_money(base_f))
            st.metric("Competitive market", fmt_money(alt_f))
            st.metric("Affordability gain", fmt_money(gap), f"{-pct_change(base_f, alt_f):.1f}%")
        except Exception as e:
            st.error(f"Scenario error: {e}")

# 3) Remove hub premium
with scenario_row1[2]:
    st.markdown("####  Remove hub premium")
    st.caption("Set hub_intensity → 0")
    if st.button("Run hub removal", use_container_width=True):
        try:
            base = ensure_base()
            alt = dict(payload); alt["hub_intensity"] = 0
            alt_out = call_api(alt)

            base_f = base["predicted_fare"]
            alt_f = alt_out["predicted_fare"]
            premium = base_f - alt_f

            st.metric("Current route", fmt_money(base_f))
            st.metric("Non-hub route", fmt_money(alt_f))
            st.metric("Hub premium", fmt_money(premium), f"{-pct_change(base_f, alt_f):.1f}%")
        except Exception as e:
            st.error(f"Scenario error: {e}")

