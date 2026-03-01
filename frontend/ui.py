import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("Airfare Affordability Simulator")
st.caption("Predict fares and simulate how competition and hubs change affordability.")

c1, c2 = st.columns(2)

with c1:
    nsmiles = st.number_input("Distance (miles)", min_value=1.0, value=1000.0, step=50.0)
    passengers = st.number_input("Passengers (demand proxy)", min_value=0.0, value=5000.0, step=100.0)
    year = st.selectbox("Year", [2021, 2022, 2023, 2024, 2025], index=3)

with c2:
    large_ms = st.slider("Dominant carrier share (large_ms)", 0.0, 1.0, 0.60, 0.01)
    lf_ms = st.slider("Low-cost carrier share (lf_ms)", 0.0, 1.0, 0.20, 0.01)
    hub_intensity = st.selectbox("Hub intensity", [0, 1, 2], index=1)

payload = {
    "nsmiles": float(nsmiles),
    "passengers": float(passengers),
    "large_ms": float(large_ms),
    "lf_ms": float(lf_ms),
    "hub_intensity": int(hub_intensity),
    "Year": int(year),
}

def call_api(p):
    r = requests.post(API_URL, json=p, timeout=15)
    r.raise_for_status()
    return r.json()

if st.button("Predict fare"):
    try:
        out = call_api(payload)
        st.subheader("Prediction")
        st.metric("Predicted fare ($)", f"{out['predicted_fare']:.2f}")
        st.caption(f"log_fare = {out['predicted_log_fare']:.4f}")
    except Exception as e:
        st.error(f"API error: {e}")

st.divider()
st.subheader("Scenario: increase low-cost competition")
st.caption("Compare current route vs lf_ms = 0.40 (holding everything else fixed).")

if st.button("Run LCC scenario (lf_ms → 0.40)"):
    try:
        base = call_api(payload)
        alt = dict(payload)
        alt["lf_ms"] = 0.40
        alt_out = call_api(alt)

        base_fare = base["predicted_fare"]
        alt_fare = alt_out["predicted_fare"]
        savings = base_fare - alt_fare
        pct = (savings / base_fare * 100) if base_fare > 0 else 0

        a, b, c = st.columns(3)
        a.metric("Current ($)", f"{base_fare:.2f}")
        b.metric("With lf_ms=0.40 ($)", f"{alt_fare:.2f}")
        c.metric("Savings", f"{savings:.2f}", f"{pct:.1f}%")
    except Exception as e:
        st.error(f"Scenario error: {e}")