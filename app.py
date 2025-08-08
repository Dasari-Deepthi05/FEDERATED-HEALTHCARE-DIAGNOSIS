# app.py
import streamlit as st
import pandas as pd
import os

st.title("🩺 Federated Learning Healthcare Simulator")

st.subheader("📊 Hospital Clients")
files = [f for f in os.listdir() if f.startswith("client") and f.endswith(".csv")]

for f in files:
    df = pd.read_csv(f)
    st.write(f"**{f}** — {len(df)} records")

if st.button("▶️ Run 1 FedAvg Round"):
    st.success("Simulated 1 round! (Code backend can be added later)")

st.caption("More features coming soon: accuracy plots, DP indicators, etc.")
