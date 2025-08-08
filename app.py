# app.py
import streamlit as st
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import os


st.title("🩺 Federated Learning Healthcare Simulator")

st.subheader("📊 Hospital Clients")
files = [f for f in os.listdir() if f.startswith("client") and f.endswith(".csv")]

for f in files:
    df = pd.read_csv(f)
    st.write(f"**{f}** — {len(df)} records")

if st.button("▶️ Run 1 FedAvg Round"):
    st.subheader("📈 Accuracy Trend Over Federated Rounds")

if os.path.exists("results/fedavg_accuracy.csv"):
    acc_df = pd.read_csv("results/fedavg_accuracy.csv")
    st.line_chart(acc_df.set_index('round'))
else:
    st.info("Run fedavg_sim.py first to generate accuracy data.")
    st.success("Simulated 1 round! (Code backend can be added later)")

st.caption("More features coming soon: accuracy plots, DP indicators, etc.")
