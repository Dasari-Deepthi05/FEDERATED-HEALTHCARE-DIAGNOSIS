import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Federated Healthcare Training", layout="wide")

st.title("🏥 Federated Learning for Privacy-Preserving Healthcare Diagnosis")

# --- Sidebar ---
st.sidebar.header("⚙ Training Settings")

# Dataset selector
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    ["diabetes", "heart"]
)

# Model selector
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["MLP", "Logistic Regression"]
)

# Number of training rounds
num_rounds = st.sidebar.slider("Training Rounds", min_value=1, max_value=10, value=5)

# Run training button
if st.sidebar.button("🚀 Run Training"):
    with st.spinner(f"Running training on {dataset_choice} with {model_choice}..."):
        cmd = f"python fedavg_sim.py --dataset {dataset_choice} --model '{model_choice}' --rounds {num_rounds}"
        os.system(cmd)

    st.success("✅ Training Completed!")

# --- Results display ---
st.subheader("📊 Results")
results_files = [f for f in os.listdir("results") if f.endswith("_accuracy.csv")]

if results_files:
    latest_accuracy_file = max(results_files, key=lambda x: os.path.getctime(os.path.join("results", x)))
    acc_df = pd.read_csv(os.path.join("results", latest_accuracy_file))

    latest_epsilon_file = latest_accuracy_file.replace("_accuracy.csv", "_epsilon.csv")
    eps_df = pd.read_csv(os.path.join("results", latest_epsilon_file))

    col1, col2 = st.columns(2)

    with col1:
        st.line_chart(acc_df.set_index("round")["accuracy"])

    with col2:
        st.line_chart(eps_df.set_index("round")["epsilon"])
else:
    st.info("No training results yet. Run training to see charts.")
