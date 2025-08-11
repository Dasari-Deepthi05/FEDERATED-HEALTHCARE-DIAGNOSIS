import streamlit as st
import pandas as pd
import subprocess
import glob
import os

st.set_page_config(page_title="Federated Healthcare Training Dashboard", layout="wide")
st.title("🏥 Federated Learning for Privacy-Preserving Healthcare Diagnosis")

# ==========================
# Utility: Find latest CSV
# ==========================
def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

# ==========================
# Sidebar Parameters
# ==========================
st.sidebar.header("⚙️ Training Parameters")
dataset = st.sidebar.text_input("Dataset (CSV)", "diabetes.csv")
rounds = st.sidebar.number_input("FedAvg Rounds", min_value=1, value=5)
epochs = st.sidebar.number_input("Local Epochs", min_value=1, value=3)
batch_size = st.sidebar.number_input("Batch Size", min_value=1, value=16)
epsilon = st.sidebar.number_input("Target Epsilon (DP)", min_value=0.1, value=5.0)
use_dp = st.sidebar.checkbox("Enable Differential Privacy", value=True)

# ==========================
# Run Training
# ==========================
if st.button("🚀 Run Training"):
    cmd = [
        "python", "fedavg_sim.py",
        "--dataset", dataset,
        "--rounds", str(rounds),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--epsilon", str(epsilon)
    ]
    if use_dp:
        cmd.append("--use_dp")

    with st.spinner("Training in progress..."):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = ""
        for line in process.stdout:
            log += line
            st.text(line.strip())
        process.wait()

    st.success("✅ Training Completed")

    # ==========================
    # Load Latest CSV Results
    # ==========================
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]
    acc_file = get_latest_file(f"results/{dataset_name}_*_accuracy.csv")
    eps_file = get_latest_file(f"results/{dataset_name}_*_epsilon.csv")

    if acc_file and eps_file:
        acc_df = pd.read_csv(acc_file)
        eps_df = pd.read_csv(eps_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Accuracy per Round")
            st.line_chart(acc_df.set_index("round"))

        with col2:
            st.subheader("🔐 Epsilon per Round (DP Privacy Cost)")
            st.line_chart(eps_df.set_index("round"))

        st.info(f"Results loaded from:\n- {acc_file}\n- {eps_file}")
    else:
        st.warning("⚠ No result files found. Run training first.")
