import streamlit as st
import pandas as pd
import subprocess
import os
import glob
from datetime import datetime

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="Federated Healthcare FL + DP", layout="wide")
st.title("🏥 Federated Learning for Privacy-Preserving Healthcare Diagnosis")

# -------------------------
# Dataset Selection
# -------------------------
DATASETS_DIR = "datasets"
available_datasets = [f.replace(".csv", "") for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")]

dataset_choice = st.selectbox("📂 Select Dataset", available_datasets)

# -------------------------
# Client Count
# -------------------------
num_clients = st.number_input("🏥 Number of Clients (Hospitals)", min_value=2, max_value=10, value=3, step=1)

# -------------------------
# Mode Selection
# -------------------------
mode_choice = st.radio("📊 Data Split Mode", ["iid", "non-iid"], index=0)

# -------------------------
# Run Training Button
# -------------------------
if st.button("▶ Run Training"):
    st.info("Splitting dataset...")
    try:
        cmd_split = ["python", "split_data.py", "--dataset", f"datasets/{dataset_choice}.csv",
                     "--num_clients", str(num_clients), "--mode", mode_choice]
        subprocess.run(cmd_split, check=True)

        st.success("✅ Data split completed!")

        st.info("Starting Federated Learning Training...")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cmd_train = ["python", "fedavg_sim.py", "--dataset", dataset_choice, "--num_clients", str(num_clients)]
        subprocess.run(cmd_train, check=True)

        st.success("✅ Training Completed!")

    except subprocess.CalledProcessError as e:
        st.error(f"❌ Error running command: {e}")

# -------------------------
# Load Latest Results
# -------------------------
def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

accuracy_file = get_latest_file(f"results/{dataset_choice}_*_accuracy.csv")
epsilon_file = get_latest_file(f"results/{dataset_choice}_*_epsilon.csv")

if accuracy_file and epsilon_file:
    st.subheader(f"📊 Accuracy & Privacy Results — {dataset_choice.capitalize()} Dataset")

    acc_df = pd.read_csv(accuracy_file)
    eps_df = pd.read_csv(epsilon_file)

    # Merge for display
    results_df = pd.DataFrame({
        "Round": acc_df.index + 1,
        "Accuracy": acc_df.iloc[:, 0],
        "Epsilon": eps_df.iloc[:, 0]
    })

    st.dataframe(results_df)

    col1, col2 = st.columns(2)

    with col1:
        st.line_chart(results_df[["Round", "Accuracy"]].set_index("Round"))

    with col2:
        st.line_chart(results_df[["Round", "Epsilon"]].set_index("Round"))

else:
    st.warning("⚠ No results found yet. Please run training first.")
