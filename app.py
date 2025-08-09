import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess

st.set_page_config(page_title="Federated Healthcare Diagnosis", layout="centered")

st.title("🏥 Federated Learning Dashboard")
st.write("Track **Accuracy** and **Privacy (Epsilon)** over training rounds.")

def run_training():
    with st.spinner("🚀 Running federated training... please wait..."):
        result = subprocess.run(
            ["python", "-u", "fedavg_sim.py"],  # -u = unbuffered
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
    full_log = (result.stdout or "") + "\n" + (result.stderr or "")
    return full_log.strip()

# Training button
if st.button("▶ Run Training"):
    log = run_training()
    st.success("✅ Training completed!")
    st.text_area("📜 Training Log", log if log else "No output", height=300)

# Load results if available
acc_file = "results/fedavg_accuracy.csv"
eps_file = "results/epsilon_per_round.csv"

if os.path.exists(acc_file) and os.path.exists(eps_file):
    df_acc = pd.read_csv(acc_file)
    df_eps = pd.read_csv(eps_file)

    # Accuracy chart
    st.subheader("📈 Accuracy over Rounds")
    fig1, ax1 = plt.subplots()
    ax1.plot(df_acc['round'], df_acc['accuracy'], marker='o', color='blue')
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True)
    st.pyplot(fig1)

    # Epsilon chart
    st.subheader("🔐 Privacy Loss (Epsilon) over Rounds")
    fig2, ax2 = plt.subplots()
    ax2.plot(df_eps['round'], df_eps['epsilon'], marker='o', color='red')
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Epsilon")
    ax2.grid(True)
    st.pyplot(fig2)
else:
    st.warning("⚠️ Results files not found. Click **Run Training** to generate them.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, PyTorch, and Opacus")
