# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess

st.set_page_config(page_title="Federated Healthcare Diagnosis", layout="centered")

st.title("🏥 Federated Learning Dashboard")
st.write("Track **Accuracy** and **Privacy (Epsilon)** over training rounds.")

# Function to run training script
def run_training():
    with st.spinner("🚀 Running federated training... please wait..."):
        result = subprocess.run(
            ["python", "fedavg_sim.py"],
            capture_output=True,
            text=True
        )
    return result.stdout, result.stderr

# Run training button
if st.button("▶ Run Training"):
    out, err = run_training()
    if err:
        st.error(f"❌ Error during training:\n{err}")
    else:
        st.success("✅ Training completed!")
        st.text_area("📜 Training Log", out, height=200)

# Load results
acc_file = "results/fedavg_accuracy.csv"
eps_file = "results/epsilon_per_round.csv"

if os.path.exists(acc_file) and os.path.exists(eps_file):
    df_acc = pd.read_csv(acc_file)
    df_eps = pd.read_csv(eps_file)

    # ---- Chart 1: Accuracy ----
    st.subheader("📈 Accuracy over Rounds")
    fig1, ax1 = plt.subplots()
    ax1.plot(df_acc['round'], df_acc['accuracy'], marker='o', color='blue')
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True)
    st.pyplot(fig1)

    # ---- Chart 2: Epsilon ----
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
