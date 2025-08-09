# 🏥 Federated Learning for Privacy-Preserving Healthcare Diagnosis

This project simulates **multiple hospitals** collaboratively training a diagnostic model **without sharing raw patient data**.  
It implements **Federated Averaging (FedAvg)** and **Differential Privacy (DP)** to ensure **data privacy** while maintaining model performance.

---

## 📌 Objective
To demonstrate how **Federated Learning** combined with **Differential Privacy** can be used in healthcare AI to protect sensitive patient data while enabling collaborative model training.

---

## 🛠 Tech Stack
- **Python**, **PyTorch**, **Flower** (FL orchestration)
- **Opacus** → Differential Privacy
- **Streamlit** → Interactive dashboard
- **Pandas**, **scikit-learn**, **matplotlib** → Data processing & visualization
- **Dataset:** [Kaggle Diabetes CSV](https://www.kaggle.com/datasets/mathchi/diabetes-data-set) *(simulated EHR data)*

---

## 📅 Progress Timeline

### **Day 1 – Project Setup**
- Created GitHub repository & README.
- Selected dataset (Kaggle Diabetes CSV).
- Implemented **centralized training** for baseline accuracy.
- Split dataset into 3 client datasets (simulated hospitals).
- Implemented minimal **Federated Averaging** loop.

### **Day 2 – Basic Streamlit Dashboard**
- Added dashboard to show dataset splits.
- Added "Run 1 Round" button.
- Displayed accuracy progress over rounds.

### **Day 3 – Differential Privacy Integration**
- Integrated **Opacus** for privacy-preserving training.
- Configured epsilon & delta parameters.
- Fixed `_module.` key mismatch in DP model weights.
- Saved accuracy results to `/results/fedavg_accuracy.csv`.

### **Day 4 – Interactive Dashboard with Privacy Tracking**
- Added **epsilon tracking** per training round.
- Stored results in `/results` folder (`accuracy_per_round.csv`, `epsilon_per_round.csv`).
- Added **dual charts** in Streamlit:
  - 📊 Accuracy per round
  - 🔐 Epsilon per round
- Added **"Run Training"** button to start full simulation from UI.
- Captured and displayed **live training logs** inside dashboard.
- Forced UTF-8 output for emoji/log display on Windows.
- Suppressed unnecessary Opacus warnings for cleaner output.

---

## 📊 Dashboard Preview
📊 Accuracy vs. Rounds
🔐 Epsilon vs. Rounds
▶ Run Training
📜 Training Log

**Example Output Table:**
| Round | Accuracy | Epsilon |
|-------|----------|---------|
| 1     | 0.66     | 4.99    |
| 5     | 0.81     | 5.01    |

---

## 📂 Project Structure
FEDERATED-HEALTHCARE/
│
├── datasets/ # Simulated client datasets
├── results/ # Accuracy & epsilon CSVs
├── logs/ # Saved training logs
│
├── central_train.py # Baseline (non-FL) training
├── split_data.py # Data partitioning into clients
├── fedavg_sim.py # Federated training simulation with DP
├── app.py # Streamlit dashboard
│
└── README.md

---

## ▶ Running the Project

### **1. Install dependencies**
```bash
pip install pandas scikit-learn torch opacus streamlit matplotlib
streamlit run app.py
```

---

Train the model (interactive mode)

- Select dataset in dashboard.
- Click "Run Training".
- View accuracy and epsilon charts in real time.

---
