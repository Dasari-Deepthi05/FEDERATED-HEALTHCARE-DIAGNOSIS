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
- **Datasets:**
  - [Kaggle Diabetes CSV](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
  - Heart Disease Dataset
  - *(Additional healthcare datasets supported — place in `/datasets` folder)*

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

### **Day 5 – Multi-Dataset & Parameter Control**
- Added **dataset selector** in dashboard.
- Added **number of clients control** (2–10 hospitals).
- Split data in both **IID** and **Non-IID** modes.
- Implemented **timestamped result saving** in `/results`:
  - `accuracy_per_round_<timestamp>.csv`
  - `epsilon_per_round_<timestamp>.csv`
- Streamlit auto-loads latest results for chart display.
- UI now shows **selected dataset & training mode** in charts.

### **Day 6 – Model Selection Support**
- Added **model selector** in dashboard (MLP / Logistic Regression).
- Updated `fedavg_sim.py` to dynamically load chosen model.
- Results now stored separately for each dataset–model combination.
- Dashboard charts label runs with **dataset + model type**.
- Codebase now supports adding more models with minimal changes.

---

## 📊 Dashboard Features
- **Dataset Selector** → Choose from multiple healthcare datasets.
- **Model Selector** → Switch between MLP & Logistic Regression.
- **Clients Control** → Adjust number of simulated hospitals.
- **Mode Selector** → IID / Non-IID data partition.
- 📊 Accuracy vs. Rounds chart.
- 🔐 Epsilon vs. Rounds chart.
- ▶ One-click training run.
- 📜 Real-time training log output.

**Example Output Table:**
| Round | Accuracy | Epsilon |
|-------|----------|---------|
| 1     | 0.66     | 4.99    |
| 5     | 0.81     | 5.01    |

---

## 📂 Project Structure
```
FEDERATED-HEALTHCARE/
│
├── data/                          # Working datasets & client splits
│   ├── diabetes/
│   │   ├── client1.csv
│   │   ├── client2.csv
│   │   └── client3.csv
│   ├── heart/
│   │   ├── client1.csv
│   │   ├── client2.csv
│   │   └── client3.csv
│
├── datasets/                      # Original raw datasets
│   ├── diabetes.csv
│   └── heart.csv
│
├── results/                       # Accuracy & epsilon logs
│   ├── diabetes_2025-08-11_12-19-28_accuracy.csv
│   ├── diabetes_2025-08-11_12-19-28_epsilon.csv
│   ├── heart_2025-08-11_12-50-09_accuracy.csv
│   ├── heart_2025-08-11_12-50-09_epsilon.csv
│   ├── fedavg_accuracy.csv
│   └── epsilon_per_round.csv
│
├── app.py                         # Streamlit dashboard
├── central_train.py               # Baseline non-FL training
├── fedavg_sim.py                   # Federated learning simulation (DP-enabled)
├── split_data.py                   # Data splitting utility
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

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
