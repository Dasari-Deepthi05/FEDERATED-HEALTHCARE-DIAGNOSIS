# fedavg_sim.py
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings("ignore", message="Secure RNG turned off")
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
warnings.filterwarnings("ignore", message="Full backward hook is firing")
import pandas as pd
from opacus import PrivacyEngine
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Track history
epsilon_history = []
accuracy_history = []

# -----------------------------
# Model
# -----------------------------
class DiabetesMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 2)
        )
    def forward(self, x):
        return self.layers(x)

# -----------------------------
# Load data
# -----------------------------
def load_data(file):
    df = pd.read_csv(file)
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    X = StandardScaler().fit_transform(X)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# -----------------------------
# Local training (with/without DP)
# -----------------------------
def train_local(model, X, y, epochs=3, use_dp=False,
                target_epsilon=5.0, target_delta=1e-5,
                batch_size=16, max_grad_norm=1.0):
    """
    Trains model on (X,y).
    If use_dp=True, uses Opacus to add DP (clipping + noise) and returns (state_dict, epsilon).
    Otherwise returns (state_dict, None).
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    if use_dp:
        # Create dataloader (Opacus expects a DataLoader)
        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Try using automatic epsilon-based privacy setting
        privacy_engine = PrivacyEngine()
        try:
            model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                epochs=epochs,
                max_grad_norm=max_grad_norm,
            )
        except Exception:
            # Manual attach fallback
            sample_rate = batch_size / max(1, len(dataset))
            noise_multiplier = 1.0  # can be tuned
            privacy_engine = PrivacyEngine(
                model,
                sample_rate=sample_rate,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
            privacy_engine.attach(optimizer)

        # Train with DP
        for _ in range(epochs):
            for batch_x, batch_y in data_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        try:
            epsilon = privacy_engine.get_epsilon(target_delta)
        except Exception:
            epsilon = None

        # ---- FIX: remove "_module." prefix so weights load cleanly ----
        state_dict = model.state_dict()
        clean_state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}

        return clean_state_dict, epsilon

    else:
        # Normal training without DP
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        return model.state_dict(), None

# -----------------------------
# FedAvg aggregation
# -----------------------------
def average_weights(w_list):
    avg_w = {}
    for key in w_list[0].keys():
        avg_w[key] = sum([w[key] for w in w_list]) / len(w_list)
    return avg_w

# -----------------------------
# Simulation
# -----------------------------
clients = ["client1.csv", "client2.csv", "client3.csv"]
X_test, y_test = load_data("client1.csv")  # Replace with separate test set if available
global_model = DiabetesMLP()

for rnd in range(5):
    weights = []
    eps_this_round = []

    for c in clients:
        model = DiabetesMLP()
        model.load_state_dict(global_model.state_dict())
        X, y = load_data(c)

        state_dict, eps = train_local(model, X, y, use_dp=True)
        weights.append(state_dict)
        eps_this_round.append(eps)

    # Aggregate weights
    avg_weights = average_weights(weights)
    global_model.load_state_dict(avg_weights)

    # Evaluate on test data
    with torch.no_grad():
        preds = global_model(X_test).argmax(dim=1)
        acc = accuracy_score(y_test, preds)

    accuracy_history.append(acc)
    epsilon_history.append(max(eps_this_round))  # track max epsilon per round

    print(f"Round {rnd + 1}: Accuracy = {acc:.2f}, Epsilon = {epsilon_history[-1]}")

# Save accuracy results
df = pd.DataFrame({'round': list(range(1, len(accuracy_history)+1)), 'accuracy': accuracy_history})
df.to_csv('results/fedavg_accuracy.csv', index=False)
print("📊 Accuracy per round saved to results/fedavg_accuracy.csv")

# Save epsilon results
df_eps = pd.DataFrame({'round': list(range(1, len(epsilon_history)+1)), 'epsilon': epsilon_history})
df_eps.to_csv('results/epsilon_per_round.csv', index=False)
print("🔐 Epsilon per round saved to results/epsilon_per_round.csv")
