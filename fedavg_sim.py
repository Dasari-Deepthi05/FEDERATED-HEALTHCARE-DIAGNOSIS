import sys
# Force UTF-8 so emojis won't crash on Windows
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings("ignore", message="Secure RNG turned off")
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
warnings.filterwarnings("ignore", message="Full backward hook is firing")

import os
import pandas as pd
from opacus import PrivacyEngine
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Track history
epsilon_history = []
accuracy_history = []

# Model
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

def load_data(file):
    df = pd.read_csv(file)
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    X = StandardScaler().fit_transform(X)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train_local(model, X, y, epochs=3, use_dp=False,
                target_epsilon=5.0, target_delta=1e-5,
                batch_size=16, max_grad_norm=1.0):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    if use_dp:
        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
            sample_rate = batch_size / max(1, len(dataset))
            noise_multiplier = 1.0
            privacy_engine = PrivacyEngine(
                model,
                sample_rate=sample_rate,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
            privacy_engine.attach(optimizer)

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

        # FIX: return only weights without '_module.' prefix
        cleaned_state_dict = {
            k.replace("_module.", ""): v for k, v in model.state_dict().items()
        }
        return cleaned_state_dict, epsilon

    else:
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        return model.state_dict(), None

def average_weights(w_list):
    avg_w = {}
    for key in w_list[0].keys():
        avg_w[key] = sum([w[key] for w in w_list]) / len(w_list)
    return avg_w

# Prepare results folder
os.makedirs("results", exist_ok=True)

clients = ["client1.csv", "client2.csv", "client3.csv"]
X_test, y_test = load_data("client1.csv")

global_model = DiabetesMLP()

for round in range(5):
    weights = []
    epsilons = []
    for c in clients:
        model = DiabetesMLP()
        model.load_state_dict(global_model.state_dict())
        X, y = load_data(c)
        w, eps = train_local(model, X, y, use_dp=True)
        weights.append(w)
        if eps is not None:
            epsilons.append(eps)

    avg_weights = average_weights(weights)
    global_model.load_state_dict(avg_weights)

    with torch.no_grad():
        preds = global_model(X_test).argmax(dim=1)
        acc = accuracy_score(y_test, preds)

    accuracy_history.append(acc)
    epsilon_history.append(sum(epsilons) / len(epsilons) if epsilons else None)

    print(f"Round {round + 1}: Accuracy = {acc:.2f}, Epsilon = {epsilon_history[-1]:.4f}")

# Save results
pd.DataFrame({'round': range(1, len(accuracy_history)+1), 'accuracy': accuracy_history}) \
    .to_csv('results/fedavg_accuracy.csv', index=False)

pd.DataFrame({'round': range(1, len(epsilon_history)+1), 'epsilon': epsilon_history}) \
    .to_csv('results/epsilon_per_round.csv', index=False)

print("📊 Accuracy per round saved to results/fedavg_accuracy.csv")
print("🔐 Epsilon per round saved to results/epsilon_per_round.csv")
