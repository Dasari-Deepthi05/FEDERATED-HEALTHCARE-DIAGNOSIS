import argparse
import os
import pandas as pd
from opacus import PrivacyEngine
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime

# --- Argument parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (diabetes or heart)")
parser.add_argument("--model", type=str, required=True, help="Model type (MLP or Logistic Regression)")
parser.add_argument("--rounds", type=int, default=5, help="Number of training rounds")
args = parser.parse_args()

# --- Models ---
class MLPModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 2)
        )
    def forward(self, x): return self.layers(x)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 2)
    def forward(self, x): return self.linear(x)

# --- Load data ---
def load_data(file):
    df = pd.read_csv(file)
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    X = StandardScaler().fit_transform(X)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# --- Training function ---
def train_local(model, X, y, epochs=3, use_dp=True,
                target_epsilon=5.0, target_delta=1e-5,
                batch_size=16, max_grad_norm=1.0):

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )

    for _ in range(epochs):
        for batch_x, batch_y in data_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    epsilon = None
    if use_dp:
        epsilon = privacy_engine.get_epsilon(target_delta)

    return model.state_dict(), epsilon

# --- Federated averaging ---
def average_weights(w_list):
    avg_w = {}
    for key in w_list[0].keys():
        avg_w[key] = sum([w[key] for w in w_list]) / len(w_list)
    return avg_w

# --- Main execution ---
client_folder = f"data/{args.dataset}"
clients = [os.path.join(client_folder, f"client{i}.csv") for i in range(1, 4)]

X_test, y_test = load_data(clients[0])

input_size = X_test.shape[1]
if args.model == "MLP":
    global_model = MLPModel(input_size)
else:
    global_model = LogisticRegressionModel(input_size)

accuracy_history = []
epsilon_history = []

for round in range(args.rounds):
    weights = []
    epsilons = []

    for c in clients:
        if args.model == "MLP":
            model = MLPModel(input_size)
        else:
            model = LogisticRegressionModel(input_size)

        model.load_state_dict(global_model.state_dict())
        X, y = load_data(c)
        w, eps = train_local(model, X, y)
        weights.append(w)
        if eps: epsilons.append(eps)

    avg_weights = average_weights(weights)
    global_model.load_state_dict(avg_weights)

    with torch.no_grad():
        preds = global_model(X_test).argmax(dim=1)
        acc = accuracy_score(y_test, preds)

    accuracy_history.append(acc)
    epsilon_history.append(sum(epsilons)/len(epsilons) if epsilons else None)

# --- Save results ---
os.makedirs("results", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
acc_df = pd.DataFrame({'round': list(range(1, args.rounds+1)), 'accuracy': accuracy_history})
eps_df = pd.DataFrame({'round': list(range(1, args.rounds+1)), 'epsilon': epsilon_history})

acc_df.to_csv(f"results/{args.dataset}_{timestamp}_accuracy.csv", index=False)
eps_df.to_csv(f"results/{args.dataset}_{timestamp}_epsilon.csv", index=False)

print(f"✅ Training completed for {args.dataset} with {args.model}")
