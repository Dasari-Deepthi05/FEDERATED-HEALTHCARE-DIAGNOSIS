# fedavg_sim.py
import sys
# ensure UTF-8 output on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# optional: suppress repetitive Opacus warnings while keeping others visible
import warnings
warnings.filterwarnings("ignore", message="Secure RNG turned off")
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
warnings.filterwarnings("ignore", message="Full backward hook is firing")

import os
import glob
import argparse
from datetime import datetime
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Opacus (for DP)
from opacus import PrivacyEngine

# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser(description="Federated Simulation (FedAvg) with optional DP")
parser.add_argument("--dataset", type=str, default="diabetes.csv",
                    help="Dataset filename in datasets/ or dataset name (e.g., 'diabetes' or 'diabetes.csv'). "
                         "Assumes client splits are in data/<dataset_name>/client*.csv")
parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
parser.add_argument("--epochs", type=int, default=3, help="Local epochs per client")
parser.add_argument("--batch_size", type=int, default=16, help="Local batch size for DP/data loader")
parser.add_argument("--epsilon", type=float, default=5.0, help="Target epsilon for DP (if DP enabled)")
parser.add_argument("--delta", type=float, default=1e-5, help="Target delta for DP accounting")
parser.add_argument("--use_dp", action="store_true", help="Enable Differential Privacy (Opacus) on clients")
args = parser.parse_args()

# Normalize dataset name (strip extension)
dataset_name_raw = args.dataset
dataset_name = os.path.splitext(os.path.basename(dataset_name_raw))[0]

num_rounds = args.rounds
local_epochs = args.epochs
batch_size = args.batch_size
target_epsilon = args.epsilon
target_delta = args.delta
use_dp = args.use_dp

print(f"Dataset: {dataset_name} | Rounds: {num_rounds} | Epochs/client: {local_epochs} | Batch size: {batch_size} | DP: {use_dp} | ε={target_epsilon}, δ={target_delta}")

# -------------------------
# Utils & Model
# -------------------------
class DiabetesMLP(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
    def forward(self, x):
        return self.layers(x)

def load_tabular(file):
    """Load CSV, assume label column named 'Outcome' (matching split_data.py behavior).
       Returns (X_tensor, y_tensor)."""
    df = pd.read_csv(file)
    if 'Outcome' not in df.columns:
        # try common alternatives (safe fallback)
        for alt in ['target', 'label', 'class', 'Class']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'Outcome'})
                break
    # drop any non-feature columns if present (keep Outcome as label)
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    X = StandardScaler().fit_transform(X)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def average_weights(state_dicts):
    """FedAvg simple average across client state_dicts (assumes same keys)."""
    avg = {}
    for k in state_dicts[0].keys():
        avg[k] = sum([sd[k] for sd in state_dicts]) / len(state_dicts)
    return avg

def train_local(model, X, y, epochs=3, batch_size=16, use_dp=False, target_epsilon=5.0, target_delta=1e-5, max_grad_norm=1.0):
    """
    Train model on local data X,y.
    If use_dp=True, attempts to make the training private using Opacus.
    Returns: (clean_state_dict, epsilon_or_None)
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    if use_dp:
        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        privacy_engine = PrivacyEngine()
        try:
            # try to pick noise multiplier to achieve target_epsilon
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
            # fallback: attach privacy engine manually with chosen noise multiplier
            sample_rate = batch_size / max(1, len(dataset))
            # default noise - can be tuned by user if make_private_with_epsilon fails
            noise_multiplier = 1.0
            privacy_engine = PrivacyEngine(
                model,
                sample_rate=sample_rate,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
            privacy_engine.attach(optimizer)

        # training loop using data_loader
        for _ in range(epochs):
            for batch_x, batch_y in data_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        # get epsilon; may raise, so guard it
        try:
            epsilon = privacy_engine.get_epsilon(target_delta)
        except Exception:
            epsilon = None

        # clean keys if Opacus wrapped model with "_module."
        raw_state = model.state_dict()
        clean_state = {k.replace("_module.", ""): v for k, v in raw_state.items()}
        return clean_state, epsilon

    else:
        # non-DP training (full-batch)
        # Use a DataLoader for batching to keep behaviour consistent (optional)
        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for batch_x, batch_y in data_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
        return model.state_dict(), None

# -------------------------
# Prepare client file list
# -------------------------
clients_dir = os.path.join("data", dataset_name)
if not os.path.isdir(clients_dir):
    raise SystemExit(f"Client data directory not found: {clients_dir}\nRun split_data.py --dataset {dataset_name}.csv first.")

client_files = sorted(glob.glob(os.path.join(clients_dir, "client*.csv")))
if len(client_files) == 0:
    raise SystemExit(f"No client files found in {clients_dir}. Expected files like client1.csv ...")

print(f"Found {len(client_files)} client files: {client_files}")

# -------------------------
# Simulation (FedAvg)
# -------------------------
os.makedirs("results", exist_ok=True)

# pick one client as test set OR you can implement separate global test; for now use client1 as proxy
X_test, y_test = load_tabular(client_files[0])

# infer input dimension from test
input_dim = X_test.shape[1]
global_model = DiabetesMLP(input_dim=input_dim)

accuracy_history = []
epsilon_history = []

for rnd in range(num_rounds):
    state_list = []
    eps_list = []

    for cfile in client_files:
        # initialize local model from global
        local_model = DiabetesMLP(input_dim=input_dim)
        local_model.load_state_dict(global_model.state_dict())

        X_local, y_local = load_tabular(cfile)

        sd, eps = train_local(local_model,
                              X_local, y_local,
                              epochs=local_epochs,
                              batch_size=batch_size,
                              use_dp=use_dp,
                              target_epsilon=target_epsilon,
                              target_delta=target_delta)

        state_list.append(sd)
        if eps is not None:
            eps_list.append(eps)

    # FedAvg aggregation
    avg_sd = average_weights(state_list)
    global_model.load_state_dict(avg_sd)

    # Evaluate global model on X_test
    global_model.eval()
    with torch.no_grad():
        preds = global_model(X_test).argmax(dim=1)
        acc = accuracy_score(y_test.numpy(), preds.numpy())
    accuracy_history.append(float(acc))
    # choose aggregated epsilon metric (mean across clients that provided value)
    if len(eps_list) > 0:
        epsilon_history.append(float(sum(eps_list) / len(eps_list)))
    else:
        epsilon_history.append(None)

    print(f"[Round {rnd+1}/{num_rounds}] Accuracy: {acc:.4f} | Epsilon (avg clients): {epsilon_history[-1]}")

# -------------------------
# Save results (timestamped)
# -------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
acc_path = os.path.join("results", f"{dataset_name}_{timestamp}_accuracy.csv")
eps_path = os.path.join("results", f"{dataset_name}_{timestamp}_epsilon.csv")

pd.DataFrame({'round': list(range(1, len(accuracy_history)+1)), 'accuracy': accuracy_history}) \
    .to_csv(acc_path, index=False)

pd.DataFrame({'round': list(range(1, len(epsilon_history)+1)), 'epsilon': epsilon_history}) \
    .to_csv(eps_path, index=False)

print(f"Saved accuracy -> {acc_path}")
print(f"Saved epsilon  -> {eps_path}")
print("Done.")
