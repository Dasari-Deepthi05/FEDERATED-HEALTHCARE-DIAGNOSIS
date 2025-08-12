import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
from opacus import PrivacyEngine
import os
from datetime import datetime

# ----------------------------
# Model definition
# ----------------------------
class DiabetesMLP(nn.Module):
    def __init__(self, input_size):
        super(DiabetesMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)


# ----------------------------
# Helper functions
# ----------------------------
def load_client_data(client_path):
    df = pd.read_csv(client_path)
    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    return train_dataset, test_dataset


def train_local(model, train_dataset, epochs=1, lr=0.01, dp=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    if dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    if dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
        return model.state_dict(), epsilon
    else:
        return model.state_dict(), None


def test_model(model, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def average_weights(w_list):
    avg_w = {}
    for k in w_list[0].keys():
        avg_w[k] = sum([w[k] for w in w_list]) / len(w_list)
    return avg_w


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., diabetes)")
    parser.add_argument("--num_clients", type=int, required=True, help="Number of clients")
    args = parser.parse_args()

    dataset_name = args.dataset
    num_clients = args.num_clients

    data_dir = os.path.join("data", dataset_name)
    clients = [f"client{i+1}.csv" for i in range(num_clients)]

    # Load data
    client_train_data = []
    client_test_data = []
    for c in clients:
        train_ds, test_ds = load_client_data(os.path.join(data_dir, c))
        client_train_data.append(train_ds)
        client_test_data.append(test_ds)

    input_size = client_train_data[0][0][0].shape[0]
    global_model = DiabetesMLP(input_size=input_size)

    # FL Simulation
    accuracy_list = []
    epsilon_list = []
    rounds = 5

    for r in range(rounds):
        local_weights = []
        local_epsilons = []

        for i in range(num_clients):
            local_model = DiabetesMLP(input_size=input_size)
            local_model.load_state_dict(global_model.state_dict())
            w, eps = train_local(local_model, client_train_data[i], epochs=1, dp=True)
            local_weights.append(w)
            if eps is not None:
                local_epsilons.append(eps)

        avg_weights = average_weights(local_weights)

        # ✅ FIX: Remove _module. prefix from keys before loading
        cleaned_weights = {k.replace("_module.", ""): v for k, v in avg_weights.items()}
        global_model.load_state_dict(cleaned_weights)

        acc = np.mean([test_model(global_model, client_test_data[i]) for i in range(num_clients)])
        eps_avg = np.mean(local_epsilons) if local_epsilons else 0.0

        accuracy_list.append(acc)
        epsilon_list.append(eps_avg)

        print(f"Round {r+1}: Accuracy={acc:.2f}, Epsilon={eps_avg:.4f}")

    # Save results with timestamp
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    acc_file = os.path.join("results", f"{dataset_name}_{timestamp}_accuracy.csv")
    eps_file = os.path.join("results", f"{dataset_name}_{timestamp}_epsilon.csv")

    pd.DataFrame({"round": list(range(1, rounds+1)), "accuracy": accuracy_list}).to_csv(acc_file, index=False)
    pd.DataFrame({"round": list(range(1, rounds+1)), "epsilon": epsilon_list}).to_csv(eps_file, index=False)

    # ✅ Windows-safe printing (no emoji)
    print(f"[INFO] Accuracy saved to {acc_file}")
    print(f"[INFO] Epsilon saved to {eps_file}")
