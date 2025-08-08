# fedavg_sim.py
import torch
from torch import nn, optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Model
class DiabetesMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 2)
        )
    def forward(self, x): return self.layers(x)

def load_data(file):
    df = pd.read_csv(file)
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    X = StandardScaler().fit_transform(X)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train_local(model, X, y, epochs=3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return model.state_dict()

def average_weights(w_list):
    avg_w = {}
    for key in w_list[0].keys():
        avg_w[key] = sum([w[key] for w in w_list]) / len(w_list)
    return avg_w

clients = ["client1.csv", "client2.csv", "client3.csv"]
X_test, y_test = load_data("client1.csv")

global_model = DiabetesMLP()

for round in range(5):
    weights = []
    for c in clients:
        model = DiabetesMLP()
        model.load_state_dict(global_model.state_dict())
        X, y = load_data(c)
        w = train_local(model, X, y)
        weights.append(w)
    avg_weights = average_weights(weights)
    global_model.load_state_dict(avg_weights)

    with torch.no_grad():
        preds = global_model(X_test).argmax(dim=1)
        acc = accuracy_score(y_test, preds)
    print(f"Round {round + 1}: Accuracy = {acc:.2f}")
