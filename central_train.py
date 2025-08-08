# central_train.py
import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("health_data.csv")
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Preprocess
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

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

model = DiabetesMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(50):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

# Evaluate
with torch.no_grad():
    predictions = model(X_test).argmax(dim=1)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

print(f"Baseline Accuracy: {acc:.2f}")
print("Confusion Matrix:")
print(cm)
