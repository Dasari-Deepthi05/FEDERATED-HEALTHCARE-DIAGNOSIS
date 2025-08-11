import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Dataset filename (inside datasets/ folder)")
parser.add_argument("--num_clients", type=int, default=3)
parser.add_argument("--mode", type=str, choices=["iid", "non_iid"], default="iid")
parser.add_argument("--major_share", type=float, default=0.7)
args = parser.parse_args()

dataset_path = os.path.join("datasets", args.dataset)
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset {args.dataset} not found in datasets/ folder")

df = pd.read_csv(dataset_path)
print(f"Loaded {dataset_path} with {len(df)} rows")

# Splitting logic stays the same, but save in:
save_dir = os.path.join("data", args.dataset.split(".")[0])
os.makedirs(save_dir, exist_ok=True)

# Example saving:
for i in range(args.num_clients):
    client_data = df.sample(frac=1/args.num_clients, random_state=i)
    client_data.to_csv(os.path.join(save_dir, f"client{i+1}.csv"), index=False)
