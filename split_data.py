# split_data.py
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV")
parser.add_argument("--num_clients", type=int, required=True, help="Number of clients to split into")
parser.add_argument("--mode", type=str, choices=["iid", "non-iid"], default="iid", help="Data split mode")
args = parser.parse_args()

# Load dataset from provided path
if not os.path.exists(args.dataset):
    raise FileNotFoundError(f"❌ Dataset not found: {args.dataset}")

df = pd.read_csv(args.dataset)

# Simple IID split
if args.mode == "iid":
    splits = [df.sample(frac=1/args.num_clients, random_state=i) for i in range(args.num_clients)]
else:
    # Non-IID: sort by Outcome, then split
    df_sorted = df.sort_values(by=df.columns[-1])  # assuming label is last column
    splits = [df_sorted.iloc[i::args.num_clients] for i in range(args.num_clients)]

# Save splits in /data/<dataset_name>/
dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
save_dir = os.path.join("data", dataset_name)
os.makedirs(save_dir, exist_ok=True)

for i, split_df in enumerate(splits, start=1):
    split_path = os.path.join(save_dir, f"client{i}.csv")
    split_df.to_csv(split_path, index=False)

print(f"✅ Data split completed! Saved in {save_dir}")
