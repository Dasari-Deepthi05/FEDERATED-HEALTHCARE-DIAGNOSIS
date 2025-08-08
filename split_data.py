# split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("health_data.csv")

# Shuffle and split
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
client1, temp = train_test_split(df, test_size=2/3, random_state=42)
client2, client3 = train_test_split(temp, test_size=0.5, random_state=42)

# Save splits
client1.to_csv("client1.csv", index=False)
client2.to_csv("client2.csv", index=False)
client3.to_csv("client3.csv", index=False)

print("✅ Split complete: client1.csv, client2.csv, client3.csv created.")
