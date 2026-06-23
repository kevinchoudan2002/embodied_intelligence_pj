import json
import numpy as np
import pandas as pd
from pathlib import Path

dataset_path = Path("/remote-home/wukehao/datasets/calvin_merge_ABC")

print("Reading parquet files...")
parquet_files = list(dataset_path.glob("data/chunk-*/file-*.parquet"))
print(f"Found {len(parquet_files)} files")

all_states = []
all_actions = []
sample_count = 0

for pf in parquet_files:
    df = pd.read_parquet(pf)
    sample_count += len(df)
    
    if "observation.state" in df.columns:
        states = np.array(df["observation.state"].tolist())
        all_states.append(states)
    if "action" in df.columns:
        actions = np.array(df["action"].tolist())
        all_actions.append(actions)

if all_states:
    all_states = np.concatenate(all_states, axis=0)
if all_actions:
    all_actions = np.concatenate(all_actions, axis=0)

def get_stats(arr):
    if arr is None or len(arr) == 0:
        return {"min": [0.0], "max": [0.0], "mean": [0.0], "std": [0.0], "count": [0]}
    
    # 如果是多维数组，按列计算
    if arr.ndim > 1:
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [len(arr)]
        }
    else:
        return {
            "min": [float(arr.min())],
            "max": [float(arr.max())],
            "mean": [float(arr.mean())],
            "std": [float(arr.std())],
            "count": [len(arr)]
        }

# 构建 stats.json
stats = {
    "observation.image": {
        "min": [0.0],
        "max": [255.0],
        "mean": [127.5],
        "std": [74.0],
        "count": [sample_count]
    },
    "observation.wrist_image": {
        "min": [0.0],
        "max": [255.0],
        "mean": [127.5],
        "std": [74.0],
        "count": [sample_count]
    }
}

if len(all_states) > 0:
    stats["observation.state"] = get_stats(all_states)

if len(all_actions) > 0:
    stats["action"] = get_stats(all_actions)

with open(dataset_path / "meta" / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"stats.json Generated.")
print(f"   observation.state: {stats.get('observation.state', {}).get('count', [0])[0]} samples")
print(f"   action: {stats.get('action', {}).get('count', [0])[0]} samples")