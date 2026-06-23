import pandas as pd
from pathlib import Path

dataset_path = Path("/remote-home/wukehao/datasets/calvin_env_ABC")

parquet_files = list(dataset_path.glob("data/chunk-*/file-*.parquet"))
print(f"{len(parquet_files)} files found")

for pf in parquet_files:
    df = pd.read_parquet(pf)

    rename_map = {
        "image": "observation.image",
        "wrist_image": "observation.wrist_image", 
        "state": "observation.state",
        "actions": "action"
    }
    df = df.rename(columns=rename_map)
    
    df.to_parquet(pf, index=False)

print("Done.")