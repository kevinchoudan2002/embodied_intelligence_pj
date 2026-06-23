from pathlib import Path
import io
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path("datasets/calvin_task_ABC_D/splitA_old")  

def episode_path(root: Path, episode_index: int, chunks_size: int = 1000) -> Path:
    episode_chunk = episode_index // chunks_size
    return root / f"data/chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.parquet"

def read_episode_df(root: Path, episode_index: int, chunks_size: int = 1000, columns=None) -> pd.DataFrame:
    p = episode_path(root, episode_index, chunks_size)
    table = pq.read_table(str(p), columns=columns)
    df = table.to_pandas()
    return df

def decode_image_cell(cell):
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    arr = np.asarray(cell)
    if arr.dtype == np.uint8 and arr.ndim in (2,3):
        return Image.fromarray(arr)
    return None

df = read_episode_df(ROOT, episode_index=12)
print(df.columns)          
first_img = decode_image_cell(df['image'].iloc[0])
first_state = df['state'].iloc[0]    # numpy / list
first_action = df['actions'].iloc[0]