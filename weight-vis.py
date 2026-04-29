import numpy as np
from PIL import Image
import os

def normalize_to_uint8(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - arr_min) / (arr_max - arr_min)
    return (norm * 255).astype(np.uint8)

ckpt = np.load("best_model.npz", allow_pickle=False)
W1 = ckpt["W1"]  # shape: (D, hidden_dim)
image_size = int(ckpt["image_size"])
hidden_dim = W1.shape[1]

num_show = min(64, hidden_dim)
cols = 8
rows = int(np.ceil(num_show / cols))
grid = Image.new("RGB", (cols * image_size, rows * image_size))

out_dir = "w1_filters"
os.makedirs(out_dir, exist_ok=True)

for i in range(num_show):
    w = W1[:, i].reshape(image_size, image_size, 3)
    img = Image.fromarray(normalize_to_uint8(w))
    r = i // cols
    c = i % cols
    grid.paste(img, (c * image_size, r * image_size))


grid.save("w1_filters_grid.png")
print("Saved to w1_filters_grid.png")