from pathlib import Path

import numpy as np
from PIL import Image


USE_ARGS_VALUE = object()


def load_image_as_vector(image_path, image_size):
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def load_eurosat_dataset(root_dir, image_size=32, max_per_class=None):
    root = Path(root_dir)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found in: {root}")

    class_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}
    vectors = []
    labels = []
    paths = []

    for class_dir in class_dirs:
        image_files = sorted(class_dir.glob("*.jpg"))
        if max_per_class is not None:
            image_files = image_files[:max_per_class]

        for image_path in image_files:
            vectors.append(load_image_as_vector(image_path, image_size))
            labels.append(class_to_idx[class_dir.name])
            paths.append(str(image_path))

    x = np.vstack(vectors).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return x, y, class_to_idx, np.array(paths, dtype=object)


def train_val_test_split(x, y, paths=None, val_ratio=0.2, test_ratio=0.1, seed=42):
    if val_ratio < 0.0 or test_ratio < 0.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >=0 and their sum < 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)
    n_total = x.shape[0]
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    if paths is None:
        return x[train_idx], y[train_idx], x[val_idx], y[val_idx], x[test_idx], y[test_idx]

    paths = np.asarray(paths, dtype=object)
    return (
        x[train_idx],
        y[train_idx],
        paths[train_idx],
        x[val_idx],
        y[val_idx],
        paths[val_idx],
        x[test_idx],
        y[test_idx],
        paths[test_idx],
    )


def iterate_minibatches(x, y, batch_size, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)
    for start in range(0, x.shape[0], batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]


def prepare_data(
    args,
    forced_mean=None,
    forced_std=None,
    override_image_size=None,
    override_val_ratio=None,
    override_test_ratio=None,
    override_seed=None,
    override_max_per_class=USE_ARGS_VALUE,
    return_paths=False,
):
    image_size = args.image_size if override_image_size is None else override_image_size
    val_ratio = args.val_ratio if override_val_ratio is None else override_val_ratio
    test_ratio = args.test_ratio if override_test_ratio is None else override_test_ratio
    seed = args.seed if override_seed is None else override_seed
    max_per_class = args.max_per_class if override_max_per_class is USE_ARGS_VALUE else override_max_per_class

    x, y, class_to_idx, paths = load_eurosat_dataset(
        root_dir=args.data_dir,
        image_size=image_size,
        max_per_class=max_per_class,
    )

    split_result = train_val_test_split(
        x,
        y,
        paths=paths if return_paths else None,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    if return_paths:
        x_train, y_train, train_paths, x_val, y_val, val_paths, x_test, y_test, test_paths = split_result
    else:
        x_train, y_train, x_val, y_val, x_test, y_test = split_result

    train_mean = np.mean(x_train, axis=0, keepdims=True) if forced_mean is None else forced_mean
    train_std = np.std(x_train, axis=0, keepdims=True) + 1e-8 if forced_std is None else forced_std
    x_train = (x_train - train_mean) / train_std
    x_val = (x_val - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    if return_paths:
        return (
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            class_to_idx,
            train_mean,
            train_std,
            train_paths,
            val_paths,
            test_paths,
        )

    return x_train, y_train, x_val, y_val, x_test, y_test, class_to_idx, train_mean, train_std
