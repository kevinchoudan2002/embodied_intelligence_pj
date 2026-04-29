import itertools

import numpy as np

from training import run_training
from utils import parse_float_list, parse_int_list, save_search_results


def run_grid_search(args, x_train, y_train, x_val, y_val, num_classes):
    lr_values = parse_float_list(args.grid_lrs)
    hidden_values = parse_int_list(args.grid_hidden_dims)
    wd_values = parse_float_list(args.grid_weight_decays)

    combinations = list(itertools.product(lr_values, hidden_values, wd_values))
    if not combinations:
        raise ValueError("Grid search has no parameter combinations to evaluate.")

    best = None
    rows = []
    for idx, (lr, hidden_dim, weight_decay) in enumerate(combinations, start=1):
        print(f"\n[Grid {idx}/{len(combinations)}] lr={lr}, hidden_dim={hidden_dim}, weight_decay={weight_decay}")
        _, metrics, _, _, _ = run_training(
            args=args,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            lr=lr,
            weight_decay=weight_decay,
            epochs=args.search_epochs,
            seed=args.seed + idx,
            verbose=False,
        )
        print(
            f"Result: val_acc={metrics['val_acc']:.4f}, val_loss={metrics['val_loss']:.4f}, "
            f"train_acc={metrics['train_acc']:.4f}, train_loss={metrics['train_loss']:.4f}"
        )

        candidate = {
            "lr": lr,
            "hidden_dim": hidden_dim,
            "weight_decay": weight_decay,
            "metrics": metrics,
        }
        rows.append(
            {
                "search_mode": "grid",
                "trial": idx,
                "lr": lr,
                "hidden_dim": hidden_dim,
                "weight_decay": weight_decay,
                "epoch": metrics["epoch"],
                "effective_lr": metrics["lr"],
                "train_loss": metrics["train_loss"],
                "train_acc": metrics["train_acc"],
                "val_loss": metrics["val_loss"],
                "val_acc": metrics["val_acc"],
            }
        )
        if best is None or candidate["metrics"]["val_acc"] > best["metrics"]["val_acc"]:
            best = candidate

    print("\nBest grid-search config:")
    print(
        f"lr={best['lr']}, hidden_dim={best['hidden_dim']}, weight_decay={best['weight_decay']} | "
        f"val_acc={best['metrics']['val_acc']:.4f}, val_loss={best['metrics']['val_loss']:.4f}"
    )

    save_search_results(rows, args.results_csv)


def run_random_search(args, x_train, y_train, x_val, y_val, num_classes):
    rng = np.random.default_rng(args.seed)
    best = None
    rows = []

    for idx in range(1, args.num_trials + 1):
        lr = float(np.exp(rng.uniform(np.log(args.random_lr_min), np.log(args.random_lr_max))))
        hidden_dim = int(rng.integers(args.random_hidden_min, args.random_hidden_max + 1))
        weight_decay = float(
            np.exp(rng.uniform(np.log(args.random_weight_decay_min), np.log(args.random_weight_decay_max)))
        )

        print(
            f"\n[Random {idx}/{args.num_trials}] "
            f"lr={lr:.6f}, hidden_dim={hidden_dim}, weight_decay={weight_decay:.6f}"
        )
        _, metrics, _, _, _ = run_training(
            args=args,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            lr=lr,
            weight_decay=weight_decay,
            epochs=args.search_epochs,
            seed=args.seed + idx,
            verbose=False,
        )
        print(
            f"Result: val_acc={metrics['val_acc']:.4f}, val_loss={metrics['val_loss']:.4f}, "
            f"train_acc={metrics['train_acc']:.4f}, train_loss={metrics['train_loss']:.4f}"
        )

        candidate = {
            "lr": lr,
            "hidden_dim": hidden_dim,
            "weight_decay": weight_decay,
            "metrics": metrics,
        }
        rows.append(
            {
                "search_mode": "random",
                "trial": idx,
                "lr": lr,
                "hidden_dim": hidden_dim,
                "weight_decay": weight_decay,
                "epoch": metrics["epoch"],
                "effective_lr": metrics["lr"],
                "train_loss": metrics["train_loss"],
                "train_acc": metrics["train_acc"],
                "val_loss": metrics["val_loss"],
                "val_acc": metrics["val_acc"],
            }
        )
        if best is None or candidate["metrics"]["val_acc"] > best["metrics"]["val_acc"]:
            best = candidate

    print("\nBest random-search config:")
    print(
        f"lr={best['lr']:.6f}, hidden_dim={best['hidden_dim']}, weight_decay={best['weight_decay']:.6f} | "
        f"val_acc={best['metrics']['val_acc']:.4f}, val_loss={best['metrics']['val_loss']:.4f}"
    )

    save_search_results(rows, args.results_csv)
