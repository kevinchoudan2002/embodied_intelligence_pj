import numpy as np

from encode_loss_calc import compute_loss_and_dlogits, l2_penalty
from mlp_model import accuracy


def build_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    return cm


def save_confusion_matrix_csv(cm, class_names, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        header = ["true\\pred"] + [str(name) for name in class_names]
        f.write(",".join(header) + "\n")
        for i, class_name in enumerate(class_names):
            row = [str(class_name)] + [str(int(v)) for v in cm[i]]
            f.write(",".join(row) + "\n")

    print(f"Saved confusion matrix to: {output_path}")


def save_checkpoint(path, model, train_mean, train_std, class_names, args, best_val_acc):
    max_per_class_value = -1 if args.max_per_class is None else int(args.max_per_class)
    np.savez(
        path,
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2,
        W3=model.W3,
        b3=model.b3,
        train_mean=train_mean,
        train_std=train_std,
        class_names=np.array(class_names),
        input_dim=np.int64(model.W1.shape[0]),
        hidden_dim=np.int64(model.W1.shape[1]),
        output_dim=np.int64(model.W3.shape[1]),
        activation=np.array(model.activation_name),
        image_size=np.int64(args.image_size),
        loss=np.array(args.loss),
        val_ratio=np.float32(args.val_ratio),
        test_ratio=np.float32(args.test_ratio),
        seed=np.int64(args.seed),
        max_per_class=np.int64(max_per_class_value),
        weight_decay=np.float32(args.weight_decay),
        lr_decay=np.float32(args.lr_decay),
        best_val_acc=np.float32(best_val_acc),
    )


def load_checkpoint(path):
    data = np.load(path, allow_pickle=False)
    return {
        "W1": data["W1"],
        "b1": data["b1"],
        "W2": data["W2"],
        "b2": data["b2"],
        "W3": data["W3"],
        "b3": data["b3"],
        "train_mean": data["train_mean"],
        "train_std": data["train_std"],
        "class_names": data["class_names"],
        "input_dim": int(data["input_dim"]),
        "hidden_dim": int(data["hidden_dim"]),
        "output_dim": int(data["output_dim"]),
        "activation": str(data["activation"]),
        "image_size": int(data["image_size"]),
        "loss": str(data["loss"]),
        "val_ratio": float(data["val_ratio"]),
        "test_ratio": float(data["test_ratio"]),
        "seed": int(data["seed"]),
        "max_per_class": None if int(data["max_per_class"]) < 0 else int(data["max_per_class"]),
        "weight_decay": float(data["weight_decay"]),
        "lr_decay": float(data["lr_decay"]),
        "best_val_acc": float(data["best_val_acc"]),
    }


def load_weights_into_model(model, checkpoint):
    model.W1 = checkpoint["W1"].astype(np.float32)
    model.b1 = checkpoint["b1"].astype(np.float32)
    model.W2 = checkpoint["W2"].astype(np.float32)
    model.b2 = checkpoint["b2"].astype(np.float32)
    model.W3 = checkpoint["W3"].astype(np.float32)
    model.b3 = checkpoint["b3"].astype(np.float32)


def snapshot_model_weights(model):
    return {
        "W1": model.W1.copy(),
        "b1": model.b1.copy(),
        "W2": model.W2.copy(),
        "b2": model.b2.copy(),
        "W3": model.W3.copy(),
        "b3": model.b3.copy(),
    }


def evaluate_split(model, x_data, y_data, loss_name, weight_decay):
    probs, _ = model.forward(x_data)
    base_loss, _ = compute_loss_and_dlogits(probs, y_data, loss_name)
    loss = base_loss + weight_decay * l2_penalty(model)
    acc = accuracy(probs, y_data)
    preds = np.argmax(probs, axis=1)
    return loss, acc, preds
