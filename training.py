from checkpoint_eval import snapshot_model_weights
from data_pipeline import iterate_minibatches
from encode_loss_calc import compute_loss_and_dlogits, l2_penalty
from mlp_model import MLP3Layer, accuracy, get_epoch_lr


def run_training(
    args,
    x_train,
    y_train,
    x_val,
    y_val,
    num_classes,
    hidden_dim,
    lr,
    weight_decay,
    epochs,
    seed,
    collect_history=False,
    verbose=True,
    prefix="",
):
    model = MLP3Layer(
        input_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        activation=args.activation,
        seed=seed,
    )

    last_metrics = None
    best_val_acc = -float("inf")
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        epoch_lr = get_epoch_lr(lr, args.lr_decay, epoch)
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        total_samples = 0

        for x_batch, y_batch in iterate_minibatches(x_train, y_train, args.batch_size, seed=seed + epoch):
            probs, cache = model.forward(x_batch)
            base_loss, dlogits = compute_loss_and_dlogits(probs, y_batch, args.loss)
            loss = base_loss + weight_decay * l2_penalty(model)
            acc = accuracy(probs, y_batch)

            grads = model.backward(cache, dlogits)
            model.step(grads, epoch_lr, weight_decay)

            n = x_batch.shape[0]
            train_loss_sum += loss * n
            train_acc_sum += acc * n
            total_samples += n

        train_loss = train_loss_sum / total_samples
        train_acc = train_acc_sum / total_samples

        val_probs, _ = model.forward(x_val)
        val_base_loss, _ = compute_loss_and_dlogits(val_probs, y_val, args.loss)
        val_loss = val_base_loss + weight_decay * l2_penalty(model)
        val_acc = accuracy(val_probs, y_val)

        last_metrics = {
            "epoch": epoch,
            "lr": epoch_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        if collect_history:
            history.append(last_metrics.copy())

        if verbose:
            print(
                f"{prefix}Epoch {epoch:03d} | "
                f"lr={epoch_lr:.6f} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        if val_acc > best_val_acc or best_state is None:
            best_val_acc = val_acc
            best_state = snapshot_model_weights(model)
            if verbose:
                print(f"Updated in-memory best model (val_acc={best_val_acc:.4f})")

    if best_state is None:
        best_state = snapshot_model_weights(model)
        best_val_acc = -float("inf") if last_metrics is None else last_metrics["val_acc"]

    return model, last_metrics, best_state, best_val_acc, history
