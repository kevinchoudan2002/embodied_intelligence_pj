"""
3-layer MLP entry script.

Core logic is split into reusable modules:
- data_pipeline.py
- mlp_model.py
- training.py
- hp_search.py
- checkpoint_eval.py
"""

import argparse

import matplotlib.pyplot as plt

from checkpoint_eval import (
	build_confusion_matrix,
	evaluate_split,
	load_checkpoint,
	load_weights_into_model,
	save_checkpoint,
	save_confusion_matrix_csv,
)
from data_pipeline import prepare_data
from hp_search import run_grid_search, run_random_search
from misclassified_examples import save_misclassified_examples
from mlp_model import MLP3Layer
from training import run_training


def plot_training_curves(history, loss_plot_path, acc_plot_path):
	epochs = [item["epoch"] for item in history]
	train_loss = [item["train_loss"] for item in history]
	val_loss = [item["val_loss"] for item in history]
	val_acc = [item["val_acc"] for item in history]

	plt.figure(figsize=(7, 4))
	plt.plot(epochs, train_loss, marker="o", label="Train Loss")
	plt.plot(epochs, val_loss, marker="o", label="Val Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Training and Validation Loss")
	plt.grid(alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(loss_plot_path, dpi=180)
	plt.close()

	plt.figure(figsize=(7, 4))
	plt.plot(epochs, val_acc, marker="o", color="tab:green", label="Val Accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.title("Validation Accuracy")
	plt.grid(alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(acc_plot_path, dpi=180)
	plt.close()

	print(f"Saved loss plot to: {loss_plot_path}")
	print(f"Saved val accuracy plot to: {acc_plot_path}")


def create_arg_parser():
	parser = argparse.ArgumentParser(description="NumPy-only 3-layer MLP for EuroSAT_RGB")
	parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB", help="Path to EuroSAT_RGB root folder")
	parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size for both hidden layers")
	parser.add_argument(
		"--activation",
		type=str,
		default="relu",
		choices=["relu", "sigmoid", "tanh"],
		help="Hidden activation function",
	)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--lr", type=float, default=0.01)
	parser.add_argument(
		"--lr-decay",
		type=float,
		default=0.01,
		help="Inverse-time LR decay factor: lr_t = lr / (1 + lr_decay * (epoch-1))",
	)
	parser.add_argument("--weight-decay", type=float, default=0.01, help="L2 weight decay coefficient")
	parser.add_argument(
		"--loss",
		type=str,
		default="cross_entropy",
		choices=["cross_entropy", "mse"],
		help="Loss function used for training and backprop",
	)
	parser.add_argument("--val-ratio", type=float, default=0.2)
	parser.add_argument("--test-ratio", type=float, default=0.1)
	parser.add_argument("--image-size", type=int, default=32)
	parser.add_argument(
		"--max-per-class",
		type=int,
		default=None,
		help="Optional cap of samples per class for faster experiments",
	)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--search-mode",
		type=str,
		default="none",
		choices=["none", "grid", "random"],
		help="Hyperparameter search mode",
	)
	parser.add_argument("--search-epochs", type=int, default=5, help="Epochs per trial in grid/random search")
	parser.add_argument("--grid-lrs", type=str, default="0.01,0.001", help="Comma-separated lr values for grid search")
	parser.add_argument(
		"--grid-hidden-dims",
		type=str,
		default="128,256",
		help="Comma-separated hidden_dim values for grid search",
	)
	parser.add_argument(
		"--grid-weight-decays",
		type=str,
		default="0.0,0.0001",
		help="Comma-separated weight_decay values for grid search",
	)
	parser.add_argument("--num-trials", type=int, default=10, help="Number of random-search trials")
	parser.add_argument("--random-lr-min", type=float, default=1e-4)
	parser.add_argument("--random-lr-max", type=float, default=1e-1)
	parser.add_argument("--random-hidden-min", type=int, default=64)
	parser.add_argument("--random-hidden-max", type=int, default=512)
	parser.add_argument("--random-weight-decay-min", type=float, default=1e-6)
	parser.add_argument("--random-weight-decay-max", type=float, default=1e-2)
	parser.add_argument(
		"--results-csv",
		type=str,
		default="search_results.csv",
		help="Output CSV file for grid/random search results",
	)
	parser.add_argument("--checkpoint-path", type=str, default="best_model.npz", help="Path to save/load best checkpoint")
	parser.add_argument(
		"--load-best",
		action="store_true",
		help="Load checkpoint from --checkpoint-path and run test evaluation only",
	)
	parser.add_argument(
		"--confusion-matrix-csv",
		type=str,
		default="confusion_matrix.csv",
		help="Output CSV file for confusion matrix",
	)
	parser.add_argument(
		"--plot-curves",
		action="store_true",
		help="Save training/validation loss and validation accuracy curves",
	)
	parser.add_argument(
		"--save-misclassified",
		action="store_true",
		help="Save misclassified test images with true/predicted class labels",
	)
	parser.add_argument(
		"--misclassified-dir",
		type=str,
		default="misclassified_examples",
		help="Output folder for misclassified test images",
	)
	parser.add_argument(
		"--loss-plot",
		type=str,
		default="loss_curve.png",
		help="Output image path for train/val loss curve",
	)
	parser.add_argument(
		"--acc-plot",
		type=str,
		default="val_acc_curve.png",
		help="Output image path for validation accuracy curve",
	)
	return parser


def main(args):
	x_train, y_train, x_val, y_val, x_test, y_test, class_to_idx, train_mean, train_std, _, _, _ = prepare_data(
		args,
		return_paths=True,
	)
	num_classes = len(class_to_idx)
	idx_to_class = {idx: name for name, idx in class_to_idx.items()}
	class_names = [idx_to_class[i] for i in range(num_classes)]

	if args.load_best:
		checkpoint = load_checkpoint(args.checkpoint_path)
		model = MLP3Layer(
			input_dim=checkpoint["input_dim"],
			hidden_dim=checkpoint["hidden_dim"],
			output_dim=checkpoint["output_dim"],
			activation=checkpoint["activation"],
			seed=checkpoint["seed"],
		)
		load_weights_into_model(model, checkpoint)

		x_train, y_train, x_val, y_val, x_test, y_test, class_to_idx, _, _, _, _, test_paths = prepare_data(
			args,
			forced_mean=checkpoint["train_mean"],
			forced_std=checkpoint["train_std"],
			override_image_size=checkpoint["image_size"],
			override_val_ratio=checkpoint["val_ratio"],
			override_test_ratio=checkpoint["test_ratio"],
			override_seed=checkpoint["seed"],
			override_max_per_class=checkpoint["max_per_class"],
			return_paths=True,
		)
		num_classes = len(class_to_idx)
		idx_to_class = {idx: name for name, idx in class_to_idx.items()}
		class_names = [idx_to_class[i] for i in range(num_classes)]

		test_loss, test_acc, test_preds = evaluate_split(
			model,
			x_test,
			y_test,
			checkpoint["loss"],
			checkpoint["weight_decay"],
		)
		cm = build_confusion_matrix(y_test, test_preds, num_classes)
		print(f"Loaded checkpoint: {args.checkpoint_path}")
		print(f"Checkpoint best val_acc={checkpoint['best_val_acc']:.4f}")
		print(f"Test loss={test_loss:.4f}, test acc={test_acc:.4f}")
		print("Confusion matrix:")
		print(cm)
		save_confusion_matrix_csv(cm, class_names, args.confusion_matrix_csv)
		if args.save_misclassified:
			save_misclassified_examples(test_paths, y_test, test_preds, class_names, args.misclassified_dir)
		return

	if args.search_mode == "none":
		model, _, best_state, best_val_acc, history = run_training(
			args=args,
			x_train=x_train,
			y_train=y_train,
			x_val=x_val,
			y_val=y_val,
			num_classes=num_classes,
			hidden_dim=args.hidden_dim,
			lr=args.lr,
			weight_decay=args.weight_decay,
			epochs=args.epochs,
			seed=args.seed,
			collect_history=args.plot_curves,
			verbose=True,
		)

		if args.plot_curves:
			plot_training_curves(history, args.loss_plot, args.acc_plot)

		load_weights_into_model(model, best_state)
		save_checkpoint(
			path=args.checkpoint_path,
			model=model,
			train_mean=train_mean,
			train_std=train_std,
			class_names=class_names,
			args=args,
			best_val_acc=best_val_acc,
		)
		print(f"Saved best checkpoint to: {args.checkpoint_path}")

		test_loss, test_acc, test_preds = evaluate_split(model, x_test, y_test, args.loss, args.weight_decay)
		cm = build_confusion_matrix(y_test, test_preds, num_classes)
		print("\nEvaluation on independent test set using best checkpoint:")
		print(f"best_val_acc={best_val_acc:.4f}")
		print(f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
		print("Confusion matrix:")
		print(cm)
		save_confusion_matrix_csv(cm, class_names, args.confusion_matrix_csv)
		if args.save_misclassified:
			save_misclassified_examples(test_paths, y_test, test_preds, class_names, args.misclassified_dir)
	elif args.search_mode == "grid":
		run_grid_search(args, x_train, y_train, x_val, y_val, num_classes)
	elif args.search_mode == "random":
		run_random_search(args, x_train, y_train, x_val, y_val, num_classes)
	else:
		raise ValueError(f"Unsupported search mode: {args.search_mode}")


if __name__ == "__main__":
	parser = create_arg_parser()
	main(parser.parse_args())

