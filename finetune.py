import argparse
import os
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Finetune a YOLOv8 model on a custom dataset."
	)
	parser.add_argument("--data", default="trafic_data/data_1.yaml")
	parser.add_argument("--model", default="yolov8m.pt")
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--imgsz", type=int, default=768)
	parser.add_argument("--batch", type=int, default=16)
	parser.add_argument("--device", default="0")
	parser.add_argument("--name", default="finetune-v8m")
	parser.add_argument("--workers", type=int, default=8)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--cache", action="store_true")
	parser.add_argument("--resume", action="store_true")
	parser.add_argument(
		"--logger",
		choices=["none", "wandb"],
		default="none",
		help="Enable external logging. W&B requires `pip install wandb`.",
	)
	parser.add_argument("--wandb-project", default="yolov8l-finetune")
	parser.add_argument("--wandb-name", default="")
	return parser.parse_args()


def setup_wandb(args: argparse.Namespace) -> None:
	os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
	if args.wandb_name:
		os.environ.setdefault("WANDB_NAME", args.wandb_name)


def main() -> None:
	args = parse_args()
	if args.logger == "wandb":
		setup_wandb(args)

	model = YOLO(args.model)
	model.train(
		data=args.data,
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		name=args.name,
		workers=args.workers,
		seed=args.seed,
		cache=args.cache,
		resume=args.resume,
		plots=True,
	)


if __name__ == "__main__":
	main()

