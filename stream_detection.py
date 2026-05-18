import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run YOLOv8 detection + multi-object tracking on a video."
	)
	parser.add_argument(
		"--source",
		default="/root/midterm/IMG_7382.MOV",
		help="Path to the input video.",
	)
	parser.add_argument(
		"--weights",
		default="/root/midterm/runs/detect/finetune-v8l/weights/best.pt",
		help="Path to the finetuned model weights.",
	)
	parser.add_argument(
		"--tracker",
		default="botsort.yaml",
		help="Tracker config (sds).",
	)
	parser.add_argument("--imgsz", type=int, default=768)
	parser.add_argument("--conf", type=float, default=0.33)
	parser.add_argument("--iou", type=float, default=0.7)
	parser.add_argument("--device", default="0")
	parser.add_argument("--project", default="track")
	parser.add_argument("--name", default="exp")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	source_path = Path(args.source)
	if not source_path.exists():
		raise FileNotFoundError(f"Video not found: {source_path}")

	model = YOLO(args.weights)
	model.track(
		source=str(source_path),
		imgsz=args.imgsz,
		conf=args.conf,
		iou=args.iou,
		device=args.device,
		tracker=args.tracker,
		persist=True,
		save=True,
		project=args.project,
		name=args.name,
		verbose=True,
	)


if __name__ == "__main__":
	main()
