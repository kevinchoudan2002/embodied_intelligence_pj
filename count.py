import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Line crossing counting with YOLOv8 tracking."
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
		help="Tracker config (botsort.yaml or bytetrack.yaml).",
	)
	parser.add_argument("--imgsz", type=int, default=768)
	parser.add_argument("--conf", type=float, default=0.33)
	parser.add_argument("--iou", type=float, default=0.7)
	parser.add_argument("--device", default="0")
	parser.add_argument("--project", default="runs/count")
	parser.add_argument("--name", default="exp")
	parser.add_argument(
		"--line",
		default="250,620,1780,820",
		help="Counting line as x1,y1,x2,y2 in pixels.",
	)
	return parser.parse_args()


def parse_line(line_str: str) -> tuple[tuple[int, int], tuple[int, int]]:
	parts = [int(p) for p in line_str.split(",")]
	if len(parts) != 4:
		raise ValueError("--line must be in format x1,y1,x2,y2")
	return (parts[0], parts[1]), (parts[2], parts[3])


def point_side(a: tuple[int, int], b: tuple[int, int], p: tuple[int, int]) -> int:
	# Cross-product sign indicates which side of the line the point lies on.
	cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
	if cross > 0:
		return 1
	if cross < 0:
		return -1
	return 0


def main() -> None:
	args = parse_args()
	source_path = Path(args.source)
	if not source_path.exists():
		raise FileNotFoundError(f"Video not found: {source_path}")

	line_a, line_b = parse_line(args.line)
	output_dir = Path(args.project) / args.name
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{source_path.stem}_count.mp4"

	cap = cv2.VideoCapture(str(source_path))
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video: {source_path}")
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	cap.release()

	writer = cv2.VideoWriter(
		str(output_path),
		cv2.VideoWriter_fourcc(*"mp4v"),
		fps,
		(width, height),
	)

	model = YOLO(args.weights)
	results = model.track(
		source=str(source_path),
		imgsz=args.imgsz,
		conf=args.conf,
		iou=args.iou,
		device=args.device,
		tracker=args.tracker,
		persist=True,
		stream=True,
		verbose=False,
	)

	last_side: dict[int, int] = {}
	counted_ids: set[int] = set()
	total_count = 0

	for result in results:
		frame = result.orig_img
		boxes = result.boxes
		ids = boxes.id
		if ids is not None:
			for xyxy, track_id in zip(boxes.xyxy, ids):
				x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
				cx = int((x1 + x2) / 2)
				cy = int((y1 + y2) / 2)
				tid = int(track_id)
				side = point_side(line_a, line_b, (cx, cy))

				if tid in last_side and side != 0:
					prev = last_side[tid]
					if prev != 0 and side != prev and tid not in counted_ids:
						counted_ids.add(tid)
						total_count += 1

				last_side[tid] = side

				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
				cv2.putText(
					frame,
					f"ID {tid}",
					(x1, max(y1 - 5, 0)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					(0, 255, 0),
					1,
					cv2.LINE_AA,
				)

		cv2.line(frame, line_a, line_b, (0, 0, 255), 2)
		cv2.putText(
			frame,
			f"Count: {total_count}",
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			1.0,
			(0, 0, 255),
			2,
			cv2.LINE_AA,
		)
		writer.write(frame)

	writer.release()
	print(f"Saved: {output_path}")


if __name__ == "__main__":
	main()
