# Midterm - YOLOv8 Finetune + Tracking

## Python environment

### Recommended
- Python 3.8+ (CUDA optional)

### Dependencies
```bash
pip install ultralytics opencv-python
```

or
```bash
pip install ultralytics opencv-python-headless
```
if there is a no-GUI environment, like running on a server without display.


Optional (for W&B logging in finetune.py):
```bash
pip install wandb
```


## Model weights

- Finetuned weights (default used by scripts):
	- ./finetune-v8l/weights/best.pt
    - ./finetune-v8m/weights/best.pt

## How to run

### 1) Finetune
```bash
python finetune.py \
	--data trafic_data/data_1.yaml \
	--model yolov8m.pt \
	--epochs 50 \
	--imgsz 768 \
	--batch 16 \
	--device 0 \
	--name finetune-v8m
```

If you want to use W&B logging:
```bash
python finetune.py --logger wandb --wandb-project yolov8l-finetune
```

Outputs will be under:
- ./runs/detect/<name>/weights/best.pt

### 2) Tracking + detection on video
```bash
python stream_detection.py \
	--source /root/midterm/IMG_7382.MOV \
	--weights ./finetune-v8l/weights/best.pt \
	--tracker botsort.yaml \
	--imgsz 768 \
	--conf 0.33 \
	--iou 0.7 \
	--device 0
```

Results are saved under:
- ./runs/track/exp/

### 3) Line-crossing counting
```bash
python count.py \
	--source /root/midterm/IMG_7382.MOV \
	--weights ./finetune-v8l/weights/best.pt \
	--tracker botsort.yaml \
	--line 250,620,1780,820 \
	--imgsz 768 \
	--conf 0.33 \
	--iou 0.7 \
	--device 0
```

This script uses default settings of the tracker (BoT-SORT) and the line coordinates (250,620,1780,820) for counting. Parameters can be adjusted as needed.
Outputs will be under:
- ./runs/count/exp/

## Notes
- If you do not have a GPU, set `--device cpu`.
- Change `--line` to match your video resolution. Format: x1,y1,x2,y2.
