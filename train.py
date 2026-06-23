import json
from pathlib import Path
from lerobot.scripts.lerobot_train import train
from lerobot.configs.train import TrainPipelineConfig

# 直接从配置文件加载
cfg = TrainPipelineConfig.from_pretrained("configs/env_best_config.json")

# 开始训练
train(cfg)