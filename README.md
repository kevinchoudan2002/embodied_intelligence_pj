## 任务一
任务一主要运用了3DGS的train.py和threestudio的launch.py，我在命令行进行训练，没有额外的信息可以提供。

## 任务二

基于 LeRobot 框架，在 CALVIN 数据集上对 ACT（Action Chunking Transformer）模型进行两阶段超参数调优（Random Search → Grid Search），并在未见过的环境 D 上进行 Zero-Shot 泛化评估。

## 项目结构

```
.
├── random_search.py              # 阶段一：随机搜索（粗筛）
├── grid_search.py                # 阶段二：网格搜索（细调）
├── train.py                      # 单次训练（加载配置）
├── zero-shot-eval.py             # Zero-Shot 评估（环境 D）
├── check-data.py                 # 数据集检查工具
├── merge-calvin-dataset.py       # 合并 splitA/B/C 数据集
├── configs/
│   ├── act_base_config.json      # ACT 基础配置
│   ├── env_best_config.json      # 最优配置
│   └── zero_shot_config.json     # Zero-Shot 评估配置
├── utils/
│   ├── config_utils.py           # 配置加载/展平/合并/保存
│   ├── train_utils.py            # 训练执行/结果提取/保存
│   ├── fix-episode-stats.py      # 修复 episodes_stats.jsonl 格式
│   ├── fix-parquet.py            # 重命名 parquet 列名（适配 LeRobot 格式）
│   └── generate-stats-json.py    # 从 parquet 生成 stats.json
└── results/
    └── zero_shot/                # Zero-Shot 评估结果
```

## 环境配置

```bash
# 创建并激活 conda 环境
conda create -n lerobot_env python=3.10
conda activate lerobot_env

# 安装 PyTorch（CUDA 11.8）
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 安装 LeRobot 及依赖
pip install lerobot
pip install wandb tensorboard opencv-python-headless pyarrow pandas pillow

# 安装 CALVIN 环境依赖（如需运行评估）
cd /remote-home/wukehao/datasets/calvin-archived/calvin_env
pip install -e .
```

具体环境依赖请见报告的supplementary material。

## 数据准备

### 数据集目录结构

```
datasets/
├── calvin_task_ABC_D/            # 原始数据（四个 split）
│   ├── splitA/
│   │   ├── data/
│   │   │   └── chunk-*/episode_*.parquet
│   │   └── meta/
│   │       ├── episodes_stats.jsonl
│   │       └── stats.json
│   ├── splitB/
│   ├── splitC/
│   └── splitD/
├── calvin_merge_ABC/             # 合并后的数据集（A+B+C）
│   ├── data/
│   └── meta/
└── calvin_env/                  # 环境代码（用于评估）
```

### 数据预处理流程

#### 1. 合并数据集（可选）

如需在更大的数据集上训练（A+B+C 共 3 个 split），运行合并脚本：

```bash
python merge-calvin-dataset.py
```

该脚本使用 `lerobot.datasets.aggregate.aggregate_datasets` 将 splitA、splitB、splitC 合并为 `calvin_merge_ABC`。

#### 2. 修复 Parquet 列名

LeRobot 要求特定列名格式，运行以下脚本将 `image` → `observation.image`，`state` → `observation.state`，`actions` → `action`：

```bash
# 修改脚本中的 dataset_path 为目标数据集路径
python utils/fix-parquet.py
```

#### 3. 生成 stats.json

从 parquet 文件中计算 `observation.state` 和 `action` 的统计量（min/max/mean/std/count）：

```bash
# 修改脚本中的 dataset_path 为目标数据集路径
python utils/generate-stats-json.py
```

#### 4. 修复 episodes_stats.jsonl（如有格式问题）

```bash
# 修改脚本中的 dataset_path 为目标数据集路径
python utils/fix-episode-stats.py
```

该脚本会：
- 将标量统计值转为列表格式
- 补全缺失的 `min`/`max`/`mean`/`std`/`count` 字段
- 对齐各列表长度

### 数据集检查

查看任意 episode 的数据结构（图像、状态、动作）：

```bash
# 修改 ROOT 和 episode_index
python check-data.py
```

输出示例：
```
Index(['image', 'wrist_image', 'state', 'actions'], dtype='object')
```

## 使用指南

### 阶段一：随机搜索（粗筛）

在较大的搜索空间内随机采样 20 组超参数，快速定位最优区域。

```bash
python random_search.py
```

**搜索空间：**

| 参数 | 候选值 |
|---|---|
| `policy.chunk_size` | 100, 150, 200 |
| `policy.dim_model` | 384, 512, 768 |
| `policy.n_encoder_layers` | 4, 5, 6 |
| `policy.n_decoder_layers` | 5, 6, 7 |
| `policy.n_heads` | 4, 8, 12 |
| `policy.dim_feedforward` | 1024, 2048, 3072 |
| `policy.kl_weight` | 10, 50, 100 |
| `batch_size` | 8, 16, 32 |
| `optimizer.lr` | 1e-5, 3e-5, 5e-5, 1e-4 |

**固定配置：**
- 训练步数: 5000
- 评估频率: 每 1000 步
- 评估回合数: 10
- 使用 AMP（混合精度训练）
- WandB 项目: `calvin_act_random_search_v2`

**输出：**
- 每个 trial 保存在 `configs/random_search_results/{timestamp}_trial_{xxx}/`
- 汇总: `configs/random_search_results/random_search_results.json`
- 最优参数: `configs/random_search_results/best_params_random.json`

### 阶段二：网格搜索（细调）

以随机搜索得到的最优参数为中心，在其邻域内进行精细化网格搜索。

**先修改 `grid_search.py` 中的 `CENTER_PARAMS`：**

```python
# 替换为 random_search 输出的 best_params_random.json 中的值
CENTER_PARAMS = {
    "policy.chunk_size": 200,
    "policy.dim_model": 512,
    "policy.n_encoder_layers": 6,
    "policy.n_decoder_layers": 7,
    "policy.n_heads": 8,
    "policy.dim_feedforward": 1024,
    "policy.kl_weight": 10,
    "batch_size": 32,
    "optimizer.lr": 5e-5,
}
```

然后运行：

```bash
python grid_search.py
```

**搜索范围（基于中心参数偏移）：**

| 参数 | 中心值 | 搜索值（中心 + 偏移） |
|---|---|---|
| `policy.kl_weight` | 10 | 5, 10, 15 |
| `batch_size` | 32 | 24, 32, 40 |
| `optimizer.lr` | 5e-5 | 3e-5, 5e-5, 7e-5 |

**固定配置：**
- `policy.n_encoder_layers`: 4
- `policy.n_decoder_layers`: 6
- `policy.dim_feedforward`: 2048
- `policy.n_heads`: 8
- `policy.dropout`: 0.1
- 训练步数: 3000
- 评估频率: 每 1000 步
- 评估回合数: 5
- WandB 项目: `calvin_act_grid_search`

**输出：**
- 每个 trial 保存在 `configs/grid_search_results/grid_{timestamp}_{xxx}/`
- 汇总: `configs/grid_search_results/{timestamp}_grid_search_results.json`

### 单次训练

使用 `train.py` 加载 `configs/env_best_config.json` 进行训练：

```bash
python train.py
```

等价命令行：

```bash
lerobot-train \
    --dataset.repo_id=/remote-home/wukehao/datasets/calvin_task_ABC_D/splitA \
    --policy.type=act \
    --policy.device=cuda \
    --policy.chunk_size=200 \
    --policy.n_action_steps=200 \
    --policy.dim_model=512 \
    --policy.n_encoder_layers=4 \
    --policy.n_decoder_layers=6 \
    --policy.n_heads=8 \
    --policy.dim_feedforward=2048 \
    --policy.kl_weight=10 \
    --policy.dropout=0.1 \
    --policy.use_vae=True \
    --policy.use_amp=True \
    --batch_size=32 \
    --optimizer.lr=5e-5 \
    --steps=3000 \
    --eval_freq=1000 \
    --eval.n_episodes=5 \
    --output_dir=./outputs/act_best \
    --job_name=act_best \
    --wandb.enable=True \
    --wandb.project=calvin_act_final
```

### Zero-Shot 跨环境评估

在**未见过的环境 D** 上评估训练好的模型，测试泛化能力。

```bash
# 基本用法
python zero-shot-eval.py --model_path /path/to/pretrained/model

# 完整示例
python zero-shot-eval.py \
    --model_path configs/grid_search_results/grid_20260624_091530_000/ \
    --model_name act_grid_best \
    --config configs/zero_shot_config.json \
    --episodes 100 \
    --no_wandb
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model_path` | 预训练模型目录（必须包含 config.json 和 checkpoints） | 必填 |
| `--model_name` | 模型名称（用于保存结果） | 目录名 |
| `--config` | Zero-Shot 配置文件 | configs/zero_shot_config.json |
| `--episodes` | 评估 episode 数 | config 中的值 |
| `--no_wandb` | 禁用 WandB | False |
| `--no_egl` | 禁用 EGL（headless 渲染出问题时用） | False |
| `--gui` | 显示 GUI（调试用） | False |

**环境配置（在 `zero_shot_config.json` 中设置）：**

```json
{
    "evaluation": {
        "scene": "calvin_scene_D_eval",
        "cameras": "static_and_gripper",
        "robot": "panda",
        "episodes": 100,
        "max_steps": 360,
        "tasks": ["move_slider_left", "turn_on_led"],
        "output_dir": "results/zero_shot"
    },
    "wandb": {
        "enable": true,
        "project": "nndl-pj3",
        "entity": null,
        "group": "zero_shot",
        "tags": ["act", "calvin_d"]
    }
}
```

**输出：**
- 结果 JSON: `results/zero_shot/{model_name}_{timestamp}.json`
- WandB 日志（如果启用）
- 每 episode 成功率曲线（WandB）

## 参数说明

| 参数 | 说明 | 典型范围 |
|---|---|---|
| `policy.chunk_size` | 动作块大小，每次预测的动作序列长度 | 100-200 |
| `policy.n_action_steps` | 同 `chunk_size`，保持同步 | 同左 |
| `policy.dim_model` | Transformer 模型维度 | 384-768 |
| `policy.n_encoder_layers` | Encoder 层数 | 4-6 |
| `policy.n_decoder_layers` | Decoder 层数 | 5-7 |
| `policy.n_heads` | 多头注意力头数 | 4-12 |
| `policy.dim_feedforward` | FFN 维度 | 1024-3072 |
| `policy.kl_weight` | VAE KL 散度权重 | 10-100 |
| `policy.dropout` | Dropout 率 | 0.1 |
| `policy.use_vae` | 是否使用 VAE | True |
| `policy.use_amp` | 混合精度训练 | True |
| `batch_size` | 批次大小 | 8-32 |
| `optimizer.lr` | 学习率 | 1e-5 ~ 1e-4 |

## 输出目录结构

```
configs/
├── random_search_results/
│   ├── 20260623_143022_trial_000/
│   │   ├── checkpoints/
│   │   │   └── checkpoint_last.pth
│   │   └── config.json
│   ├── ...
│   ├── random_search_results.json
│   └── best_params_random.json
├── grid_search_results/
│   ├── grid_20260624_091530_000/
│   │   ├── checkpoints/
│   │   └── config.json
│   ├── ...
│   └── 20260624_091530_grid_search_results.json
├── env_best_config.json
└── zero_shot_config.json

results/
└── zero_shot/
    ├── act_grid_best_20260625_153022.json
    └── ...
```

## 典型工作流

```bash
# 1. 数据准备（如果数据集格式有问题）
python utils/fix-parquet.py
python utils/generate-stats-json.py
python utils/fix-episode-stats.py

# 2. 运行随机搜索（粗筛）
python random_search.py

# 3. 查看 best_params_random.json，更新 grid_search.py 中的 CENTER_PARAMS

# 4. 运行网格搜索（细调）
python grid_search.py

# 5. 用最优配置进行单次训练（可选）
python train.py

# 6. 在环境 D 上 Zero-Shot 评估
python zero-shot-eval.py \
    --model_path configs/grid_search_results/grid_20260624_091530_000/ \
    --episodes 100
```

## 注意事项

1. **显存要求**：16GB显存可以运行，但比较勉强
2. **数据路径**：修改脚本中的 `DATASET_PATH`、`_CALVIN_ROOT` 等路径为你的实际路径
3. **环境 D 评估**：需要提前下载完整的 `calvin_env` 仓库并安装相应的环境依赖
4. **Hydra 兼容性**：Zero-Shot 评估脚本使用 Hydra 创建环境，确保 `calvin_env/conf` 目录存在
5. **调试模式**：评估时加 `--gui --episodes 1` 可查看可视化效果
6. **数据集合并后**：必须运行 `fix-parquet.py` 和 `generate-stats-json.py` 修复格式
7. **并行执行**：如需并行运行多个 trial，建议改用 slurm 或 `--multi-run`
