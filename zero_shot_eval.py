#!/usr/bin/env python3
"""
Zero-shot evaluation for ACT models on CALVIN environment D.

This script bypasses lerobot-eval and creates the CALVIN env directly via
hydra, loads a pretrained ACT policy, and runs online rollouts to compute
success rates on a configurable set of tasks.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import sys

import numpy as np
import torch
from tqdm import tqdm
import hydra
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION

try:
    import wandb
except ImportError:
    wandb = None


# CALVIN env lives under datasets/calvin-archived/calvin_env/calvin_env
_CALVIN_ROOT = Path("/remote-home/wukehao/datasets/calvin-archived/calvin_env")

# 把 CALVIN env 加到 Python 路径，确保 hydra instantiate 能找到 calvin_env 包
calvin_env_path = str(_CALVIN_ROOT)
if calvin_env_path not in sys.path:
    sys.path.insert(0, calvin_env_path)


def load_config(config_path: str = "configs/zero_shot_config.json") -> Dict:
    """加载 JSON 配置文件"""
    # 如果传的是相对路径，则相对于脚本所在目录解析
    if not Path(config_path).is_absolute():
        config_path = str(Path(__file__).parent / config_path)
    with open(config_path) as f:
        return json.load(f)


def make_calvin_env(
    scene: str = "calvin_scene_D_eval",
    cameras: str = "static_and_gripper",
    robot: str = "panda",
    use_egl: bool = True,
    headless: bool = True,
):
    """用 hydra 创建 CALVIN PlayTableSimEnv"""
    conf_dir = str(_CALVIN_ROOT / "conf")

    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(
            config_name="config_data_collection",
            overrides=[
                f"cameras={cameras}",
                f"scene={scene}",
                f"robot={robot}",
                "env=play_table_env",
                "tasks=new_playtable_tasks",
            ],
        )

    # 覆盖掉数据收集相关配置，改为评估模式
    cfg.use_vr = False
    cfg.data_path = str(_CALVIN_ROOT / "data")
    cfg.env.show_gui = not headless
    cfg.env.use_egl = use_egl
    cfg.env.use_vr = False
    cfg.env.use_scene_info = True  # 需要 scene_info 才能做 task 成功判断

    env = instantiate(cfg.env)
    tasks_checker = instantiate(cfg.tasks)
    return env, tasks_checker


def calvin_obs_to_policy_obs(obs: Dict) -> Dict[str, torch.Tensor]:
    """把 CALVIN env 的 obs 转成 LeRobot preprocessor 认识的 torch tensor 格式

    CALVIN obs 结构:
        obs['rgb_obs']['rgb_static']:  (H, W, 3) uint8
        obs['rgb_obs']['rgb_gripper']: (H, W, 3) uint8
        obs['robot_obs']:              (15,)  [tcp_pos(3), tcp_orn_euler(3), gripper(1), joints(7)]

    返回:
        observation.image:       (1, 3, H, W) float32 in [0, 1]
        observation.wrist_image: (1, 3, H, W) float32 in [0, 1]
        observation.state:       (1, 15) float32
    """
    rgb_static = obs["rgb_obs"]["rgb_static"]
    rgb_gripper = obs["rgb_obs"]["rgb_gripper"]
    robot_state = obs["robot_obs"]

    # 保证是 numpy
    if torch.is_tensor(rgb_static):
        rgb_static = rgb_static.cpu().numpy()
    if torch.is_tensor(rgb_gripper):
        rgb_gripper = rgb_gripper.cpu().numpy()
    if torch.is_tensor(robot_state):
        robot_state = robot_state.cpu().numpy()

    # preprocessor 的 device_processor 只处理 torch.Tensor，因此这里先转成 tensor
    # 图像 channel-first、/255 到 [0,1]，并保留 batch 维度
    return {
        "observation.image": torch.from_numpy(
            np.asarray(rgb_static, dtype=np.float32).transpose(2, 0, 1) / 255.0
        ).unsqueeze(0),
        "observation.wrist_image": torch.from_numpy(
            np.asarray(rgb_gripper, dtype=np.float32).transpose(2, 0, 1) / 255.0
        ).unsqueeze(0),
        "observation.state": torch.from_numpy(
            np.asarray(robot_state, dtype=np.float32)
        ).unsqueeze(0),
    }


def evaluate_policy(
    policy: ACTPolicy,
    env,
    tasks_checker,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    num_episodes: int = 100,
    max_steps: int = 360,
    task_names: Optional[List[str]] = None,
    seed: int = 42,
    wandb_run=None,
) -> Dict:
    """在 CALVIN env 上跑 N 个 episode，统计成功率等指标"""
    rng = np.random.default_rng(seed)
    successes = []
    rewards = []
    steps_list = []
    per_task_success = {name: [] for name in (task_names or [])}

    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        if hasattr(env, 'seed'):
            env.seed(seed + episode)
        obs = env.reset()
        policy.reset()  # 清空 ACT 内部的 action queue
        start_info = env.get_info() if hasattr(env, "get_info") else None

        episode_reward = 0.0
        episode_steps = 0

        for step in range(max_steps):
            obs_np = calvin_obs_to_policy_obs(obs)
            obs_tensor = preprocessor(obs_np)

            with torch.inference_mode():
                action = policy.select_action(obs_tensor)

            # 反归一化 action 并移到 CPU
            action = postprocessor({ACTION: action})[ACTION]
            action_np = action.cpu().numpy()

            # action 形状 (1, 7) 或 (7,)
            if action_np.ndim == 2:
                action_np = action_np.squeeze(0)

            # CALVIN action 空间要求 [-1, 1]，对模型输出做裁剪
            action_np = np.clip(action_np, -1.0, 1.0)
            # CALVIN gripper 必须是 -1 或 1
            action_np[-1] = 1.0 if action_np[-1] > 0 else -1.0

            obs, reward, done, info = env.step(action_np)
            episode_reward += reward
            episode_steps += 1

            if done:
                break

        end_info = env.get_info() if hasattr(env, "get_info") else None

        # 检查任务是否完成
        if tasks_checker is not None and start_info is not None and end_info is not None:
            achieved = tasks_checker.get_task_info_for_set(start_info, end_info, set(task_names or []))
            episode_success = len(achieved) > 0
            for name in task_names or []:
                per_task_success[name].append(1.0 if name in achieved else 0.0)
        else:
            episode_success = False

        successes.append(float(episode_success))
        rewards.append(episode_reward)
        steps_list.append(episode_steps)

        # 每 episode 记录到 wandb，画出曲线
        if wandb_run:
            wandb_run.log({
                "episode": episode,
                "episode_success": float(episode_success),
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                "running_success_rate": float(np.mean(successes)),
            })

    success_rate = float(np.mean(successes))
    avg_reward = float(np.mean(rewards))
    avg_steps = float(np.mean(steps_list))

    result = {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "num_episodes": num_episodes,
        "successes": successes,
        "rewards": rewards,
        "steps": steps_list,
        "per_task_success_rate": {name: float(np.mean(vals)) for name, vals in per_task_success.items()},
    }
    return result


def save_results(result: Dict, model_name: str, output_dir: str = "/remote-home/wukehao/nndl-pj3/cross-env/results/zero_shot"):
    """保存聚合结果（不包含大列表）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = output_dir / f"{model_name}_{timestamp}.json"

    save_data = {
        "model_name": model_name,
        "success_rate": result["success_rate"],
        "avg_reward": result["avg_reward"],
        "avg_steps": result["avg_steps"],
        "num_episodes": result["num_episodes"],
        "per_task_success_rate": result["per_task_success_rate"],
        "timestamp": timestamp,
    }

    with open(json_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"💾 Results saved to: {json_file}")
    return json_file


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on CALVIN env D")
    parser.add_argument("--model_path", required=True, help="预训练模型目录")
    parser.add_argument("--model_name", default=None, help="模型名称")
    parser.add_argument("--config", default="configs/zero_shot_config.json", help="配置文件")
    parser.add_argument("--episodes", type=int, default=None, help="覆盖 episode 数")
    parser.add_argument("--no_wandb", action="store_true", help="禁用 wandb")
    parser.add_argument("--no_egl", action="store_true", help="禁用 EGL（如果 headless 渲染有问题）")
    parser.add_argument("--gui", action="store_true", help="显示 GUI（调试用）")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    wandb_cfg = config.get("wandb", {})

    model_name = args.model_name or Path(args.model_path).parent.name
    num_episodes = args.episodes or eval_cfg.get("episodes", 100)
    max_steps = eval_cfg.get("max_steps", 360)
    task_names = eval_cfg.get("tasks", None)

    print("\n" + "=" * 60)
    print("🚀 ZERO-SHOT EVALUATION ON CALVIN ENV D")
    print("=" * 60)
    print(f"📁 Model: {model_name}")
    print(f"📂 Path: {args.model_path}")
    print(f"🎯 Episodes: {num_episodes}")
    print(f"📊 Max steps/episode: {max_steps}")
    print("=" * 60)

    # 创建 CALVIN env
    print("\n🏗️  Creating CALVIN environment...")
    env, tasks_checker = make_calvin_env(
        scene=eval_cfg.get("scene", "calvin_scene_D_eval"),
        cameras=eval_cfg.get("cameras", "static_and_gripper"),
        robot=eval_cfg.get("robot", "panda"),
        use_egl=not args.no_egl,
        headless=not args.gui,
    )
    print("✅ Environment ready")

    # 加载策略
    print("\n🧠 Loading ACT policy...")
    policy = ACTPolicy.from_pretrained(args.model_path)
    policy.eval()
    if torch.cuda.is_available():
        policy = policy.to("cuda")
    print(f"✅ Policy loaded on {next(policy.parameters()).device}")

    # 加载 LeRobot 的 preprocessor / postprocessor
    print("\n⚙️  Loading preprocessors...")
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        args.model_path, config_filename="policy_preprocessor.json"
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        args.model_path, config_filename="policy_postprocessor.json"
    )
    print("✅ Preprocessors loaded")

    # 初始化 wandb
    wandb_run = None
    if not args.no_wandb and wandb_cfg.get("enable", False):
        if wandb:
            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "nndl-pj3"),
                entity=wandb_cfg.get("entity"),
                group=wandb_cfg.get("group"),
                name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=wandb_cfg.get("tags", []),
                notes=wandb_cfg.get("notes", ""),
                config={
                    "model_name": model_name,
                    "model_path": args.model_path,
                    "scene": eval_cfg.get("scene"),
                    "episodes": num_episodes,
                    "max_steps": max_steps,
                },
            )
            print(f"WandB: {wandb_cfg['project']}/{wandb_run.name}")

    # 评估
    result = evaluate_policy(
        policy=policy,
        env=env,
        tasks_checker=tasks_checker,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        num_episodes=num_episodes,
        max_steps=max_steps,
        task_names=task_names,
        wandb_run=wandb_run,
    )

    print(f"\n📊 Results for {model_name}:")
    print(f"   Success Rate: {result['success_rate']:.2%}")
    print(f"   Avg Reward: {result['avg_reward']:.2f}")
    print(f"   Avg Steps: {result['avg_steps']:.1f}")
    if result["per_task_success_rate"]:
        print("   Per-task success rates:")
        for name, rate in result["per_task_success_rate"].items():
            print(f"      {name}: {rate:.2%}")

    # 记录到 wandb
    if wandb_run:
        # 最终聚合指标写入 summary（也再 log 一次方便查看）
        final_log = {
            f"{model_name}/success_rate": result["success_rate"],
            f"{model_name}/avg_reward": result["avg_reward"],
            f"{model_name}/avg_steps": result["avg_steps"],
        }
        for task_name, rate in result["per_task_success_rate"].items():
            final_log[f"{model_name}/{task_name}"] = rate
        wandb_run.log(final_log)

        wandb_run.summary["success_rate"] = result["success_rate"]
        wandb_run.summary["avg_reward"] = result["avg_reward"]
        wandb_run.summary["avg_steps"] = result["avg_steps"]
        for task_name, rate in result["per_task_success_rate"].items():
            wandb_run.summary[task_name] = rate
        wandb_run.finish()
        print("WandB finished")

    # 保存结果
    save_results(result, model_name, eval_cfg.get("output_dir", "results/zero_shot"))

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
