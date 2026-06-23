import itertools
import subprocess
import json
from pathlib import Path
from datetime import datetime

# ========== 配置 ==========
DATASET_PATH = "/remote-home/wukehao/datasets/calvin_task_ABC_D/splitA"
OUTPUT_BASE = Path("configs/grid_search_results")

# Center point based on best random search results(018)
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

GRID_SEARCH_RANGES = {
    "policy.kl_weight": [-5, 0, 5],     
    "batch_size": [-8, 0, 8],           
    "optimizer.lr": [-2e-5, 0, 2e-5],
}

FIXED_PARAMS = {
    "policy.type": "act",
    "policy.device": "cuda",
    "policy.repo_id": "local/grid_search",
    "policy.push_to_hub": False,
    "policy.n_encoder_layers": 4,
    "policy.n_decoder_layers": 6,
    "policy.n_heads": 8,
    "policy.dim_feedforward": 2048,
    "policy.dropout": 0.1,
    "policy.use_vae": True,
}

TRAIN_CONFIG = {
    "dataset.repo_id": DATASET_PATH,
    "policy.type": "act",
    "policy.device": "cuda",
    "policy.repo_id": "local/grid_search",
    "policy.push_to_hub": False,
    "policy.dropout": 0.1,
    "policy.use_vae": True,
    "policy.use_amp": True,
    "steps": 3000,
    "eval_freq": 1000,
    "eval.n_episodes": 5,
    "eval.batch_size": 5,
    "log_freq": 200,
    "save_freq": 1000,
    "num_workers": 16,
    "wandb.enable": True,
    "wandb.project": "calvin_act_grid_search",
}


def generate_grid_params():
    param_names = []
    param_values = []
    
    for param_name, ranges in GRID_SEARCH_RANGES.items():
        if param_name not in CENTER_PARAMS:
            continue
        
        base_value = CENTER_PARAMS[param_name]
        values = []
        
        for delta in ranges:
            if param_name == "optimizer.lr":
                new_value = base_value + delta
                if new_value > 0:
                    values.append(new_value)
            else:
                new_value = base_value + delta
                
                # make sure values are within reasonable bounds
                if param_name == "policy.chunk_size" and 50 <= new_value <= 500:
                    values.append(int(new_value))
                elif param_name == "policy.dim_model" and 256 <= new_value <= 1024:
                    values.append(int(new_value))
                elif param_name == "policy.n_encoder_layers" and 2 <= new_value <= 12:
                    values.append(int(new_value))
                elif param_name == "policy.n_decoder_layers" and 2 <= new_value <= 12:
                    values.append(int(new_value))
                elif param_name == "policy.n_heads" and 2 <= new_value <= 16:
                    values.append(int(new_value))
                elif param_name == "policy.dim_feedforward" and 512 <= new_value <= 4096:
                    values.append(int(new_value))
                elif param_name == "policy.kl_weight" and 1 <= new_value <= 100:
                    values.append(int(new_value))
                elif param_name == "batch_size" and 4 <= new_value <= 64:
                    values.append(int(new_value))

        values = list(dict.fromkeys(values))
        
        if values:
            param_names.append(param_name)
            param_values.append(values)

    combinations = list(itertools.product(*param_values))

    params_list = []
    for combo in combinations:
        params = {}
        for i, param_name in enumerate(param_names):
            params[param_name] = combo[i]
        params_list.append(params)
    
    return params_list


def build_command(params):
    cmd = ["lerobot-train"]

    for k, v in FIXED_PARAMS.items():
        if isinstance(v, bool):
            v = str(v).lower()
        cmd.append(f"--{k}={v}")

    for k, v in TRAIN_CONFIG.items():
        if isinstance(v, bool):
            v = str(v).lower()
        cmd.append(f"--{k}={v}")

    for k, v in params.items():
        cmd.append(f"--{k}={v}")
    
    # ensure n_action_steps = chunk_size
    if "policy.chunk_size" in params:
        cmd.append(f"--policy.n_action_steps={params['policy.chunk_size']}")
    
    return cmd



def save_results(results, save_path):
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


def grid_search():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    grid_params_list = generate_grid_params()
    print(f"Starting Grid Search with {len(grid_params_list)} experiments")
    print(f"Searching: {list(GRID_SEARCH_RANGES.keys())}")
    
    results = []
    
    for idx, params in enumerate(grid_params_list):
        output_dir = OUTPUT_BASE / f"grid_{timestamp}_{idx:03d}"
        job_name = f"grid_{timestamp}_{idx:03d}"

        cmd = build_command(params)
        cmd.append(f"--output_dir={output_dir}")
        cmd.append(f"--job_name={job_name}")
        
        print(f"\n{'='*60}")
        print(f"Experiment {idx+1}/{len(grid_params_list)}")
        print(f"Parameters: {json.dumps(params, indent=2)}")
        print(f"Output: {output_dir}")
        print('='*60)
        
        # Execute training
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        results.append({
            "trial_id": idx,
            "params": params,
            "output_dir": str(output_dir),
            "success": result.returncode == 0
        })
        
        save_results(results, OUTPUT_BASE / f"{timestamp}_grid_search_results.json")
    
    print(f"Grid Search completed! Results saved to: {OUTPUT_BASE}")
    return results


if __name__ == "__main__":
    grid_search()