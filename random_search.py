#!/usr/bin/env python3
"""
Random Search for Hyperparameter Tuning, only on dataset A.
"""

import random
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from utils.train_utils import run_training, save_results
from utils.config_utils import load_base_config, merge_configs

BASE_CONFIG_PATH = Path("configs/act_base_config.json") 
OUTPUT_BASE = Path("configs/random_search_results")
NUM_TRIALS = 20 

SEARCH_SPACE = {
    "policy.chunk_size": [100, 150, 200],
    "policy.dim_model": [384, 512, 768],           
    "policy.n_encoder_layers": [4, 5, 6], 
    "policy.n_decoder_layers": [5, 6, 7],
    "policy.n_heads": [4, 8, 12],
    "policy.dim_feedforward": [1024, 2048, 3072],
    "policy.kl_weight": [10, 50, 100],
    "batch_size": [8, 16, 32],
    "optimizer.lr": [1e-5, 3e-5, 5e-5, 1e-4],
}

RANDOM_OVERRIDES = {
    "steps": 5000,                 
    "eval_freq": 1000,              
    "eval.n_episodes": 10,         
    "eval.batch_size": 10,
    "policy.device": "cuda",
    "policy.push_to_hub": False,
    "policy.use_amp": True,          
    "num_workers": 16,               
    "log_freq": 200,
    "save_freq": 1000,
    "wandb.enable": True,
    "wandb.project": "calvin_act_random_search_v2",
}


def generate_random_params() -> Dict[str, Any]:
    chunk_size = random.choice(SEARCH_SPACE["policy.chunk_size"])
    
    params = {
        "policy.chunk_size": chunk_size,
        "policy.n_action_steps": chunk_size, 
        "policy.dim_model": random.choice(SEARCH_SPACE["policy.dim_model"]),
        "policy.n_encoder_layers": random.choice(SEARCH_SPACE["policy.n_encoder_layers"]),
        "policy.n_decoder_layers": random.choice(SEARCH_SPACE["policy.n_decoder_layers"]),
        "policy.n_heads": random.choice(SEARCH_SPACE["policy.n_heads"]),
        "policy.dim_feedforward": random.choice(SEARCH_SPACE["policy.dim_feedforward"]),
        "policy.kl_weight": random.choice(SEARCH_SPACE["policy.kl_weight"]),
        "batch_size": random.choice(SEARCH_SPACE["batch_size"]),
        "optimizer.lr": random.choice(SEARCH_SPACE["optimizer.lr"]),
    }
    return params


def random_search():
    print(f"Loading base config: {BASE_CONFIG_PATH}")
    base_config = load_base_config(BASE_CONFIG_PATH)

    base_config = merge_configs(base_config, RANDOM_OVERRIDES)
    
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = []
    
    print(f"Starting random search, total {NUM_TRIALS} trials")
    print(f"Base config:")
    for key, value in base_config.items():
        if key not in SEARCH_SPACE:
            print(f"   {key}: {value}")
    print(f"\nSearch space: {len(SEARCH_SPACE)} parameters")
    print(f"Results save directory: {OUTPUT_BASE}")
    
    for trial_id in range(NUM_TRIALS):
        random_params = generate_random_params()
        config = merge_configs(base_config, random_params)

        output_dir = OUTPUT_BASE / f"{timestamp}_trial_{trial_id:03d}"
        job_name = f"random_{timestamp}_{trial_id:03d}"
        config["output_dir"] = str(output_dir)
        config["job_name"] = job_name
        
        result = run_training(config, output_dir, job_name)
        result["trial_id"] = trial_id
        result["params"] = random_params
        
        results.append(result)

        save_results(results, OUTPUT_BASE / "random_search_results.json")

        print(f"Progress: {trial_id + 1}/{NUM_TRIALS}")
        if result.get("eval_score"):
            print(f"   Score: {result['eval_score']:.4f}")
        else:
            print("   No evaluation score")
        if result.get("success"):
            print(f"   Time: {result.get('elapsed_minutes', 0):.1f} min")

    print_random_results_summary(results)
    
    return results


def print_random_results_summary(results):
    successful = [r for r in results if r.get("success") and r.get("eval_score")]
    
    if not successful:
        print("Unsuccessful experiments")
        return
    sorted_results = sorted(successful, key=lambda x: x["eval_score"], reverse=True)
    best = sorted_results[0]
    
    print(f"\n{'='*60}")
    print("DONE - Random Search Summary ")
    print('='*60)
    print(f"Total experiments: {len(results)}")
    print(f"Successful experiments: {len(successful)}")
    print(f"\nBest result:")
    print(f"   Score: {best['eval_score']:.4f}")
    print(f"   Parameters: {json.dumps(best['params'], indent=2)}")
    print(f"   Output: {best['output_dir']}")

    best_params_path = OUTPUT_BASE / "best_params_random.json"
    with open(best_params_path, 'w') as f:
        json.dump(best['params'], f, indent=2)
    print(f"Best parameters saved to: {best_params_path}")


if __name__ == "__main__":
    random_search()