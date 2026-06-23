import json
import subprocess
import time
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


def _resolve_unique_output_dir(output_dir: Path, job_name: str, resume: bool) -> Path:
    if resume or not output_dir.exists():
        return output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir.parent / f"{output_dir.name}_{job_name}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = output_dir.parent / f"{output_dir.name}_{job_name}_{timestamp}_{suffix:02d}"
        suffix += 1
    return candidate


def run_training(config: Dict[str, Any], output_dir: Path, job_name: str) -> Dict[str, Any]:
    """
    single-time training
    
    Args:
        config: training configuration dictionary
        output_dir: output directory
        job_name: job name
    
    Returns:
        dictionary containing training results
    """
    output_dir = Path(output_dir)
    resume = bool(config.get("resume", False))
    output_dir = _resolve_unique_output_dir(output_dir, job_name, resume)
    config["output_dir"] = str(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    config_fd, config_path_str = tempfile.mkstemp(prefix=f"{output_dir.name}_", suffix="_config.json", dir=output_dir.parent)
    config_path = Path(config_path_str)
    with open(config_fd, 'w') as f:
        json.dump(config, f, indent=2)
    
    cmd = ["lerobot-train"]
    for key, value in config.items():
        if isinstance(value, bool):
            value = str(value).lower()
        cmd.append(f"--{key}={value}")

    print(f"\n{'='*60}")
    print(f"Job: {job_name}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=36000)
        elapsed = time.time() - start_time
        
        success = result.returncode == 0

        if output_dir.exists():
            try:
                final_config_path = output_dir / "config.json"
                with open(final_config_path, "w") as f:
                    json.dump(config, f, indent=2)
            except Exception:
                pass

        eval_score = extract_eval_score(output_dir)
        
        return {
            "success": success,
            "eval_score": eval_score,
            "elapsed_minutes": elapsed / 60,
            "output_dir": str(output_dir),
            "config": config,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Timeout",
            "output_dir": str(output_dir),
            "config": config
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "output_dir": str(output_dir),
            "config": config
        }
    finally:
        if config_path.exists():
            try:
                config_path.unlink()
            except Exception:
                pass


def extract_eval_score(output_dir: Path) -> Optional[float]:
    eval_file = output_dir / "eval" / "eval_results.json"
    if eval_file.exists():
        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
                return data.get("mean_score") or data.get("success_rate")
        except:
            pass
    
    log_file = output_dir / "logs" / "training.log"
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if "eval/mean_score" in line:
                        import re
                        match = re.search(r'"eval/mean_score":\s*([0-9.]+)', line)
                        if match:
                            return float(match.group(1))
        except:
            pass
    
    return None


def save_results(results: List[Dict], save_path: Path):
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(results_path: Path) -> List[Dict]:
    with open(results_path, 'r') as f:
        return json.load(f)