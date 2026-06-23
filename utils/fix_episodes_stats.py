#!/usr/bin/env python3
"""
fix episodes_stats.jsonl: 1. fill missing part 2. change non-list count/min/max/mean/std to list
"""

import json
import numpy as np
from pathlib import Path

def fix_episodes_stats_complete(stats_path):
    backup_path = stats_path.with_suffix('.jsonl.bak2')
    if not backup_path.exists():
        stats_path.rename(backup_path)
        print(f"backup file created: {backup_path}")
    else:
        print(f"⚠️ backup file already exists, will overwrite: {backup_path}")
        stats_path.rename(backup_path)
    
    fixed_lines = []
    with open(backup_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            entry = json.loads(line)
            
            if 'stats' not in entry:
                print(f"Line {line_num}: missing stats")
                continue

            for feat_key, feat_stats in entry['stats'].items():
                if 'count' not in feat_stats:
                    print(f"  Line {line_num}, {feat_key}: missing count field, adding default value [0]")
                    feat_stats['count'] = [0]
                elif not isinstance(feat_stats['count'], list):
                    print(f"  Line {line_num}, {feat_key}.count: {feat_stats['count']} -> [{feat_stats['count']}]")
                    feat_stats['count'] = [feat_stats['count']]

                for stat_name in ['min', 'max', 'mean', 'std']:
                    if stat_name not in feat_stats:
                        print(f"  Line {line_num}, {feat_key}.{stat_name}: missing, adding default value [0.0]")
                        feat_stats[stat_name] = [0.0]
                    elif not isinstance(feat_stats[stat_name], list):
                        print(f"  Line {line_num}, {feat_key}.{stat_name}: {feat_stats[stat_name]} -> [{feat_stats[stat_name]}]")
                        feat_stats[stat_name] = [feat_stats[stat_name]]
                
                expected_len = len(feat_stats['count'])
                for stat_name in ['min', 'max', 'mean', 'std']:
                    if len(feat_stats[stat_name]) != expected_len:
                        print(f"  Line {line_num}, {feat_key}.{stat_name}:  ({len(feat_stats[stat_name])} vs {expected_len}), fixing...")
                        if len(feat_stats[stat_name]) < expected_len:
                            # fill missing values with last known value
                            last_val = feat_stats[stat_name][-1] if feat_stats[stat_name] else 0.0
                            feat_stats[stat_name].extend([last_val] * (expected_len - len(feat_stats[stat_name])))
                        else:
                            feat_stats[stat_name] = feat_stats[stat_name][:expected_len]
            
            fixed_lines.append(json.dumps(entry))
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
        if fixed_lines:
            f.write('\n')
    
    print(f"fixed {len(fixed_lines)} lines")
    print(f"   fixed file: {stats_path}")

def main():
    dataset_path = Path("/remote-home/wukehao/datasets/calvin_env_ABC")
    stats_file = dataset_path / "meta" / "episodes_stats.jsonl"
    
    if not stats_file.exists():
        print(f"❌ File does not exist: {stats_file}")
        # Try to create an empty file
        print(f"   Creating empty file: {stats_file}")
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        stats_file.touch()
        print(f"Please rerun the LeRobot conversion script, it will automatically generate the correct statistics")
        return
    
    print(f"Starting repair: {stats_file}")
    fix_episodes_stats_complete(stats_file)
    
    print("Verifying repair results...")
    with open(stats_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            for feat_key, feat_stats in entry.get('stats', {}).items():
                if 'count' not in feat_stats or not isinstance(feat_stats['count'], list):
                    print(f"  ❌ Line {line_num}, {feat_key}: still problematic count = {feat_stats.get('count')}")
                else:
                    print(f"  ✅ Line {line_num}, {feat_key}: count = {feat_stats['count'][:3]}...")

if __name__ == "__main__":
    main()