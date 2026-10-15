"""
Experiment Tracking & Provenance Logger.
Records git commit hash, config file, random seed, dataset version, hardware,
and metric outcomes for all pipeline and baseline runs.
"""

import os
import sys
import csv
import json
import subprocess
import datetime
from typing import Dict, Any, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_LOG_CSV = os.path.join(ROOT, "results", "experiments_log.csv")
DEFAULT_LOG_JSONL = os.path.join(ROOT, "results", "experiments_log.jsonl")


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_git_branch() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


class ExperimentTracker:
    def __init__(self, experiment_name: str, config_path: str = "config/config.yaml", seed: int = 42):
        self.experiment_name = experiment_name
        self.config_path = config_path
        self.seed = seed
        self.commit_hash = get_git_commit_hash()
        self.branch = get_git_branch()
        self.timestamp = datetime.datetime.now().isoformat()
        self.metadata: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}

    def log_metadata(self, key: str, value: Any):
        self.metadata[key] = value

    def log_metrics(self, metrics: Dict[str, Any]):
        self.metrics.update(metrics)

    def log_metric(self, key: str, value: Any):
        self.metrics[key] = value

    def save(self, csv_path: str = DEFAULT_LOG_CSV, jsonl_path: str = DEFAULT_LOG_JSONL):
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        
        record = {
            "timestamp": self.timestamp,
            "experiment_name": self.experiment_name,
            "commit_hash": self.commit_hash,
            "branch": self.branch,
            "seed": self.seed,
            "config_path": self.config_path,
            "metrics": json.dumps(self.metrics),
            "metadata": json.dumps(self.metadata),
        }
        
        # Write to JSONL
        with open(jsonl_path, "a") as f:
            f.write(json.dumps({
                "timestamp": self.timestamp,
                "experiment_name": self.experiment_name,
                "commit_hash": self.commit_hash,
                "branch": self.branch,
                "seed": self.seed,
                "config_path": self.config_path,
                "metrics": self.metrics,
                "metadata": self.metadata,
            }) + "\n")
            
        # Write to CSV
        file_exists = os.path.exists(csv_path)
        fieldnames = ["timestamp", "experiment_name", "commit_hash", "branch", "seed", "config_path", "metrics", "metadata"]
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
            
        print(f"[ExperimentTracker] Logged run '{self.experiment_name}' ({self.commit_hash[:8]}) to {csv_path}")
