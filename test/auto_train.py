"""
Autonomous Training Agent
-------------------------
This script automates the loop of:
1. Training (Optimization)
2. Verification (Backtest/Validation)
3. Reporting
4. Decision Making (Keep/Discard/Retrain)

It parses the output of the training process to make decisions.
"""

import subprocess
import sys
import re
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("AutoAgent")

PYTHON_EXE = sys.executable
# Adjusted for test/auto_train.py
ROOT_DIR = Path(__file__).resolve().parents[1]

def run_command(cmd: list[str]) -> tuple[int, str]:
    """Run a command and return exit code and stdout."""
    cmd_str = ' '.join(cmd)
    LOGGER.info(f"Running: {cmd_str}")
    print(f"\n[DEBUG] Command to reproduce:\n{cmd_str}\n")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)
    if result.returncode != 0:
        LOGGER.error(f"Command failed: {result.stderr}")
    return result.returncode, result.stdout

def parse_best_value(stdout: str) -> float:
    """Extract Best Value from training output."""
    match = re.search(r"Best Value: ([\d\.-]+)", stdout)
    if match:
        return float(match.group(1))
    return -999.0

def auto_train_loop(
    strategy: str,
    symbol: str,
    n_trials: int = 10,
    target_score: float = 0.05, # e.g. 5% return
    max_attempts: int = 3
):
    LOGGER.info(f"Starting Auto-Train Loop for {strategy} on {symbol}")
    
    attempt = 0
    best_run_score = -999.0
    
    while attempt < max_attempts:
        attempt += 1
        LOGGER.info(f"--- Attempt {attempt}/{max_attempts} ---")
        
        # 1. Train
        cmd = [
            PYTHON_EXE, "scripts/train.py",
            "--strategy", strategy,
            "--symbol", symbol,
            "--n-trials", str(n_trials),
            "--n-seeds", "3", # Use stability
            "--store-path", "storage/auto_train.db"
        ]
        
        code, stdout = run_command(cmd)
        if code != 0:
            LOGGER.error("Training failed. Retrying...")
            continue
            
        score = parse_best_value(stdout)
        LOGGER.info(f"Training finished. Score: {score:.4f}")
        
        if score > best_run_score:
            best_run_score = score
            
        # 2. Decision
        if score >= target_score:
            LOGGER.info(f"Target score {target_score} met! Generating report...")
            
            # 3. Report
            report_cmd = [
                PYTHON_EXE, "scripts/report.py",
                "--strategy", strategy,
                "--symbol", symbol,
                "--store-path", "storage/auto_train.db",
                "--output", f"reports/auto_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "--rerun"
            ]
            run_command(report_cmd)
            LOGGER.info("Auto-training successful.")
            return
        else:
            LOGGER.warning(f"Score {score:.4f} below target {target_score}.")
            
    LOGGER.info(f"Max attempts reached. Best score: {best_run_score:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="star_xgb")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    
    auto_train_loop(args.strategy, args.symbol, n_trials=args.trials)
