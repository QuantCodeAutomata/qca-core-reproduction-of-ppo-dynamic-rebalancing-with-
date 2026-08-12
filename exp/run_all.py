"""Master runner — executes all four experiments in sequence.

Usage (from project root):
    python exp/run_all.py
"""

import subprocess
import sys

experiments = [
    'exp/exp_1_core.py',
    'exp/exp_2_comparison.py',
    'exp/exp_3_ablation.py',
    'exp/exp_4_sensitivity.py',
]

if __name__ == '__main__':
    for script in experiments:
        print(f'Running {script}...')
        result = subprocess.run([sys.executable, script], capture_output=False)
        print(f'Finished {script} with exit code {result.returncode}')
