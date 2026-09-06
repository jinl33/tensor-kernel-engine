#!/bin/bash
#SBATCH --job-name=triton_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gpu-type=a100
#SBATCH --time=00:30:00
#SBATCH --output=bench_%j.log

set -euo pipefail

module load python/3.10 cuda/12.1
source ~/venv/bin/activate
pip install triton torch torchvision

python main.py --benchmark --dtype fp16 --output assets/kernel_perf_panel.png
