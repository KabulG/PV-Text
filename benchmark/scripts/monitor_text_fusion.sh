#!/usr/bin/env bash
set -euo pipefail
echo "Processes:"
ps -ef | grep run_text_fusion_benchmark | grep -v grep || true
echo
echo "GPU:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo
echo "Summary:"
tail -20 /root/KDD/experiments_text_fusion_models/stage12_and_len96_stdout.log || true
