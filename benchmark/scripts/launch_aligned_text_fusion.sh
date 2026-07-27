#!/usr/bin/env bash
set -euo pipefail
cd /root/KDD
/anaconda3/envs/itransformer/bin/python3.8 /root/KDD/run_text_fusion_benchmark.py \
  --epochs 8 \
  --patience 3 \
  --batch_size 32 \
  --gpus 0,1,2,3 \
  --models TimeCMA,CALFAdapter,TimeLLMAdapter,TextFiLM \
  --phases text_stage1,text_stage2,text_stage12 \
  --stations hebei_station00,hebei_station01,hebei_station02,hebei_station03,hebei_station04,hebei_station05,hebei_station06,hebei_station07,hebei_station08,hebei_station09 \
  --pred_lens 16,32,48,96
