# PV Text-Fusion Forecasting Experiment Code

This package contains the runner currently used on the KDD server for the photovoltaic time-series + text forecasting experiments.

## Scope

The code is designed for the authorized Hebei subset only:

- `hebei_station00` to `hebei_station09`
- Do not use non-Hebei stations unless data authorization changes.

The benchmark uses three text settings:

- `text_stage1`: fuse only stage-1 generated text
- `text_stage2`: fuse only stage-2 generated text
- `text_stage12`: fuse stage-1 and stage-2 generated text together

The aligned main prediction lengths are:

- `16, 32, 48, 96`

The older `pred_len=64` text-fusion results may still exist in the result directory, but should not be used in the aligned main comparison table.

## Server Paths

Expected dataset path:

```bash
/root/KDD/hebei_daytime_0800_1900
```

Expected station files:

```text
/root/KDD/hebei_daytime_0800_1900/hebei_stationXX/solar.csv
/root/KDD/hebei_daytime_0800_1900/hebei_stationXX/rt_text1.jsonl
/root/KDD/hebei_daytime_0800_1900/hebei_stationXX/rt_text2.jsonl
```

Output path:

```bash
/root/KDD/experiments_text_fusion_models
```

Main summary:

```bash
/root/KDD/experiments_text_fusion_models/summary_text_fusion_models.csv
```

## Models

The runner includes four text-fusion forecasting models:

- `TimeCMA`: adapted from the TimeCMA dual/cross-modal architecture
- `CALFAdapter`: cross-attention/gated fusion adapter inspired by CALF-style text-time fusion
- `TimeLLMAdapter`: patch-token + text cross-attention adapter inspired by Time-LLM-style reprogramming
- `TextFiLM`: text-conditioned FiLM baseline

Text is encoded using deterministic hashing embeddings to avoid external model download instability on the server. For each sample, only the last observed input hour's text is used, which avoids using future text from the prediction window.

## Environment

The experiments were launched with:

```bash
/anaconda3/envs/itransformer/bin/python3.8
```

Required Python packages include:

- torch
- numpy
- pandas
- scikit-learn

For `TimeCMA`, the official repository must be available at:

```bash
/root/TimeCMA
```

The runner imports:

```python
from models.TimeCMA import Dual as TimeCMADual
```

## Run Commands

### Full aligned text-fusion benchmark

This is the current corrected experiment design:

```bash
cd /root/KDD
nohup /anaconda3/envs/itransformer/bin/python3.8 /root/KDD/run_text_fusion_benchmark.py \
  --epochs 8 \
  --patience 3 \
  --batch_size 32 \
  --gpus 0,1,2,3 \
  --models TimeCMA,CALFAdapter,TimeLLMAdapter,TextFiLM \
  --phases text_stage1,text_stage2,text_stage12 \
  --stations hebei_station00,hebei_station01,hebei_station02,hebei_station03,hebei_station04,hebei_station05,hebei_station06,hebei_station07,hebei_station08,hebei_station09 \
  --pred_lens 16,32,48,96 \
  > /root/KDD/experiments_text_fusion_models/stage12_and_len96_stdout.log 2>&1 &
```

The runner skips existing successful result JSON files unless `--force` is specified.

### Re-run everything from scratch

Use this only when you intentionally want to overwrite previous result JSON files:

```bash
cd /root/KDD
/anaconda3/envs/itransformer/bin/python3.8 /root/KDD/run_text_fusion_benchmark.py \
  --force \
  --epochs 8 \
  --patience 3 \
  --batch_size 32 \
  --gpus 0,1,2,3 \
  --models TimeCMA,CALFAdapter,TimeLLMAdapter,TextFiLM \
  --phases text_stage1,text_stage2,text_stage12 \
  --pred_lens 16,32,48,96
```

### Smoke test

```bash
cd /root/KDD
/anaconda3/envs/itransformer/bin/python3.8 /root/KDD/run_text_fusion_benchmark.py --smoke --epochs 1 --gpus 0
```

## Monitoring

Check processes:

```bash
ps -ef | grep run_text_fusion_benchmark | grep -v grep
```

Check GPU usage:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

Check progress:

```bash
tail -50 /root/KDD/experiments_text_fusion_models/stage12_and_len96_stdout.log
```

Count result files:

```bash
find /root/KDD/experiments_text_fusion_models/results -name '*.json' | wc -l
```

## Result Format

Each task writes one JSON file under:

```bash
/root/KDD/experiments_text_fusion_models/results
```

The summary CSV contains:

```text
model, phase, station, pred_len, status, mae, mse, rmse, val_loss, test_loss, elapsed_sec, gpu, error
```

For the corrected main text-fusion benchmark, filter rows by:

```text
phase in {text_stage1, text_stage2, text_stage12}
pred_len in {16, 32, 48, 96}
status == success
```

Ignore older `pred_len=64` rows when preparing the aligned benchmark table.
