# Dataset Card: PV-Text Hebei

## Summary

PV-Text Hebei is a text-augmented photovoltaic forecasting dataset. It extends an already public Hebei photovoltaic time-series dataset with two aligned hourly operational text fields generated from auditable rules.

## Data Scope

- Region: authorized Hebei subset only.
- Stations: `hebei_station00` to `hebei_station09`.
- Sampling window: daytime records from 08:00 to 19:00.
- Base numerical file: `solar.csv`.
- Text files: `rt_text1.jsonl` and `rt_text2.jsonl`.

No non-Hebei station data is included in this release package.

## Text Construction

The text construction process uses expert photovoltaic operation knowledge, photovoltaic operation knowledge graph constraints, and LLM-assisted prompt/rule drafting. The LLM is used to draft and refine auditable rules and templates; it is not treated as an uncontrolled direct label generator.

Stage-1 operational text describes:

- operating state
- recent trend
- variability

Stage-2 decomposition text describes:

- low-frequency trend
- high-frequency component

Both text stages are aligned to hourly timestamps.

## Intended Uses

- Pure photovoltaic time-series forecasting.
- Text-fusion photovoltaic forecasting.
- Text-control experiments for temporal text alignment.
- Benchmarking multimodal renewable-energy forecasting models.

## Evaluation Protocol

Prediction horizons:

```text
16, 32, 48, 96
```

Metrics:

```text
MAE, MSE, RMSE
```

Recommended reporting:

- station-level results
- horizon-level averages
- overall average across the 10 authorized Hebei stations

## Known Caveats

- The text is rule-generated from numerical windows and expert-designed templates; it is not real operator log text.
- Template repetition is expected and should be reported.
- Future-looking wording in generated templates should be audited before a final public release.
- Benchmark comparisons should use a single data loader and metric path. Mixing customized model repositories can produce non-comparable results.

## Release Contents

This package contains data, rules, prompt templates, split metadata, benchmark scripts, and audit scripts. It excludes training outputs and model checkpoints.
