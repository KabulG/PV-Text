# Recovered PV-Text Prompt Templates

This document records the closest recoverable template contract for
`datasets/station00/source_station_stage1_hourly_prompts.jsonl` and
`datasets/station00/source_station_stage2_hourly_prompts.jsonl`.

The original generator source code was not found in the workspace. The
templates below were recovered from:

- `datasets/station00/source_station_stage1_hourly_prompts.jsonl`
- `datasets/station00/source_station_stage2_hourly_prompts.jsonl`
- `datasets/station00/source_station_stage1_description.txt`
- `datasets/station00/source_station_stage2_description.txt`
- intermediate backups: `.bak`, `.shortbak`, `.naturalbak`, `.asciibak`
- `prompt_requirements_for_codex.md`

## Output Files

Stage 1 file:

```json
{
  "timestamp": "YYYY-MM-DD HH:00:00",
  "state_prompt": "...",
  "recent_trend_prompt": "...",
  "statistical_variability_prompt": "..."
}
```

Stage 2 file:

```json
{
  "timestamp": "YYYY-MM-DD HH:00:00",
  "low_frequency_trend_prompt": "...",
  "high_frequency_component_prompt": "..."
}
```

## Stage 1 State Prompt

The final template is a base state sentence plus optional modifiers.

Base state options:

```text
The station is under nighttime or very weak radiation conditions, and photovoltaic output remains near zero.
The station is under weak radiation conditions, and photovoltaic output is still suppressed.
The station is under moderate radiation conditions, and photovoltaic output is in an active but not peak-producing state.
The station is under strong radiation conditions, and photovoltaic output is in an active daytime regime.
```

Optional modifiers observed in the final file:

```text
The thermal background is relatively cool
wind is light
the NWP radiation background is available for guidance
```

Modifier rendering:

```text
{base}
{base} {modifier}.
{base} {modifier1}; {modifier2}.
{base} {modifier1}; {modifier2}; {modifier3}.
```

The final file contains 27 unique `state_prompt` strings, produced by this
base-plus-modifiers pattern.

## Stage 1 Recent Trend Prompt

This field is a fixed concatenation of three sub-prompts:

```text
{output_trend} {observed_radiation_trend} {nwp_shortwave_trend}
```

Output trend options:

```text
Output has been recovering over the recent hour.
Output has softened over the recent hour.
Output has stayed broadly steady over the recent hour.
```

Observed radiation trend options:

```text
Observed radiation has strengthened.
Observed radiation has weakened.
Observed radiation has stayed broadly stable.
```

NWP shortwave trend options:

```text
The NWP shortwave background is also trending upward.
The NWP shortwave background is also trending downward.
The NWP shortwave background shows little net change.
```

The final file contains 27 unique combinations.

## Stage 1 Statistical Variability Prompt

Final options:

```text
Recent variability is low, and the series remains locally smooth.
Recent variability is moderate, with some short-term fluctuation but no severe instability.
Recent variability is high, and the series shows clear short-term disturbance.
```

## Stage 2 Low Frequency Trend Prompt

Final options are one nighttime baseline sentence plus a 3x3 combination of
trend direction and radiation background regime.

Night baseline:

```text
The low-frequency background remains near a nighttime baseline through the coming horizon.
```

Strengthening trend:

```text
The low-frequency background is expected to strengthen gradually, with the radiation regime leaning toward a direct-beam-dominant background.
The low-frequency background is expected to strengthen gradually, with the radiation regime leaning toward a mixed direct and diffuse background.
The low-frequency background is expected to strengthen gradually, with the radiation regime leaning toward a diffuse-cloud-dominant background.
```

Weakening trend:

```text
The low-frequency background is expected to weaken gradually, with the radiation regime characterized by a direct-beam-dominant background.
The low-frequency background is expected to weaken gradually, with the radiation regime characterized by a mixed direct and diffuse background.
The low-frequency background is expected to weaken gradually, with the radiation regime characterized by a diffuse-cloud-dominant background.
```

Stable trend:

```text
The low-frequency background is expected to stay broadly stable, under a direct-beam-dominant background.
The low-frequency background is expected to stay broadly stable, under a mixed direct and diffuse background.
The low-frequency background is expected to stay broadly stable, under a diffuse-cloud-dominant background.
```

The final file contains 10 unique strings.

## Stage 2 High Frequency Component Prompt

Final options:

```text
The high-frequency component is very weak, and obvious mutation-like jumps are unlikely.
The high-frequency component remains limited, with only mild short-lived fluctuation expected.
The high-frequency component may contain intermittent cloud-driven jumps and short-lived reversals.
The high-frequency component is active, and sharp cloud-edge-like jumps or rapid reversals are likely.
```

## Historical Evolution In Backups

The backup files show the prompt design was compressed over several iterations:

- `.bak`: verbose numeric prompts with `scenario_prompt` and raw values.
- `.shortbak`: shorter numeric prompts with explicit slopes, stds, and ramps.
- `.naturalbak`: compact numeric-natural prompts.
- final `.jsonl`: short semantic labels without raw numeric values.

This supports the current interpretation: the final generator was intended to
be rule/label driven, not free-form LLM text.

