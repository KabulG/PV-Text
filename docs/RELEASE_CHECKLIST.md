# Release Checklist

Before publishing this repository on GitHub:

- Confirm the license for the incremental text dataset and code.
- Confirm the citation text for the original public Hebei photovoltaic time-series dataset.
- Run `benchmark/scripts/check_authorized_scope.py`.
- Verify that only `hebei_station00` to `hebei_station09` appear under `data/hebei_daytime_0800_1900`.
- Verify that no training output directories are included.
- Verify that no `pred.npy`, `true.npy`, checkpoint, or large log file is included.
- Re-run leakage and text-numeric consistency audits after any text template change.
- Do not publish non-Hebei stations without separate authorization.
