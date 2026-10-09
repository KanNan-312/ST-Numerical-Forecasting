# Archive

Stale pre-existing files, retired during a repo cleanup pass, kept here
(not deleted) for reference. None of these are imported by the active
codebase.

- `collate_metrics.py` — an older, hardcoded-config duplicate of
  `scripts/make_report.py` (which uses `src/st_numeric_baselines/metrics/reporting.py`
  and takes `--runs`/`--out` CLI args instead of editing constants in the
  file). Superseded.
- `results_table.csv`, `results_table.md` — a stale generated report
  (model list didn't match any run under `runs/` at the time of the
  cleanup) that had been committed instead of regenerated on demand.
- `graph_structure.png` — a one-off plot with no accompanying script or
  caption, unreferenced anywhere in the active codebase or docs.
- `interpretability.ipynb` — an untitled scratch notebook (ad hoc pandas
  exploration against `library/seattle_house/*.csv`), no narrative.
