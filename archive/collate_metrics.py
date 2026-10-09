"""
Build a single results table (test-set metrics) from a `runs/` folder.

Expected structure:

    runs/
      dlinear__multivariate__w12_h6/
        metrics.json
      timesfm_ft__multivariate__w12_h12/
        metrics.json
      ...

Each metrics.json looks like:

    {
      "model": "timesfm_ft",
      "task": "multivariate",
      "window": "w12_h12",
      "pipeline": "...",
      "val": {...},
      "test": {
        "log_rmse": ...,
        "rmse": ...,
        "mape": ...,
        "mae": ...,
        "log_mae": ...,
        "n_points": ...
      }
    }

Output: a DataFrame with
  - rows: model
  - columns: MultiIndex (window, metric)  e.g. (w12_h6, rmse), (w12_h6, log_rmse), ...
"""

import json
from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG — edit these
# ----------------------------------------------------------------------

RUNS_DIR = "runs"          # path to the runs folder
METRICS = ["rmse", "mae", "log_rmse", "mape"]   # which metrics to show, and in what order

# Order of rows (models). Leave as None to sort alphabetically / by discovery.
MODEL_ORDER = ["xgb", "dlinear", "patchtst", "itransformer", "timemixer",
               "chronos2_ft", "timesfm_ft",
               "timellm", "gpt4ts",
               "gcn_tcn", "stgcn", "graph_wavenet", "stsgcn", "stllm_plus"]
# e.g. MODEL_ORDER = ["dlinear", "timesfm", "timesfm_ft"]

# Order of top-level columns (window settings). Leave as None to sort naturally.
WINDOW_ORDER = ["w12_h12", "w12_h6"]
# e.g. WINDOW_ORDER = ["w12_h6", "w12_h12", "w24_h12"]

# ----------------------------------------------------------------------


def load_results(runs_dir: str) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(Path(runs_dir).rglob("metrics.json")):
        with open(metrics_path) as f:
            data = json.load(f)

        model = data.get("model")
        window = data.get("window")
        test = data.get("test", {})

        row = {"model": model, "window": window}
        for m in METRICS:
            row[m] = test.get(m)
        rows.append(row)

    if not rows:
        raise ValueError(f"No metrics.json files found under {runs_dir}")

    df = pd.DataFrame(rows)
    return df
 
 
def pivot_table(df: pd.DataFrame,
                 model_order=None,
                 window_order=None,
                 metrics=METRICS) -> pd.DataFrame:
    # Pivot: rows = model, columns = (window, metric)
    pivoted = df.pivot(index="model", columns="window", values=metrics)
 
    # pivoted columns are currently (metric, window) -> reorder to (window, metric)
    pivoted = pivoted.swaplevel(0, 1, axis=1)
 
    # Determine window / metric orders
    windows_present = list(pivoted.columns.get_level_values(0).unique())
    if window_order is None:
        window_order = sorted(windows_present)
    else:
        # keep only windows that were explicitly requested (and that exist)
        window_order = [w for w in window_order if w in windows_present]
 
    metric_order = [m for m in metrics if m in pivoted.columns.get_level_values(1).unique()]
 
    # Reindex columns in desired order
    pivoted = pivoted.reindex(
        columns=pd.MultiIndex.from_product([window_order, metric_order]),
    )
 
    # Reindex rows (models)
    models_present = list(pivoted.index)
    if model_order is None:
        model_order = sorted(models_present)
    else:
        # keep only models that were explicitly requested (and that exist)
        model_order = [m for m in model_order if m in models_present]
 
    pivoted = pivoted.reindex(index=model_order)
 
    return pivoted
 
 
if __name__ == "__main__":
    df = load_results(RUNS_DIR)
    table = pivot_table(df, model_order=MODEL_ORDER, window_order=WINDOW_ORDER, metrics=METRICS)
 
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(table)
 
    # Save to CSV / markdown for convenience
    table.to_csv("results_table.csv")
    with open("results_table.md", "w") as f:
        f.write(table.to_markdown())
 
    print("\nSaved to results_table.csv and results_table.md")