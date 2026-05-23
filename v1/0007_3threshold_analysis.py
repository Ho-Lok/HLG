#!/usr/bin/env python3
"""
THRESHOLD ANALYSIS PIPELINE (0007_3threshold_analysis.py)

USAGE:
    Run individually (analyze existing forward results):
        python 0007_3threshold_analysis.py
  
    Scan specific file:
        python 0007_3threshold_analysis.py --file 0007_forward_output/final_ensemble_results_live_v1.csv
  
    Scan custom directory:
        python 0007_3threshold_analysis.py --input ./my_results_dir
  
    Specify output date:
        python 0007_3threshold_analysis.py --date 2026-05-12
  
    With options:
        python 0007_3threshold_analysis.py --help
  
    Run as part of daily workflow (via orchestrator):
        python 0007_0run_daily.py

OUTPUTS:
    - threshold_analysis/YYYY-MM-DD/
        - best_params.json (recommended min_pred_return, min_xgb_score)
        - grid_analysis.csv (heatmap of all threshold combinations)
        - band_summary.csv (quantile bands with performance metrics)
        - plots/ (visualization charts if matplotlib available)

PURPOSE:
    - Find optimal prediction return threshold (min_pred_return)
    - Find optimal rank score threshold (min_xgb_score)
    - Analyze which score ranges have best realized outcomes
    - Identify overfitting or regime changes

DEPENDENCIES:
    - final_ensemble_results_live_v1.csv (from 0007_1forward.py)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover
    plt = None
    sns = None


PRED_CANDIDATES = ["Pred_Return_%", "Pred_Return", "Predicted_Alpha_%", "Pred"]
XGB_CANDIDATES = ["XGB_Rank_Score", "XGB_Rank", "Rank_Score", "Rank"]
ACTUAL_CANDIDATES = ["Actual_20D_Return_%", "Actual_15D_Return_%", "Actual_Return_%", "Actual_Alpha_%", "Actual_Return", "Actual"]


@dataclass(frozen=True)
class ThresholdResult:
    pred_min: float
    xgb_min: float
    n: int
    mean_actual: float
    median_actual: float
    win_rate: float
    mean_pred: float
    mean_xgb: float


def find_input_files(root: Path) -> list[Path]:
    patterns = [
        "**/final_ensemble_results*.csv",
        "**/*ensemble_results*.csv",
        "**/*forward*.csv",
    ]
    files: set[Path] = set()
    for pattern in patterns:
        for file_path in root.glob(pattern):
            if file_path.is_file():
                files.add(file_path)
    return sorted(files)


def pick_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        lower_candidate = candidate.lower()
        if lower_candidate in lower_map:
            return lower_map[lower_candidate]
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_forward_data(files: list[Path]) -> pd.DataFrame:
    frames = []
    for file_path in files:
        try:
            frame = pd.read_csv(file_path)
        except Exception:
            continue
        frame["__source_file"] = str(file_path)
        frames.append(frame)

    if not frames:
        raise ValueError("No readable forward-result CSV files were found.")

    return pd.concat(frames, ignore_index=True)


def detect_schema(df: pd.DataFrame) -> tuple[str, str, str]:
    pred_col = pick_column(df.columns, PRED_CANDIDATES)
    xgb_col = pick_column(df.columns, XGB_CANDIDATES)
    actual_col = pick_column(df.columns, ACTUAL_CANDIDATES)

    if not pred_col or not xgb_col or not actual_col:
        raise ValueError(
            "Could not detect required columns. Needed one of: "
            f"pred={PRED_CANDIDATES}, xgb={XGB_CANDIDATES}, actual={ACTUAL_CANDIDATES}. "
            f"Available columns: {list(df.columns)}"
        )

    return pred_col, xgb_col, actual_col


def prepare_data(df: pd.DataFrame, pred_col: str, xgb_col: str, actual_col: str) -> pd.DataFrame:
    data = df.copy()
    data[pred_col] = coerce_numeric(data[pred_col])
    data[xgb_col] = coerce_numeric(data[xgb_col])
    data[actual_col] = coerce_numeric(data[actual_col])
    data = data[[pred_col, xgb_col, actual_col, "__source_file"]].dropna(subset=[pred_col, xgb_col])
    data = data.rename(columns={pred_col: "pred", xgb_col: "xgb", actual_col: "actual"})
    return data


def summarize_threshold_pair(data: pd.DataFrame, pred_min: float, xgb_min: float) -> ThresholdResult | None:
    subset = data[(data["pred"] >= pred_min) & (data["xgb"] >= xgb_min)]
    if subset.empty:
        return None

    actual = subset["actual"]
    return ThresholdResult(
        pred_min=float(pred_min),
        xgb_min=float(xgb_min),
        n=int(len(subset)),
        mean_actual=float(actual.mean()),
        median_actual=float(actual.median()),
        win_rate=float((actual > 0).mean()),
        mean_pred=float(subset["pred"].mean()),
        mean_xgb=float(subset["xgb"].mean()),
    )


def build_threshold_grid(data: pd.DataFrame, grid_size: int, min_samples: int) -> pd.DataFrame:
    pred_thresholds = np.unique(np.quantile(data["pred"], np.linspace(0.0, 0.95, grid_size)))
    xgb_thresholds = np.unique(np.quantile(data["xgb"], np.linspace(0.0, 0.95, grid_size)))

    rows = []
    for pred_min in pred_thresholds:
        for xgb_min in xgb_thresholds:
            result = summarize_threshold_pair(data, pred_min, xgb_min)
            if result is None or result.n < min_samples:
                continue
            rows.append({
                "pred_min": result.pred_min,
                "xgb_min": result.xgb_min,
                "n": result.n,
                "mean_actual": result.mean_actual,
                "median_actual": result.median_actual,
                "win_rate": result.win_rate,
                "mean_pred": result.mean_pred,
                "mean_xgb": result.mean_xgb,
                # Balance effect size and sample size so tiny buckets do not dominate.
                "score": result.mean_actual * np.log1p(result.n),
            })

    return pd.DataFrame(rows)


def add_binned_summary(data: pd.DataFrame, bins: int) -> pd.DataFrame:
    summary = data.copy()
    summary["pred_bin"] = pd.qcut(summary["pred"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    summary["xgb_bin"] = pd.qcut(summary["xgb"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    grouped = (
        summary.groupby(["pred_bin", "xgb_bin"], dropna=True)
        .agg(
            n=("actual", "size"),
            mean_actual=("actual", "mean"),
            median_actual=("actual", "median"),
            win_rate=("actual", lambda s: float((s > 0).mean())),
            mean_pred=("pred", "mean"),
            mean_xgb=("xgb", "mean"),
        )
        .reset_index()
    )
    return grouped


def write_heatmap(grid: pd.DataFrame, outdir: Path) -> Path | None:
    if plt is None or sns is None or grid.empty:
        return None

    pivot = grid.pivot(index="xgb_min", columns="pred_min", values="mean_actual")
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot.sort_index(ascending=True), cmap="RdYlGn", center=0.0)
    plt.title("Mean Actual Forward Return by Threshold Pair")
    plt.xlabel("min_pred_return")
    plt.ylabel("min_xgb_score")
    plt.tight_layout()
    path = outdir / "threshold_heatmap.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def write_band_heatmap(bands: pd.DataFrame, outdir: Path) -> Path | None:
    if plt is None or sns is None or bands.empty:
        return None

    pivot = bands.pivot(index="xgb_bin", columns="pred_bin", values="mean_actual")
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot.sort_index(ascending=True), cmap="RdYlGn", center=0.0)
    plt.title("Mean Actual Forward Return by Score Band")
    plt.xlabel("Pred Return Band")
    plt.ylabel("XGB Score Band")
    plt.tight_layout()
    path = outdir / "score_band_heatmap.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze forward data to find the best min_pred_return and min_xgb_score range.")
    parser.add_argument("--input", default=".", help="Folder to scan for forward-result CSVs")
    parser.add_argument("--outdir", default="0007_forward_output/0007_3", help="Output folder for analysis artifacts")
    parser.add_argument("--grid-size", type=int, default=15, help="Number of candidate threshold values per axis")
    parser.add_argument("--bins", type=int, default=8, help="Number of quantile bands for the band summary")
    parser.add_argument("--min-samples", type=int, default=25, help="Minimum rows required for a threshold pair")
    parser.add_argument("--date", default=None, help="Output date folder override (YYYY-MM-DD)")
    parser.add_argument("--file", action="append", default=None, help="Specific CSV file to analyze. Can be repeated.")
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    base_out = Path(args.outdir).resolve()
    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    outdir = base_out / date_str
    outdir.mkdir(parents=True, exist_ok=True)

    files = [Path(file_path).resolve() for file_path in args.file] if args.file else find_input_files(input_root)
    if not files:
        raise SystemExit(f"No forward-result CSV files found under {input_root}")

    df_raw = load_forward_data(files)
    pred_col, xgb_col, actual_col = detect_schema(df_raw)
    data = prepare_data(df_raw, pred_col, xgb_col, actual_col)

    if data.empty:
        raise SystemExit("No usable pred/xgb rows after numeric coercion and NA filtering.")

    has_actuals = data["actual"].notna().any()
    generated_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not has_actuals:
        note_lines = [
            "# Forward Threshold Analysis",
            "",
            f"Generated At (UTC): {generated_utc}",
            "",
            f"Input files: {len(files)}",
            f"Rows analyzed: {len(data)}",
            f"Detected columns: pred=`{pred_col}`, xgb=`{xgb_col}`, actual=`{actual_col}`",
            "",
            "## Result",
            "No realized actual-return values were available in the input CSV, so threshold optimization was skipped.",
            "The script completed successfully and preserved the forward run artifacts.",
            "",
            "## Files",
        ]
        threshold_csv = outdir / "threshold_grid_summary.csv"
        band_csv = outdir / "score_band_summary.csv"
        pd.DataFrame(columns=["pred_min", "xgb_min", "n", "mean_actual", "median_actual", "win_rate", "mean_pred", "mean_xgb", "score", "generated_utc"]).to_csv(threshold_csv, index=False)
        pd.DataFrame(columns=["pred_bin", "xgb_bin", "n", "mean_actual", "median_actual", "win_rate", "mean_pred", "mean_xgb", "generated_utc"]).to_csv(band_csv, index=False)
        (outdir / "analysis_report.md").write_text("\n".join(note_lines + [f"- Threshold grid: {threshold_csv.name}", f"- Score bands: {band_csv.name}", "- Heatmap: not generated", "- Band heatmap: not generated"]), encoding="utf-8")
        summary = {
            "generated_utc": generated_utc,
            "input_files": [str(path) for path in files],
            "rows_analyzed": int(len(data)),
            "detected_columns": {"pred": pred_col, "xgb": xgb_col, "actual": actual_col},
            "best_threshold_pair": {},
            "status": "no_actual_data",
            "artifacts": {
                "threshold_csv": str(threshold_csv),
                "band_csv": str(band_csv),
                "heatmap": None,
                "band_heatmap": None,
                "report": str(outdir / "analysis_report.md"),
            },
        }
        (outdir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Analysis written to: {outdir}")
        print("No actual-return data found; threshold optimization skipped.")
        return

    threshold_grid = build_threshold_grid(data, grid_size=args.grid_size, min_samples=args.min_samples)
    threshold_grid = threshold_grid.sort_values(["mean_actual", "win_rate", "n", "score"], ascending=[False, False, False, False])
    threshold_grid["generated_utc"] = generated_utc

    band_summary = add_binned_summary(data, bins=args.bins)
    band_summary = band_summary.sort_values(["mean_actual", "win_rate", "n"], ascending=[False, False, False])
    band_summary["generated_utc"] = generated_utc

    threshold_csv = outdir / "threshold_grid_summary.csv"
    band_csv = outdir / "score_band_summary.csv"
    threshold_grid.to_csv(threshold_csv, index=False)
    band_summary.to_csv(band_csv, index=False)

    heatmap_path = write_heatmap(threshold_grid, outdir)
    band_heatmap_path = write_band_heatmap(band_summary, outdir)

    best_row = threshold_grid.iloc[0].to_dict() if not threshold_grid.empty else {}
    top_ranges = threshold_grid.head(20).copy()

    report_lines = [
        f"# Forward Threshold Analysis",
        "",
        f"Generated At (UTC): {generated_utc}",
        "",
        f"Input files: {len(files)}",
        f"Rows analyzed: {len(data)}",
        f"Detected columns: pred=`{pred_col}`, xgb=`{xgb_col}`, actual=`{actual_col}`",
        "",
        "## Best Threshold Pair",
    ]

    if best_row:
        report_lines.extend([
            f"- `min_pred_return`: {best_row['pred_min']:.6g}",
            f"- `min_xgb_score`: {best_row['xgb_min']:.6g}",
            f"- Rows selected: {int(best_row['n'])}",
            f"- Mean actual return: {best_row['mean_actual']:.4f}",
            f"- Median actual return: {best_row['median_actual']:.4f}",
            f"- Win rate: {best_row['win_rate']:.2%}",
            f"- Mean predicted return: {best_row['mean_pred']:.4f}",
            f"- Mean XGB score: {best_row['mean_xgb']:.4f}",
            f"- Selection score: {best_row['score']:.4f}",
            "",
        ])

    report_lines.append("## Top 20 Threshold Pairs")
    if top_ranges.empty:
        report_lines.append("No threshold pairs met the minimum sample requirement.")
    else:
        for _, row in top_ranges.iterrows():
            report_lines.append(
                f"- pred>={row['pred_min']:.6g}, xgb>={row['xgb_min']:.6g} | n={int(row['n'])} | mean_actual={row['mean_actual']:.4f} | win_rate={row['win_rate']:.2%}"
            )

    report_lines.extend([
        "",
        "## Files",
        f"- Threshold grid: {threshold_csv.name}",
        f"- Score bands: {band_csv.name}",
        f"- Heatmap: {heatmap_path.name if heatmap_path else 'not generated'}",
        f"- Band heatmap: {band_heatmap_path.name if band_heatmap_path else 'not generated'}",
    ])

    (outdir / "analysis_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "generated_utc": generated_utc,
        "input_files": [str(path) for path in files],
        "rows_analyzed": int(len(data)),
        "detected_columns": {"pred": pred_col, "xgb": xgb_col, "actual": actual_col},
        "best_threshold_pair": best_row,
        "artifacts": {
            "threshold_csv": str(threshold_csv),
            "band_csv": str(band_csv),
            "heatmap": str(heatmap_path) if heatmap_path else None,
            "band_heatmap": str(band_heatmap_path) if band_heatmap_path else None,
            "report": str(outdir / "analysis_report.md"),
        },
    }
    (outdir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Analysis written to: {outdir}")
    print(f"Best min_pred_return: {best_row.get('pred_min')}")
    print(f"Best min_xgb_score: {best_row.get('xgb_min')}")


if __name__ == "__main__":
    main()
