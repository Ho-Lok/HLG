#!/usr/bin/env python3
"""
DAILY ORCHESTRATOR (0007_0run_daily.py)

Orchestrates the complete forward prediction pipeline:
    1. 0007_1forward.py              -> Generate predictions
    2. 0007_2forward_with_tracker.py -> Track portfolio performance
    3. 0007_3threshold_analysis.py   -> Optimize thresholds
    4. 0007_4ai_screen_then_model_crosscheck.py -> AI screen + model cross-check

USAGE - Full workflow:
    python 0007_0run_daily.py                         # Run all steps
    python 0007_0run_daily.py --skip-forward          # Use existing predictions
    python 0007_0run_daily.py --skip-distribution     # Skip distribution analysis
    python 0007_0run_daily.py --date 2026-05-13       # Set output date

USAGE - Individual steps (run independently):
    python 0007_1forward.py                           # Generate predictions only
    python 0007_2forward_with_tracker.py              # Track trades only
    python 0007_3threshold_analysis.py                # Analyze thresholds only
    python 0007_4ai_screen_then_model_crosscheck.py   # AI screen + model cross-check only

OUTPUTS in 0007_forward_output/:
    - final_ensemble_results_live_v1.csv     (all predictions)
    - merged_forward_trades.csv              (executed trade signals)
    - merged_forward_current_by_ticker.csv   (current holdings)
    - merged_forward_summary.csv             (portfolio summary)
    - threshold_analysis/YYYY-MM-DD/         (optimal parameters)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import json

import numpy as np
import pandas as pd


try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

# ==========================================
# USER-EDITABLE SETTINGS (Edit these directly instead of using CLI flags)
# ==========================================
T = True
F = False

USER_LLM_MODEL = "deepseek-v4-pro"  # Options: deepseek-v4-pro, deepseek-v4-flash
USER_RUN_FORWARD = T             # Toggle behavior: True = run 0007_1forward.py, False = skip
USER_RUN_DISTRIBUTION = T        # Toggle behavior: True = run distribution analysis, False = skip
USER_RUN_0007_4 = T             # Toggle behavior: True = run 0007_4 AI-screen + model cross-check, False = skip
USER_DATE_OVERRIDE = ""             # Set to YYYY-MM-DD to override today's date, or leave empty for auto


def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def run_cmd(command: list[str], cwd: Path, env_vars: dict | None = None) -> None:
    log("RUN: " + " ".join(command))
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    subprocess.run(command, cwd=str(cwd), check=True, env=env)


def find_candidate_files(workdir: Path):
    patterns = [
        "**/*Forward*.csv",
        "**/final_ensemble_results_*.csv",
        "**/*ensemble_results*.csv",
    ]
    files = set()
    for pattern in patterns:
        for file_path in workdir.glob(pattern):
            if file_path.is_file():
                files.add(file_path)
    return sorted(files)


def detect_columns(df: pd.DataFrame):
    pred_col = next((c for c in df.columns if c in ["Pred_Return_%", "Pred_Return", "Predicted_Alpha_%", "Pred"]), None)
    xgb_col = next((c for c in df.columns if c in ["XGB_Rank_Score", "XGB_Rank", "Rank_Score", "Rank"]), None)
    act_col = next((c for c in df.columns if c in ["Actual_20D_Return_%", "Actual_15D_Return_%", "Actual_Return_%", "Actual_Alpha_%", "Actual_Return", "Actual"]), None)
    if not pred_col:
        for c in df.columns:
            if "pred" in c.lower() or "alpha" in c.lower():
                pred_col = c
                break
    if not xgb_col:
        for c in df.columns:
            if "score" in c.lower() or "rank" in c.lower():
                xgb_col = c
                break
    if not act_col:
        for c in df.columns:
            if "actual" in c.lower() or "return" in c.lower():
                act_col = c
                break

    cols = [c for c in (pred_col, xgb_col, act_col) if c]
    return cols


def safe_numeric(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")


def analyze_file(filepath: Path, outdir: Path):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return {"file": str(filepath), "error": f"read_error: {e}"}

    cols = detect_columns(df)
    if not cols:
        return {"file": str(filepath), "error": "no_numeric_columns_detected"}

    generated_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = {"file": str(filepath), "cols": cols, "date": generated_utc, "generated_utc": generated_utc}
    stats_frames = []

    for col in cols:
        ser = safe_numeric(df[col]).dropna()
        if ser.empty:
            report[col] = "no_data"
            continue

        desc = ser.describe()
        quantiles = ser.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        _, bin_edges = np.histogram(ser, bins=20)

        plot_name = None
        if plt is not None and sns is not None:
            plt.figure(figsize=(8, 4))
            sns.histplot(ser, bins=bin_edges, kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plot_path = outdir / f"{filepath.stem}__{col.replace(' ', '_')}.png"
            plt.savefig(plot_path)
            plt.close()
            plot_name = plot_path.name

        stats = pd.DataFrame(desc).T
        stats["generated_utc"] = generated_utc
        stats = stats.assign(**{f"q_{int(q*100)}": quantiles.loc[q] for q in quantiles.index})
        stats_frames.append((col, stats))

        report[col] = {
            "count": int(desc.get("count", 0)),
            "mean": float(desc.get("mean", np.nan)),
            "std": float(desc.get("std", np.nan)),
            "min": float(desc.get("min", np.nan)),
            "25%": float(desc.get("25%", np.nan)),
            "50%": float(desc.get("50%", np.nan)),
            "75%": float(desc.get("75%", np.nan)),
            "max": float(desc.get("max", np.nan)),
            "plot": plot_name,
        }

    summary_rows = []
    for col, sf in stats_frames:
        row = sf.copy()
        row.insert(0, "column", col)
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.concat(summary_rows, ignore_index=True)
        summary_csv = outdir / f"{filepath.stem}__summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        report["summary_csv"] = str(summary_csv.name)

    md = [f"# Analysis for {filepath.name}", "", f"Generated At (UTC): {report['generated_utc']}", "", "## Columns analyzed", ""]
    for c in cols:
        md.append(f"- {c}")
    md.append("")
    for c in cols:
        md.append(f"## {c}")
        if isinstance(report.get(c), dict):
            for k, v in report[c].items():
                md.append(f"- **{k}**: {v}")
        else:
            md.append(f"- {report.get(c)}")
        md.append("")

    md_path = outdir / f"{filepath.stem}__report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    return report


def run_daily_distribution_analysis(workdir: Path, outdir: Path, date_str: str) -> None:
    outdir = outdir / date_str
    outdir.mkdir(parents=True, exist_ok=True)

    files = find_candidate_files(workdir)
    if not files:
        log("No candidate CSV files found for daily distribution analysis")
        return

    reports = []
    for file_path in files:
        log(f"Analyzing {file_path}...")
        reports.append(analyze_file(file_path, outdir))

    master = outdir / "master_report.json"
    master.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    log(f"Distribution analysis written to {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily 0007 workflow in one command.")
    default_run_forward = bool(USER_RUN_FORWARD)
    default_run_distribution = bool(USER_RUN_DISTRIBUTION)
    default_run_0007_4 = bool(USER_RUN_0007_4)

    parser.add_argument("--date", default=USER_DATE_OVERRIDE or datetime.now().strftime("%Y-%m-%d"), help="Date folder override (YYYY-MM-DD)")
    parser.add_argument("--skip-forward", dest="run_forward", action="store_false", default=default_run_forward, help="Skip 0007_1forward.py")
    parser.add_argument("--run-forward", dest="run_forward", action="store_true", help="Force run 0007_1forward.py")
    parser.add_argument("--skip-distribution", dest="run_distribution", action="store_false", default=default_run_distribution, help="Skip built-in daily distribution analysis")
    parser.add_argument("--run-distribution", dest="run_distribution", action="store_true", help="Force run built-in daily distribution analysis")
    parser.add_argument("--run-0007-4", dest="run_0007_4", action="store_true", default=default_run_0007_4, help="Run 0007_4ai_screen_then_model_crosscheck.py after 0007 pipeline")
    parser.add_argument("--skip-0007-4", dest="run_0007_4", action="store_false", help="Skip 0007_4ai_screen_then_model_crosscheck.py")
    parser.add_argument("--python-bin", default=sys.executable or "python3", help="Python executable to use")
    parser.add_argument("--model", default=USER_LLM_MODEL, help="LLM model to use (e.g., gemini-2.5-pro, gemini-2.5-pro-thinking, gemini-2.5-flash)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    output_root = base_dir / "0007_forward_output"
    step_0007_1_dir = output_root / "0007_1"
    step_0007_2_dir = output_root / "0007_2"
    step_0007_3_dir = output_root / "0007_3"
    step_0007_4_dir = output_root / "0007_4"
    dist_dir = output_root / "0007_0_distribution"

    for path in [output_root, step_0007_1_dir, step_0007_2_dir, step_0007_3_dir, step_0007_4_dir, dist_dir]:
        path.mkdir(parents=True, exist_ok=True)

    log("Starting daily 0007 workflow")
    log(f"BASE_DIR={base_dir}")
    log(f"LLM Model: {args.model}")
    log(f"Run Forward: {args.run_forward}")
    log(f"Run Distribution: {args.run_distribution}")
    log(f"Run 0007_4: {args.run_0007_4}")
    log("Workflow order: 0007_1forward -> 0007_2forward_with_tracker -> 0007_3threshold_analysis -> [optional] 0007_4ai_screen_then_model_crosscheck -> daily distribution analysis")

    python_bin = args.python_bin
    model_env = {
        "TRADINGAGENTS_DEEP_MODEL": args.model,
        "TRADINGAGENTS_QUICK_MODEL": args.model,
        "FORWARD_OUTPUT_DIR": str(step_0007_1_dir),
    }

    if args.run_forward:
        run_cmd([python_bin, "-u", str(base_dir / "0007_1forward.py")], cwd=base_dir, env_vars=model_env)
    else:
        log("Skipping forward generation")

    # Ensure forward outputs exist; if they don't and user skipped forward, run 0007_1 to generate them
    required_file = step_0007_1_dir / "final_ensemble_results_live_v1.csv"
    if not required_file.exists():
        log(f"Forward output missing: {required_file}")
        if not args.run_forward:
            log("Forward outputs missing while skipping forward — running 0007_1forward.py now to generate them.")
            run_cmd([python_bin, "-u", str(base_dir / "0007_1forward.py")], cwd=base_dir, env_vars=model_env)
        else:
            log("Forward requested but outputs still missing; continuing and downstream steps may fail.")

    run_cmd(
        [
            python_bin,
            "-u",
            str(base_dir / "0007_2forward_with_tracker.py"),
            "--no-run-forward",
            "--input-dir",
            str(step_0007_1_dir),
            "--output-dir",
            str(step_0007_2_dir),
        ],
        cwd=base_dir,
        env_vars=model_env,
    )

    run_cmd(
        [
            python_bin,
            "-u",
            str(base_dir / "0007_3threshold_analysis.py"),
            "--file",
            str(step_0007_1_dir / "final_ensemble_results_live_v1.csv"),
            "--outdir",
            str(step_0007_3_dir),
            "--date",
            args.date,
        ],
        cwd=base_dir,
    )

    if args.run_0007_4:
        run_cmd(
            [
                python_bin,
                "-u",
                str(base_dir / "0007_4ai_screen_then_model_crosscheck.py"),
                "--input-csv",
                str(step_0007_1_dir / "final_ensemble_results_live_v1.csv"),
                "--output-dir",
                str(step_0007_4_dir),
                "--model",
                args.model,
            ],
            cwd=base_dir,
            env_vars=model_env,
        )
    else:
        log("Skipping 0007_4 AI-screen + model cross-check")

    if args.run_distribution:
        run_daily_distribution_analysis(output_root, dist_dir, args.date)
    else:
        log("Skipping daily distribution analysis")
    
    log(f"✅ Daily workflow completed with model: {args.model}")

    log("Daily 0007 workflow completed successfully")


if __name__ == "__main__":
    main()
