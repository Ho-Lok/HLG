#!/usr/bin/env python3
"""AI-first stock screening, then Sentinel model cross-check.

Workflow:
1) Load latest prediction universe (final_ensemble_results_live_v1.csv)
2) Build an AI screening pool that excludes model outputs
3) Ask TradingAgents to screen which tickers are worthy to buy
4) Cross-check AI picks against Sentinel model thresholds
5) Write ranked CSV outputs and a markdown summary report
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import torch
import xgboost as xgb
import json
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FORWARD_MODULE_PATH = os.path.join(SCRIPT_DIR, "0007_1forward.py")
DEFAULT_INPUT_CSV = os.path.join(SCRIPT_DIR, "0007_forward_output", "0007_1", "final_ensemble_results_live_v1.csv")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "0007_forward_output", "0007_4")

USER_MODEL = "deepseek-v4-pro"
USER_PREFILTER_TOP_N = 40
USER_AGENT_TOP_N = 10
USER_AI_CONFIDENCE_MIN = 0.55


def _apply_model_override(model: str | None) -> None:
    if model and str(model).strip():
        os.environ["TRADINGAGENTS_DEEP_MODEL"] = str(model).strip()


def _load_forward_module():
    spec = importlib.util.spec_from_file_location("sentinel_forward", FORWARD_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load forward module from {FORWARD_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_forward(python_bin: str) -> None:
    cmd = [python_bin, "-u", os.path.join(SCRIPT_DIR, "0007_1forward.py")]
    print("Running forward generation first:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def _prepare_latest_snapshot(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    snapshot_df = df.copy()
    if "Date" in snapshot_df.columns:
        parsed = pd.to_datetime(snapshot_df["Date"], errors="coerce")
        if parsed.notna().any():
            latest_date = parsed.max().normalize()
            snapshot_df = snapshot_df[parsed.dt.normalize() == latest_date].copy()
            date_str = latest_date.strftime("%Y-%m-%d")
            return snapshot_df, date_str
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return snapshot_df, date_str


def _prepare_ai_pool(df: pd.DataFrame, min_turnover: float, prefilter_top_n: int) -> pd.DataFrame:
    pool = df.copy()

    defaults = {
        "Ticker": "",
        "Current_Price": np.nan,
        "Current_Volume": np.nan,
        "Dollar_Turnover": np.nan,
        "Recent_20D_Return_%": np.nan,
        "FinBERT_Pos": np.nan,
        "FinBERT_Neg": np.nan,
        "News_Corpus": "",
        "News_Corpus_Len": 0,
        "Date": "",
    }
    for col, default_val in defaults.items():
        if col not in pool.columns:
            pool[col] = default_val

    pool["Ticker"] = pool["Ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    pool = pool[pool["Ticker"] != ""].copy()

    pool["Current_Price"] = pd.to_numeric(pool["Current_Price"], errors="coerce")
    pool["Current_Volume"] = pd.to_numeric(pool["Current_Volume"], errors="coerce")
    pool["Dollar_Turnover"] = pd.to_numeric(pool["Dollar_Turnover"], errors="coerce")

    missing_turnover = pool["Dollar_Turnover"].isna()
    pool.loc[missing_turnover, "Dollar_Turnover"] = (
        pool.loc[missing_turnover, "Current_Price"] * pool.loc[missing_turnover, "Current_Volume"]
    )

    if min_turnover > 0:
        filtered = pool[pool["Dollar_Turnover"] >= float(min_turnover)].copy()
        if not filtered.empty:
            pool = filtered

    pool = pool.sort_values(
        by=["Dollar_Turnover", "News_Corpus_Len"],
        ascending=[False, False],
        na_position="last",
    )

    if prefilter_top_n > 0:
        pool = pool.head(int(prefilter_top_n)).copy()

    # Keep only non-model fields for AI-first screening.
    ai_cols = [
        "Date",
        "Ticker",
        "Current_Price",
        "Current_Volume",
        "Dollar_Turnover",
        "Recent_20D_Return_%",
        "FinBERT_Pos",
        "FinBERT_Neg",
        "News_Corpus",
        "News_Corpus_Len",
    ]
    ai_pool = pool[ai_cols].drop_duplicates(subset=["Ticker"], keep="first").copy()
    return ai_pool


def _run_ai_screen(forward, ai_pool: pd.DataFrame, date_str: str, agent_top_n: int) -> pd.DataFrame:
    if ai_pool.empty:
        return pd.DataFrame(columns=["Ticker", "Agent_Status", "Agent_Recommendation", "Agent_Confidence", "Agent_Pass"])

    old_top_n = getattr(forward, "TOP_N_CROSSCHECK", 3)
    try:
        # Cap actual TradingAgents calls to the requested agent_top_n.
        # prefilter_top_n controls the candidate universe; agent_top_n controls the LLM workload.
        agent_limit = max(1, int(agent_top_n))
        forward.TOP_N_CROSSCHECK = agent_limit
        agent_df = forward.run_tradingagents_crosscheck(ai_pool.head(agent_limit), date_str)
    finally:
        forward.TOP_N_CROSSCHECK = old_top_n

    if agent_df.empty:
        return agent_df

    agent_df = agent_df.copy()
    agent_df["Agent_Recommendation"] = agent_df["Agent_Recommendation"].astype(str).str.strip().str.lower()
    agent_df["Agent_Confidence"] = pd.to_numeric(agent_df["Agent_Confidence"], errors="coerce")
    agent_df["Agent_Status"] = agent_df["Agent_Status"].astype(str).str.strip().str.lower()
    return agent_df


def _top_ai_buy_recommendations(agent_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Return the strongest AI buy/long recommendations, independent of model cross-checking."""
    if agent_df.empty:
        return pd.DataFrame(columns=["Ticker", "Agent_Recommendation", "Agent_Confidence", "Agent_Status"])

    result = agent_df.copy()
    result["Agent_Recommendation"] = result["Agent_Recommendation"].astype(str).str.strip().str.lower()
    result["Agent_Confidence"] = pd.to_numeric(result["Agent_Confidence"], errors="coerce")
    result["Agent_Status"] = result["Agent_Status"].astype(str).str.strip().str.lower()

    result = result[
        result["Agent_Status"].eq("ok")
        & result["Agent_Recommendation"].isin({"buy", "long"})
    ].copy()

    if result.empty:
        return result

    sort_cols = [col for col in ["Agent_Confidence", "Dollar_Turnover", "Pred_Return_%"] if col in result.columns]
    if sort_cols:
        result = result.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    else:
        result = result.sort_values(by=["Ticker"], ascending=[True], na_position="last")

    return result.head(max(1, int(top_n))).reset_index(drop=True)


def _top_ai_ideas(agent_df: pd.DataFrame, top_n: int = 3) -> tuple[pd.DataFrame, bool]:
    """Return the top AI ideas, preferring buys but falling back to highest-confidence ideas."""
    buy_df = _top_ai_buy_recommendations(agent_df, top_n=top_n)
    if not buy_df.empty:
        return buy_df, True

    if agent_df.empty:
        return pd.DataFrame(columns=["Ticker", "Agent_Recommendation", "Agent_Confidence", "Agent_Status"]), False

    result = agent_df.copy()
    result["Agent_Recommendation"] = result["Agent_Recommendation"].astype(str).str.strip().str.lower()
    result["Agent_Confidence"] = pd.to_numeric(result["Agent_Confidence"], errors="coerce")
    result["Agent_Status"] = result["Agent_Status"].astype(str).str.strip().str.lower()

    result = result[result["Agent_Status"].eq("ok")].copy()
    if result.empty:
        return result, False

    sort_cols = [col for col in ["Agent_Confidence", "Dollar_Turnover", "Pred_Return_%"] if col in result.columns]
    if sort_cols:
        result = result.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    else:
        result = result.sort_values(by=["Ticker"], ascending=[True], na_position="last")

    return result.head(max(1, int(top_n))).reset_index(drop=True), False


def _top_ai_buy_model_analysis(cross_df: pd.DataFrame, top_ai_buys: pd.DataFrame) -> pd.DataFrame:
    """Return the model cross-check rows for the AI-picked top buy ideas, preserving AI rank order."""
    if cross_df.empty or top_ai_buys.empty:
        return pd.DataFrame()

    tickers = [str(t).strip().upper() for t in top_ai_buys["Ticker"].tolist() if str(t).strip()]
    if not tickers:
        return pd.DataFrame()

    analysis = cross_df[cross_df["Ticker"].astype(str).str.strip().str.upper().isin(tickers)].copy()
    if analysis.empty:
        return analysis

    analysis["_AI_Rank"] = pd.Categorical(
        analysis["Ticker"].astype(str).str.strip().str.upper(),
        categories=tickers,
        ordered=True,
    )
    analysis = analysis.sort_values(by=["_AI_Rank"], ascending=True).drop(columns=["_AI_Rank"])
    return analysis.reset_index(drop=True)


def _load_models_from_forward(forward):
    """Load PatchTST and XGBoost models and scalers from forward module paths."""
    PT_MODEL_PATH = getattr(forward, "PT_MODEL_PATH", "0004_saved_models_20/patchtst_live_20.pth")
    XGB_MODEL_PATH = getattr(forward, "XGB_MODEL_PATH", "0004_saved_models_20/xgboost_live_20.json")
    PT_TARGET_STATS_PATH = getattr(forward, "PT_TARGET_STATS_PATH", "0004_saved_models_20/patchtst_target_stats.json")
    SCALER_PATH = getattr(forward, "SCALER_PATH", "0003_scalers_20/global_scaler.pkl")
    SCHEMA_PATH = getattr(forward, "SCHEMA_PATH", "0003_scalers_20/schema_cols.pkl")
    VECTOR_DIR = getattr(forward, "VECTOR_DIR", "0002_final_vectors")
    PRICE_DIR = getattr(forward, "PRICE_DIR", "001_final_db_daily")

    # Load scalers / schema
    pca = None
    pca_scaler = None
    global_scaler = None
    schema_cols = None
    if os.path.exists(SCALER_PATH):
        global_scaler = joblib.load(SCALER_PATH)
    if os.path.exists(SCHEMA_PATH):
        schema_cols = joblib.load(SCHEMA_PATH)

    # Target stats
    pt_target_mean = 0.0
    pt_target_std = 1.0
    if os.path.exists(PT_TARGET_STATS_PATH):
        with open(PT_TARGET_STATS_PATH, "r", encoding="utf-8") as f:
            stats = json.load(f)
            pt_target_mean = float(stats.get("mean", 0.0))
            pt_target_std = float(stats.get("std", 1.0)) or 1.0

    # Load PatchTST model (architecture from forward)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PT = None
    try:
        PatchTST = getattr(forward, "PatchTST")
        if schema_cols is None:
            raise RuntimeError("schema cols not found; cannot instantiate PatchTST")
        PT = PatchTST(num_features=len(schema_cols), seq_len=60).to(DEVICE)
        state = torch.load(PT_MODEL_PATH, map_location=DEVICE, weights_only=True)
        if list(state.keys())[0].startswith('module.'):
            state = {k[7:]: v for k, v in state.items()}
        PT.load_state_dict(state)
        PT.eval()
    except Exception:
        PT = None

    # Load XGBoost
    XGB = None
    try:
        XGB = xgb.Booster()
        XGB.load_model(XGB_MODEL_PATH)
    except Exception:
        XGB = None

    return {
        "pt_model": PT,
        "xgb_model": XGB,
        "global_scaler": global_scaler,
        "schema_cols": schema_cols,
        "pt_target_mean": pt_target_mean,
        "pt_target_std": pt_target_std,
        "vector_dir": VECTOR_DIR,
        "price_dir": PRICE_DIR,
        "device": DEVICE,
    }


def _predict_for_ticker(ticker: str, forward, models: dict):
    """Run the Sentinel models for one ticker using local vector and price files.

    Returns dict with Pred_Return_Raw, Pred_Return_Model, Pred_Return_%, XGB_Rank_Score
    or empty dict on failure.
    """
    ticker = str(ticker).upper().replace('.', '-')
    vec_path = Path(models["vector_dir"]) / f"{ticker}_vec.csv"
    price_path = Path(models["price_dir"]) / f"{ticker}_daily.csv"
    schema_cols = models.get("schema_cols")
    global_scaler = models.get("global_scaler")
    pt_model = models.get("pt_model")
    xgb_model = models.get("xgb_model")
    pt_mean = models.get("pt_target_mean", 0.0)
    pt_std = models.get("pt_target_std", 1.0)

    if not vec_path.exists() or not price_path.exists() or schema_cols is None or global_scaler is None:
        return {}

    try:
        df_vec = pd.read_csv(vec_path)
        if df_vec.empty:
            return {}
        latest_vec = df_vec.iloc[-1]
        finbert_vec = [float(latest_vec.get(f"dim_{i+1}", 0.0)) for i in range(int(getattr(forward, 'VECTOR_DIM', 28)))]
        sents = [float(latest_vec.get('sent_pos', 0.0)), float(latest_vec.get('sent_neg', 0.0)), float(latest_vec.get('sent_neu', 0.0))]

        df_price = pd.read_csv(price_path, index_col=0)
        if df_price.empty or len(df_price) < 62:
            return {}
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = [c[0] for c in df_price.columns]
        df_price.columns = [c.lower() for c in df_price.columns]
        df_price.index = pd.to_datetime(df_price.index, errors='coerce')
        df_price = forward.add_technical_features(df_price)

        # join macro if available
        df_macro = forward.get_live_macro_data()
        df = df_price.join(df_macro, how='left').ffill()
        for i, val in enumerate(finbert_vec): df[f'dim_{i+1}'] = val
        df['sent_pos'] = sents[0]
        df['sent_neg'] = sents[1]
        df['sent_neu'] = sents[2]

        for i in range(int(getattr(forward, 'VECTOR_DIM', 28))):
            df[f'market_dim_{i+1}'] = float(0.0)
        df['market_sent_pos'] = 0.0
        df['market_sent_neg'] = 0.0
        df['market_sent_neu'] = 0.0

        df_aligned = df.reindex(columns=schema_cols, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        recent_60 = df_aligned.values[-60:]
        if len(recent_60) != 60:
            return {}

        data_scaled = global_scaler.transform(recent_60)
        x_3d = torch.tensor(np.array([data_scaled]), dtype=torch.float32).to(models.get('device'))
        feats = np.concatenate([data_scaled[-1], data_scaled.mean(axis=0), data_scaled.std(axis=0)])
        x_1d = xgb.DMatrix(np.array([feats], dtype=np.float32))

        with torch.no_grad():
            pred_ret_raw = float(pt_model(x_3d).cpu().numpy().flatten()[0]) if pt_model is not None else np.nan
            pred_ret_model = pred_ret_raw * pt_std + pt_mean
            pred_rank = float(xgb_model.predict(x_1d)[0]) if xgb_model is not None else np.nan

        recent_20d_return = forward.estimate_recent_return_pct(df_price, horizon=20)
        pred_ret = pred_ret_model
        used_fallback = False
        if recent_20d_return is not None:
            if not np.isfinite(pred_ret_model) or abs(pred_ret_model) < 0.25:
                pred_ret = 0.5 * pred_ret_model + 0.5 * recent_20d_return
                used_fallback = True

        return {
            "Pred_Return_Raw": pred_ret_raw,
            "Pred_Return_Model": pred_ret_model,
            "Pred_Return_%": pred_ret,
            "XGB_Rank_Score": pred_rank,
            "Used_Fallback": used_fallback,
        }
    except Exception:
        return {}


def _crosscheck_with_model(
    model_df: pd.DataFrame,
    agent_df: pd.DataFrame,
    ai_confidence_min: float,
    min_pred_return: float,
    min_xgb_score: float,
) -> pd.DataFrame:
    if agent_df.empty:
        return pd.DataFrame()

    keep_cols = [
        "Date",
        "Ticker",
        "Current_Price",
        "Dollar_Turnover",
        "Recent_20D_Return_%",
        "Pred_Return_%",
        "XGB_Rank_Score",
        "FinBERT_Pos",
        "FinBERT_Neg",
        "News_Corpus",
    ]

    base = model_df.copy()
    for col in keep_cols:
        if col not in base.columns:
            base[col] = np.nan if col != "News_Corpus" else ""

    base = base[keep_cols].drop_duplicates(subset=["Ticker"], keep="first").copy()
    base["Pred_Return_%"] = pd.to_numeric(base["Pred_Return_%"], errors="coerce")
    base["XGB_Rank_Score"] = pd.to_numeric(base["XGB_Rank_Score"], errors="coerce")

    merged = agent_df.merge(base, on="Ticker", how="left")

    rec = merged["Agent_Recommendation"].astype(str).str.lower()
    conf = pd.to_numeric(merged["Agent_Confidence"], errors="coerce")
    status = merged["Agent_Status"].astype(str).str.lower()

    ai_rec_pass = rec.isin({"buy", "long"})
    ai_conf_pass = conf.isna() | (conf >= float(ai_confidence_min))
    ai_status_ok = status.eq("ok")
    merged["AI_Pass"] = ai_rec_pass & ai_conf_pass & ai_status_ok

    merged["Model_Pass"] = (
        (merged["Pred_Return_%"] >= float(min_pred_return))
        & (merged["XGB_Rank_Score"] >= float(min_xgb_score))
    )

    merged["Crosscheck_Result"] = np.select(
        [
            merged["AI_Pass"] & merged["Model_Pass"],
            merged["AI_Pass"] & ~merged["Model_Pass"],
        ],
        [
            "BUY_CONFIRMED",
            "AI_ONLY_WATCH",
        ],
        default="AI_REJECT",
    )

    priority = {
        "BUY_CONFIRMED": 0,
        "AI_ONLY_WATCH": 1,
        "AI_REJECT": 2,
    }
    merged["Priority"] = merged["Crosscheck_Result"].map(priority).fillna(9)

    merged = merged.sort_values(
        by=["Priority", "Agent_Confidence", "XGB_Rank_Score", "Pred_Return_%"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return merged


def _write_outputs(output_dir: str, date_str: str, ai_pool: pd.DataFrame, agent_df: pd.DataFrame, cross_df: pd.DataFrame) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    ai_pool_path = os.path.join(output_dir, f"0007_4_ai_pool_{date_str}.csv")
    ai_result_path = os.path.join(output_dir, f"0007_4_ai_screen_result_{date_str}.csv")
    crosscheck_path = os.path.join(output_dir, f"0007_4_ai_model_crosscheck_{date_str}.csv")
    confirmed_path = os.path.join(output_dir, f"0007_4_buy_confirmed_{date_str}.csv")
    report_path = os.path.join(output_dir, f"0007_4_ai_model_crosscheck_report_{date_str}.md")

    ai_pool.to_csv(ai_pool_path, index=False)
    agent_df.to_csv(ai_result_path, index=False)
    cross_df.to_csv(crosscheck_path, index=False)

    confirmed = cross_df[cross_df["Crosscheck_Result"] == "BUY_CONFIRMED"].copy()
    confirmed.to_csv(confirmed_path, index=False)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = []
    lines.append("# 0007_4 AI-First Screen + Model Cross-Check")
    lines.append(f"- Date: {date_str}")
    lines.append(f"- Generated At (UTC): {generated_at}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- AI Pool Size: {len(ai_pool)}")
    lines.append(f"- AI Screened: {len(agent_df)}")
    top_ai_ideas, top_ai_is_buy_only = _top_ai_ideas(agent_df, top_n=3)
    lines.append(f"- AI Buy Ideas (Top 3): {len(top_ai_ideas)}")
    lines.append(f"- BUY_CONFIRMED (AI + Model): {int((cross_df['Crosscheck_Result'] == 'BUY_CONFIRMED').sum())}")
    lines.append(f"- AI_ONLY_WATCH: {int((cross_df['Crosscheck_Result'] == 'AI_ONLY_WATCH').sum())}")
    lines.append(f"- AI_REJECT: {int((cross_df['Crosscheck_Result'] == 'AI_REJECT').sum())}")
    lines.append("")

    lines.append("## Top 3 AI Buy Ideas")
    if top_ai_ideas.empty:
        lines.append("No AI ideas were returned in this screen.")
    else:
        if top_ai_is_buy_only:
            lines.append("These are the AI agents' direct buy/long calls.")
        else:
            lines.append("No buy/long calls were returned, so this section falls back to the highest-confidence AI ideas.")
        buy_cols = [
            "Ticker",
            "Agent_Recommendation",
            "Agent_Confidence",
            "Agent_Status",
        ]
        if "Agent_Input_Summary" in top_ai_ideas.columns:
            buy_cols.append("Agent_Input_Summary")
        if "Dollar_Turnover" in top_ai_ideas.columns:
            buy_cols.append("Dollar_Turnover")
        if "Pred_Return_%" in top_ai_ideas.columns:
            buy_cols.append("Pred_Return_%")
        lines.extend(_dataframe_to_markdown_lines(top_ai_ideas[buy_cols]))
    lines.append("")

    top_ai_buy_model = _top_ai_buy_model_analysis(cross_df, top_ai_ideas)
    lines.append("## Top 3 AI Buy Ideas - Model Analysis")
    if top_ai_buy_model.empty:
        lines.append("No model rows were found for the selected AI ideas.")
    else:
        model_cols = [
            "Ticker",
            "Crosscheck_Result",
            "Agent_Recommendation",
            "Agent_Confidence",
            "AI_Pass",
            "Model_Pass",
            "Pred_Return_%",
            "XGB_Rank_Score",
            "Current_Price",
            "Dollar_Turnover",
            "Recent_20D_Return_%",
        ]
        available_model_cols = [c for c in model_cols if c in top_ai_buy_model.columns]
        lines.extend(_dataframe_to_markdown_lines(top_ai_buy_model[available_model_cols]))
    lines.append("")

    top_cols = [
        "Ticker",
        "Crosscheck_Result",
        "Agent_Recommendation",
        "Agent_Confidence",
        "Pred_Return_%",
        "XGB_Rank_Score",
        "Current_Price",
        "Dollar_Turnover",
    ]
    available_top_cols = [c for c in top_cols if c in cross_df.columns]
    preview = cross_df[available_top_cols].head(20)

    lines.append("## Top 20 Ranked")
    if preview.empty:
        lines.append("No rows to display.")
    else:
        lines.extend(_dataframe_to_markdown_lines(preview))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return {
        "ai_pool": ai_pool_path,
        "ai_result": ai_result_path,
        "crosscheck": crosscheck_path,
        "confirmed": confirmed_path,
        "report": report_path,
    }


def _dataframe_to_markdown_lines(df: pd.DataFrame) -> list[str]:
    """Render a compact markdown table without requiring external packages."""
    if df.empty:
        return ["No rows to display."]

    cols = [str(c) for c in df.columns]

    def _fmt(v: object) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = [_fmt(row[c]) for c in df.columns]
        rows.append("| " + " | ".join(cells) + " |")

    return [header, divider] + rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-first ticker screening using TradingAgents, then Sentinel model cross-check.",
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Input universe CSV path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for 0007_4 results")
    parser.add_argument("--run-forward", action="store_true", help="Run 0007_1forward.py first to refresh input CSV")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used for --run-forward")
    parser.add_argument("--model", default=USER_MODEL, help="TradingAgents model (for example gemini-2.5-pro-thinking)")
    parser.add_argument("--prefilter-top-n", type=int, default=USER_PREFILTER_TOP_N, help="Liquidity-first candidate pool size before AI screening")
    parser.add_argument("--agent-top-n", type=int, default=USER_AGENT_TOP_N, help="How many tickers AI will evaluate")
    parser.add_argument("--ai-confidence-min", type=float, default=USER_AI_CONFIDENCE_MIN, help="Minimum AI confidence to count as AI_Pass")
    parser.add_argument("--model-min-pred-return", type=float, default=np.nan, help="Override model min Pred_Return_% threshold")
    parser.add_argument("--model-min-xgb-score", type=float, default=np.nan, help="Override model min XGB_Rank_Score threshold")
    args = parser.parse_args()

    _apply_model_override(args.model)

    if args.run_forward:
        _run_forward(args.python_bin)

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    forward = _load_forward_module()
    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {args.input_csv}")

    latest_df, date_str = _prepare_latest_snapshot(df)

    min_turnover = float(getattr(forward, "MIN_TURNOVER_DOLLARS", 5_000_000.0))
    min_pred_return = float(getattr(forward, "MIN_PRED_RETURN", 1.0))
    min_xgb_score = float(getattr(forward, "MIN_XGB_SCORE", -0.25))

    if pd.notna(args.model_min_pred_return):
        min_pred_return = float(args.model_min_pred_return)
    if pd.notna(args.model_min_xgb_score):
        min_xgb_score = float(args.model_min_xgb_score)

    ai_pool = _prepare_ai_pool(latest_df, min_turnover=min_turnover, prefilter_top_n=args.prefilter_top_n)
    if ai_pool.empty:
        raise ValueError("AI pool is empty after turnover and prefilter steps.")

    print(f"AI pool size: {len(ai_pool)} | agent_top_n: {max(1, int(args.agent_top_n))}")
    agent_df = _run_ai_screen(forward, ai_pool, date_str=date_str, agent_top_n=args.agent_top_n)
    cross_df = _crosscheck_with_model(
        model_df=latest_df,
        agent_df=agent_df,
        ai_confidence_min=args.ai_confidence_min,
        min_pred_return=min_pred_return,
        min_xgb_score=min_xgb_score,
    )

    paths = _write_outputs(args.output_dir, date_str, ai_pool, agent_df, cross_df)

    print("\n0007_4 outputs written:")
    print(paths["ai_pool"])
    print(paths["ai_result"])
    print(paths["crosscheck"])
    print(paths["confirmed"])
    print(paths["report"])

    confirmed_count = int((cross_df["Crosscheck_Result"] == "BUY_CONFIRMED").sum())
    print("\nCounts:")
    print(f"- BUY_CONFIRMED: {confirmed_count}")
    print(f"- AI_ONLY_WATCH: {int((cross_df['Crosscheck_Result'] == 'AI_ONLY_WATCH').sum())}")
    print(f"- AI_REJECT: {int((cross_df['Crosscheck_Result'] == 'AI_REJECT').sum())}")


if __name__ == "__main__":
    main()
