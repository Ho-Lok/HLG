#!/usr/bin/env python3
"""
FORWARD TRACKER PIPELINE (0007_2forward_with_tracker.py)

USAGE:
    Run individually (uses existing predictions from 0007_1forward):
        python 0007_2forward_with_tracker.py
  
    Run with automatic forward generation:
        python 0007_2forward_with_tracker.py --run-forward
  
    With options:
        python 0007_2forward_with_tracker.py --help
  
    Run as part of daily workflow (via orchestrator):
        python 0007_0run_daily.py

OUTPUTS:
    - merged_forward_trades.csv (all executed trade signals with P&L)
    - merged_forward_current_by_ticker.csv (current portfolio holdings)
    - merged_forward_current_positions.csv (detailed position breakdown)
    - merged_forward_current_positions_history.csv (daily open-position snapshots)
    - merged_forward_summary.csv (portfolio-level summary statistics)

DEPENDENCIES:
    - final_ensemble_results_live_v1.csv (from 0007_1forward.py)
    - Historical price data in 001_final_db_daily/
    - SPY benchmark data for CAPM calculations

FEATURES:
    - Evaluates daily recommendations with max 1 buy per day
    - Tracks 20-day holding periods
    - Calculates CAPM alpha, Sharpe ratio, beta
    - Handles closed and open positions
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


BASE = Path(__file__).resolve().parent
DEFAULT_FORWARD = BASE / "0007_1forward.py"
OUTPUT_DIR = BASE / "0007_forward_output" / "0007_2"
FORWARD_INPUT_DIR = BASE / "0007_forward_output" / "0007_1"
DEFAULT_MAX_BUYS_PER_DAY = 1
MIN_TURNOVER_DOLLARS = 5_000_000


@dataclass
class TradeResult:
    signal_date: pd.Timestamp
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    holding_days_target: int
    trading_days_held: int
    status: str
    capital_per_trade: float
    shares: float
    pnl_usd: float
    return_pct: float
    pred_return: float
    rank_score: float
    benchmark_return_pct: float
    capm_beta: float
    capm_expected_return_pct: float
    capm_alpha_pct: float
    capm_alpha_raw: float
    capm_alpha_usd: float


def find_final_csv(outdir: Path) -> Path | None:
    candidates = [outdir / "final_ensemble_results_live_v1.csv", outdir / "final_ensemble_results.csv"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Look for ensemble files but explicitly exclude summary files
    for path in sorted(outdir.glob("*ensemble*.csv"), reverse=True):
        if "__summary" not in path.name and "_summary" not in path.name:
            return path
    return None


def find_top3_tracker_csv(outdir: Path) -> Path | None:
    tracker = outdir / "Sentinel_v1_Top_3_Tracker.csv"
    return tracker if tracker.exists() else None


def map_forward_to_tracker_df(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    cols = {c.lower(): c for c in df.columns}

    if "date" in cols:
        col_map["Date"] = cols["date"]
    for candidate in ("ticker", "symbol"):
        if candidate in cols:
            col_map["Ticker"] = cols[candidate]
            break
    for candidate in ("current_price", "currentprice", "price", "close"):
        if candidate in cols:
            col_map["Current_Price"] = cols[candidate]
            break
    for candidate in ("pred_return", "pred_return_%", "pred_return_pct", "pred"):
        if candidate in cols:
            col_map["Pred_Return"] = cols[candidate]
            break
    for candidate in ("xgb_rank_score", "xgb_rank", "rank_score", "rank", "xgb_score"):
        if candidate in cols:
            col_map["Rank_Score"] = cols[candidate]
            break
    for candidate in ("dollar_turnover", "turnover", "notional", "liquidity"):
        if candidate in cols:
            col_map["Dollar_Turnover"] = cols[candidate]
            break

    if "Pred_Return" not in col_map:
        for c in df.columns:
            if "pred" in c.lower() or "alpha" in c.lower():
                col_map["Pred_Return"] = c
                break
    if "Rank_Score" not in col_map:
        for c in df.columns:
            if "score" in c.lower() or "rank" in c.lower():
                col_map["Rank_Score"] = c
                break

    out = pd.DataFrame()
    for required in ["Date", "Ticker", "Current_Price", "Pred_Return", "Rank_Score"]:
        out[required] = df[col_map[required]] if required in col_map else pd.NA
    out["Dollar_Turnover"] = df[col_map["Dollar_Turnover"]] if "Dollar_Turnover" in col_map else pd.NA

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Ticker"] = out["Ticker"].astype(str)
    out["Dollar_Turnover"] = pd.to_numeric(out["Dollar_Turnover"], errors="coerce")
    return out


def load_tracker_recommendations(outdir: Path) -> tuple[pd.DataFrame, Path | None]:
    tracker_csv = find_top3_tracker_csv(outdir)
    if tracker_csv is not None:
        tracker_df = pd.read_csv(tracker_csv)
        if not tracker_df.empty and {"Date", "Ticker"}.issubset(tracker_df.columns):
            tracker_df["Date"] = pd.to_datetime(tracker_df["Date"], errors="coerce")
            tracker_df["Ticker"] = tracker_df["Ticker"].astype(str)
            return tracker_df[["Date", "Ticker"]].dropna(subset=["Date", "Ticker"]).copy(), tracker_csv

    final_csv = find_final_csv(outdir)
    if final_csv is None:
        return pd.DataFrame(columns=["Date", "Ticker"]), None

    df_preds = pd.read_csv(final_csv)
    recs = map_forward_to_tracker_df(df_preds)
    recs = recs[recs["Dollar_Turnover"].fillna(0) >= MIN_TURNOVER_DOLLARS].copy()
    recs = recs.sort_values(["Date", "Rank_Score", "Pred_Return", "Ticker"], ascending=[True, False, False, True])
    recs = recs.groupby("Date", as_index=False, group_keys=False).head(DEFAULT_MAX_BUYS_PER_DAY)
    return recs[["Date", "Ticker"]].copy(), final_csv


def append_daily_open_positions_history(outdir: Path, snapshot_date: str, current_positions_df: pd.DataFrame) -> Path:
    history_path = outdir / "merged_forward_current_positions_history.csv"
    snapshot_df = current_positions_df.copy()
    snapshot_df.insert(0, "snapshot_date", snapshot_date)

    if history_path.exists():
        existing_history = pd.read_csv(history_path)
        if "snapshot_date" in existing_history.columns:
            existing_history = existing_history[existing_history["snapshot_date"] != snapshot_date]
        snapshot_df = pd.concat([existing_history, snapshot_df], ignore_index=True)

    snapshot_df.to_csv(history_path, index=False)
    return history_path


def apply_daily_buy_limit(recs_df: pd.DataFrame, max_buys_per_day: int | None) -> tuple[pd.DataFrame, int]:
    if max_buys_per_day is None or max_buys_per_day <= 0 or recs_df.empty:
        return recs_df.copy(), 0

    ranked = recs_df.sort_values(["Date", "Rank_Score", "Pred_Return", "Ticker"], ascending=[True, False, False, True])
    limited = ranked.groupby("Date", as_index=False, group_keys=False).head(max_buys_per_day)
    filtered_count = int(len(recs_df) - len(limited))
    return limited.sort_values(["Date", "Ticker"]).reset_index(drop=True), filtered_count


def _normalize_yf_close_series(raw: pd.DataFrame) -> pd.Series:
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    close_col = None
    for candidate in ["Close", "Adj Close", "close", "adj close"]:
        if candidate in raw.columns:
            close_col = candidate
            break
    if close_col is None:
        return pd.Series(dtype=float)

    out = raw[[close_col]].copy()
    out.columns = ["close"]
    out.index = pd.to_datetime(out.index, errors="coerce").normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna()
    if out.empty:
        return pd.Series(dtype=float)
    return out["close"]


def load_price_history(price_dir: str, ticker: str, use_live_refresh: bool = True) -> pd.Series:
    path = os.path.join(price_dir, f"{ticker}_daily.csv")
    local_prices = pd.Series(dtype=float)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "Date" in df.columns and "close" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["Date", "close"]).sort_values("Date")
            if not df.empty:
                local_prices = df.drop_duplicates(subset=["Date"], keep="last").set_index("Date")["close"]

    if not use_live_refresh:
        return local_prices

    try:
        live_df = yf.download(ticker, period="9mo", progress=False, auto_adjust=False)
        live_prices = _normalize_yf_close_series(live_df)
    except Exception:
        live_prices = pd.Series(dtype=float)

    if local_prices.empty:
        return live_prices
    if live_prices.empty:
        return local_prices

    merged = pd.concat([local_prices, live_prices])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def load_live_mark_price(ticker: str) -> float:
    try:
        tk = yf.Ticker(ticker)
        fi = getattr(tk, "fast_info", None)
        if fi:
            for key in ("lastPrice", "regularMarketPrice", "previousClose"):
                value = fi.get(key, None) if hasattr(fi, "get") else None
                if value is not None and pd.notna(value) and float(value) > 0:
                    return float(value)

        intraday = tk.history(period="1d", interval="1m")
        if intraday is not None and not intraday.empty:
            cols = [c for c in ["Close", "close"] if c in intraday.columns]
            if cols:
                values = pd.to_numeric(intraday[cols[0]], errors="coerce").dropna()
                if not values.empty and float(values.iloc[-1]) > 0:
                    return float(values.iloc[-1])
    except Exception:
        pass

    return np.nan


def compute_capm_metrics(
    stock_prices: pd.Series,
    benchmark_prices: pd.Series,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    beta_lookback_days: int,
    risk_free_rate_annual: float,
    trading_days_held: int,
    capm_beta: float = None,
) -> tuple[float, float, float, float]:
    if stock_prices.empty or benchmark_prices.empty:
        return np.nan, np.nan, np.nan, np.nan

    if capm_beta is None:
        stock_window = stock_prices.loc[:entry_date].tail(beta_lookback_days + 1)
        benchmark_window = benchmark_prices.loc[:entry_date].tail(beta_lookback_days + 1)
        beta_frame = pd.concat([stock_window, benchmark_window], axis=1, join="inner").dropna()
        if len(beta_frame) < 30:
            capm_beta = np.nan
        else:
            beta_returns = beta_frame.pct_change().dropna()
            benchmark_var = beta_returns.iloc[:, 1].var()
            if pd.isna(benchmark_var) or benchmark_var <= 0:
                capm_beta = np.nan
            else:
                capm_beta = float(beta_returns.iloc[:, 0].cov(beta_returns.iloc[:, 1]) / benchmark_var)

    period_frame = pd.concat(
        [stock_prices.loc[entry_date:exit_date], benchmark_prices.loc[entry_date:exit_date]],
        axis=1,
        join="inner",
    ).dropna()
    if len(period_frame) < 2:
        return np.nan, capm_beta, np.nan, np.nan

    stock_return = float(period_frame.iloc[-1, 0] / period_frame.iloc[0, 0] - 1.0)
    benchmark_return = float(period_frame.iloc[-1, 1] / period_frame.iloc[0, 1] - 1.0)
    rf_period = (1.0 + risk_free_rate_annual) ** (max(trading_days_held, 1) / 252.0) - 1.0 if risk_free_rate_annual > -1.0 else 0.0

    if pd.isna(capm_beta):
        capm_expected_return = np.nan
        capm_alpha = np.nan
    else:
        capm_expected_return = rf_period + capm_beta * (benchmark_return - rf_period)
        capm_alpha = stock_return - capm_expected_return

    return (
        benchmark_return * 100.0,
        capm_beta,
        capm_expected_return * 100.0 if pd.notna(capm_expected_return) else np.nan,
        capm_alpha * 100.0 if pd.notna(capm_alpha) else np.nan,
    )


def evaluate_trade(
    signal_date: pd.Timestamp,
    ticker: str,
    prices: pd.Series,
    benchmark_prices: pd.Series,
    holding_days: int,
    capital_per_trade: float,
    pred_return: float,
    rank_score: float,
    capm_lookback_days: int,
    risk_free_rate_annual: float,
    capm_beta: float = None,
    entry_price_override: float | None = None,
) -> TradeResult | None:
    if prices.empty:
        return None

    eligible_idx = prices.index[prices.index >= signal_date]
    if len(eligible_idx) == 0:
        return None

    entry_date = eligible_idx[0]
    entry_pos = prices.index.get_loc(entry_date)
    if entry_price_override is not None and pd.notna(entry_price_override) and float(entry_price_override) > 0:
        entry_price = float(entry_price_override)
    else:
        entry_price = float(prices.iloc[entry_pos])

    target_exit_pos = entry_pos + holding_days
    last_pos = len(prices) - 1
    if target_exit_pos <= last_pos:
        exit_pos = target_exit_pos
        status = "CLOSED"
    else:
        exit_pos = last_pos
        status = "OPEN"

    exit_date = prices.index[exit_pos]
    exit_price = float(prices.iloc[exit_pos])

    shares = capital_per_trade / entry_price if entry_price > 0 else 0.0
    pnl_usd = shares * (exit_price - entry_price)
    return_pct = ((exit_price / entry_price) - 1.0) * 100.0 if entry_price > 0 else np.nan

    benchmark_return_pct, capm_beta_computed, capm_expected_return_pct, capm_alpha_pct = compute_capm_metrics(
        stock_prices=prices,
        benchmark_prices=benchmark_prices,
        entry_date=entry_date,
        exit_date=exit_date,
        beta_lookback_days=capm_lookback_days,
        risk_free_rate_annual=risk_free_rate_annual,
        trading_days_held=max(exit_pos - entry_pos, 1),
        capm_beta=capm_beta,
    )
    final_capm_beta = capm_beta if capm_beta is not None else capm_beta_computed
    capm_alpha_raw = (capm_alpha_pct / 100.0) if pd.notna(capm_alpha_pct) else np.nan
    capm_alpha_usd = capital_per_trade * ((capm_alpha_pct / 100.0) if pd.notna(capm_alpha_pct) else np.nan)

    return TradeResult(
        signal_date=signal_date,
        ticker=ticker,
        entry_date=entry_date,
        entry_price=entry_price,
        exit_date=exit_date,
        exit_price=exit_price,
        holding_days_target=holding_days,
        trading_days_held=exit_pos - entry_pos,
        status=status,
        capital_per_trade=capital_per_trade,
        shares=shares,
        pnl_usd=pnl_usd,
        return_pct=return_pct,
        pred_return=pred_return,
        rank_score=rank_score,
        benchmark_return_pct=benchmark_return_pct,
        capm_beta=final_capm_beta,
        capm_expected_return_pct=capm_expected_return_pct,
        capm_alpha_pct=capm_alpha_pct,
        capm_alpha_raw=capm_alpha_raw,
        capm_alpha_usd=capm_alpha_usd,
    )


def build_trade_table(
    recs_df: pd.DataFrame,
    price_dir: str,
    benchmark_prices: pd.Series,
    holding_days: int,
    capital_per_trade: float,
    capm_lookback_days: int,
    risk_free_rate_annual: float,
    use_live_refresh: bool,
) -> tuple[pd.DataFrame, dict]:
    cache = {}
    beta_cache = {}
    rows = []
    stats = {
        "input_signals": int(len(recs_df)),
        "evaluated_trades": 0,
        "skipped_missing_price_file_or_close": 0,
        "skipped_no_price_on_or_after_signal_date": 0,
    }

    for _, row in recs_df.iterrows():
        ticker = row["Ticker"]
        if ticker not in cache:
            cache[ticker] = load_price_history(price_dir, ticker, use_live_refresh=use_live_refresh)

        if cache[ticker].empty:
            stats["skipped_missing_price_file_or_close"] += 1
            continue

        if (cache[ticker].index >= row["Date"]).sum() == 0:
            stats["skipped_no_price_on_or_after_signal_date"] += 1
            continue

        if ticker not in beta_cache:
            entry_date = cache[ticker].index[(cache[ticker].index >= row["Date"])][0]
            stock_window = cache[ticker].loc[:entry_date].tail(capm_lookback_days + 1)
            bm_window = benchmark_prices.loc[:entry_date].tail(capm_lookback_days + 1)
            beta_frame = pd.concat([stock_window, bm_window], axis=1, join="inner").dropna()
            if len(beta_frame) < 30:
                beta_cache[ticker] = np.nan
            else:
                beta_returns = beta_frame.pct_change().dropna()
                benchmark_var = beta_returns.iloc[:, 1].var()
                if pd.isna(benchmark_var) or benchmark_var <= 0:
                    beta_cache[ticker] = np.nan
                else:
                    beta_cache[ticker] = float(beta_returns.iloc[:, 0].cov(beta_returns.iloc[:, 1]) / benchmark_var)

        result = evaluate_trade(
            signal_date=row["Date"],
            ticker=ticker,
            prices=cache[ticker],
            benchmark_prices=benchmark_prices,
            holding_days=holding_days,
            capital_per_trade=capital_per_trade,
            pred_return=float(row["Pred_Return"]),
            rank_score=float(row["Rank_Score"]),
            capm_lookback_days=capm_lookback_days,
            risk_free_rate_annual=risk_free_rate_annual,
            capm_beta=beta_cache[ticker],
            entry_price_override=float(row["Current_Price"]) if "Current_Price" in row and pd.notna(row["Current_Price"]) else None,
        )
        if result is not None:
            rows.append(result)
            stats["evaluated_trades"] += 1

    if not rows:
        return pd.DataFrame(), stats

    df = pd.DataFrame([r.__dict__ for r in rows])
    for col in ["signal_date", "entry_date", "exit_date"]:
        df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
    return df.sort_values(["signal_date", "ticker"]).reset_index(drop=True), stats


def summarize_trade_table(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "total_trades": 0,
            "closed_trades": 0,
            "open_trades": 0,
            "closed_win_rate_pct": np.nan,
            "closed_avg_return_pct": np.nan,
            "closed_total_pnl_usd": 0.0,
            "open_mtm_pnl_usd": 0.0,
            "total_pnl_usd": 0.0,
            "total_return_on_deployed_pct": np.nan,
            "closed_avg_capm_alpha_pct": np.nan,
            "closed_avg_capm_alpha_raw": np.nan,
            "closed_total_capm_alpha_usd": 0.0,
            "total_capm_alpha_usd": 0.0,
            "total_capm_alpha_pct": np.nan,
            "total_capm_alpha_raw": np.nan,
        }

    closed = trades_df[trades_df["status"] == "CLOSED"].copy()
    open_ = trades_df[trades_df["status"] == "OPEN"].copy()
    total_capital = trades_df["capital_per_trade"].sum()
    total_pnl = trades_df["pnl_usd"].sum()

    return {
        "total_trades": int(len(trades_df)),
        "closed_trades": int(len(closed)),
        "open_trades": int(len(open_)),
        "closed_win_rate_pct": float((closed["pnl_usd"] > 0).mean() * 100.0) if len(closed) else np.nan,
        "closed_avg_return_pct": float(closed["return_pct"].mean()) if len(closed) else np.nan,
        "closed_total_pnl_usd": float(closed["pnl_usd"].sum()) if len(closed) else 0.0,
        "open_mtm_pnl_usd": float(open_["pnl_usd"].sum()) if len(open_) else 0.0,
        "total_pnl_usd": float(total_pnl),
        "total_return_on_deployed_pct": float((total_pnl / total_capital) * 100.0) if total_capital > 0 else np.nan,
        "closed_avg_capm_alpha_pct": float(closed["capm_alpha_pct"].mean()) if len(closed) else np.nan,
        "closed_avg_capm_alpha_raw": float(closed["capm_alpha_raw"].mean()) if len(closed) else np.nan,
        "closed_total_capm_alpha_usd": float(closed["capm_alpha_usd"].sum()) if len(closed) else 0.0,
        "total_capm_alpha_usd": float(trades_df["capm_alpha_usd"].sum()) if "capm_alpha_usd" in trades_df.columns else 0.0,
        "total_capm_alpha_pct": float((trades_df["capm_alpha_usd"].sum() / total_capital) * 100.0) if total_capital > 0 and "capm_alpha_usd" in trades_df.columns else np.nan,
        "total_capm_alpha_raw": float(trades_df["capm_alpha_usd"].sum() / total_capital) if total_capital > 0 and "capm_alpha_usd" in trades_df.columns else np.nan,
    }


def build_consolidated_positions_by_ticker(
    trades_df: pd.DataFrame,
    price_dir: str = None,
    benchmark_prices: pd.Series = None,
    capm_lookback_days: int = None,
    risk_free_rate_annual: float = None,
    use_live_refresh: bool = True,
) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    tmp = trades_df.copy()
    tmp["entry_price"] = pd.to_numeric(tmp["entry_price"], errors="coerce")
    tmp["shares"] = pd.to_numeric(tmp["shares"], errors="coerce")
    tmp["capital_per_trade"] = pd.to_numeric(tmp["capital_per_trade"], errors="coerce")
    tmp["capm_alpha_usd"] = pd.to_numeric(tmp["capm_alpha_usd"], errors="coerce")

    grouped = tmp.groupby("ticker", as_index=False).agg(
        total_shares=("shares", "sum"),
        total_cost_basis_usd=("capital_per_trade", "sum"),
        signal_date_first=("signal_date", "first"),
        signal_date_last=("signal_date", "last"),
        entry_date_first=("entry_date", "first"),
        capm_alpha_usd=("capm_alpha_usd", "sum"),
    ).copy()

    grouped["weighted_avg_entry_price"] = grouped["total_cost_basis_usd"] / grouped["total_shares"]
    grouped = grouped.rename(columns={"capm_alpha_usd": "consolidated_capm_alpha_usd"})
    grouped["consolidated_capm_alpha_raw"] = np.where(
        grouped["total_cost_basis_usd"] > 0,
        grouped["consolidated_capm_alpha_usd"] / grouped["total_cost_basis_usd"],
        np.nan,
    )
    grouped = grouped.sort_values(
        ["consolidated_capm_alpha_raw", "weighted_avg_entry_price"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    return grouped


def transpose_summary_to_readable(summary_dict: dict) -> pd.DataFrame:
    """Convert flat summary dict into readable 2-column format: Metric | Value."""
    import math
    rows = []
    for key, value in summary_dict.items():
        # Pretty-print the key (convert snake_case to Title Case)
        metric_name = key.replace("_", " ").title()
        
        # Format the value
        if isinstance(value, float):
            if math.isnan(value):
                formatted_value = "N/A"
            elif abs(value) > 100 or key.endswith("_usd"):
                formatted_value = f"{value:,.2f}"
            elif value == int(value):
                formatted_value = f"{int(value)}"
            else:
                formatted_value = f"{value:.4f}"
        elif isinstance(value, (int, type(None))):
            formatted_value = str(value) if value is not None else "N/A"
        else:
            formatted_value = str(value)
        
        rows.append({"Metric": metric_name, "Value": formatted_value})
    
    return pd.DataFrame(rows)


def build_current_equity_snapshot(
    trades_df: pd.DataFrame,
    price_dir: str,
    benchmark_prices: pd.Series,
    capm_lookback_days: int,
    risk_free_rate_annual: float,
    use_live_refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if trades_df.empty:
        empty_positions = pd.DataFrame(
            columns=[
                "signal_date",
                "ticker",
                "status",
                "shares",
                "entry_price",
                "mark_price",
                "cost_basis_usd",
                "market_value_usd",
                "unrealized_pnl_usd",
                "unrealized_return_pct",
            ]
        )
        empty_ticker = pd.DataFrame(
            columns=["ticker", "total_shares", "weighted_avg_entry_price", "total_cost_basis_usd", "mark_price", "market_value_usd", "unrealized_pnl_usd", "unrealized_return_pct", "consolidated_capm_alpha_usd", "consolidated_capm_alpha_raw"]
        )
        return empty_positions, empty_ticker, {
            "open_positions": 0,
            "closed_positions": 0,
            "open_cost_basis_usd": 0.0,
            "open_market_value_usd": 0.0,
            "open_unrealized_pnl_usd": 0.0,
            "closed_realized_pnl_usd": 0.0,
            "book_cost_basis_usd": 0.0,
            "book_value_now_usd": 0.0,
            "book_profit_loss_now_usd": 0.0,
            "book_profit_loss_now_pct": np.nan,
            "portfolio_capm_alpha_usd": 0.0,
            "portfolio_capm_alpha_pct": np.nan,
            "portfolio_capm_alpha_raw": np.nan,
        }

    pos = trades_df.copy()
    latest_price_cache = {}

    for ticker in pos["ticker"].unique():
        prices = load_price_history(price_dir, ticker, use_live_refresh=use_live_refresh)
        hist_last = float(prices.iloc[-1]) if not prices.empty else np.nan
        live_mark = load_live_mark_price(ticker) if use_live_refresh else np.nan
        latest_price_cache[ticker] = float(live_mark) if pd.notna(live_mark) else hist_last

    pos["mark_price"] = pos["ticker"].map(latest_price_cache)
    pos["cost_basis_usd"] = pos["shares"] * pos["entry_price"]
    pos["market_value_usd"] = pos["shares"] * pos["mark_price"]
    pos["unrealized_pnl_usd"] = pos["market_value_usd"] - pos["cost_basis_usd"]
    pos["unrealized_return_pct"] = np.where(
        pos["cost_basis_usd"] > 0,
        (pos["unrealized_pnl_usd"] / pos["cost_basis_usd"]) * 100.0,
        np.nan,
    )

    out_positions = pos[
        [
            "signal_date",
            "ticker",
            "status",
            "shares",
            "entry_price",
            "mark_price",
            "cost_basis_usd",
            "market_value_usd",
            "unrealized_pnl_usd",
            "unrealized_return_pct",
        ]
    ].copy()

    out_ticker = (
        out_positions.groupby("ticker", as_index=False)
        .agg(
            total_shares=("shares", "sum"),
            total_cost_basis_usd=("cost_basis_usd", "sum"),
            mark_price=("mark_price", "last"),
            market_value_usd=("market_value_usd", "sum"),
            unrealized_pnl_usd=("unrealized_pnl_usd", "sum"),
        )
        .sort_values("ticker", ascending=True)
        .reset_index(drop=True)
    )

    consolidated = build_consolidated_positions_by_ticker(
        trades_df,
        price_dir,
        benchmark_prices,
        capm_lookback_days,
        risk_free_rate_annual,
        use_live_refresh,
    )
    if not consolidated.empty:
        out_ticker = out_ticker.merge(
            consolidated[["ticker", "weighted_avg_entry_price", "consolidated_capm_alpha_usd", "consolidated_capm_alpha_raw"]],
            on="ticker",
            how="left",
        )
    else:
        out_ticker["weighted_avg_entry_price"] = np.nan
        out_ticker["consolidated_capm_alpha_usd"] = 0.0
        out_ticker["consolidated_capm_alpha_raw"] = np.nan

    out_ticker["unrealized_return_pct"] = np.where(
        out_ticker["total_cost_basis_usd"] > 0,
        ((out_ticker["mark_price"] - out_ticker["weighted_avg_entry_price"]) / out_ticker["weighted_avg_entry_price"]) * 100.0,
        np.nan,
    )

    out_ticker = out_ticker.sort_values(
        ["unrealized_return_pct", "consolidated_capm_alpha_raw"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    open_pos = out_positions[out_positions["status"] == "OPEN"].copy()
    closed_pos = trades_df[trades_df["status"] == "CLOSED"].copy()

    open_cost = float(open_pos["cost_basis_usd"].sum()) if not open_pos.empty else 0.0
    open_value = float(open_pos["market_value_usd"].sum()) if not open_pos.empty else 0.0
    open_unrealized = open_value - open_cost
    closed_realized = float(closed_pos["pnl_usd"].sum()) if not closed_pos.empty else 0.0

    total_cost = float(trades_df["capital_per_trade"].sum())
    book_value = open_value + float(closed_pos["capital_per_trade"].sum()) + closed_realized
    book_pnl = book_value - total_cost

    summary = {
        "open_positions": int(len(open_pos)),
        "closed_positions": int(len(closed_pos)),
        "open_cost_basis_usd": open_cost,
        "open_market_value_usd": open_value,
        "open_unrealized_pnl_usd": float(open_unrealized),
        "closed_realized_pnl_usd": closed_realized,
        "book_cost_basis_usd": total_cost,
        "book_value_now_usd": float(book_value),
        "book_profit_loss_now_usd": float(book_pnl),
        "book_profit_loss_now_pct": float((book_pnl / total_cost) * 100.0) if total_cost > 0 else np.nan,
        "portfolio_capm_alpha_usd": float(out_ticker["consolidated_capm_alpha_usd"].sum()) if "consolidated_capm_alpha_usd" in out_ticker.columns and not out_ticker.empty else 0.0,
        "portfolio_capm_alpha_pct": float((out_ticker["consolidated_capm_alpha_usd"].sum() / total_cost) * 100.0) if total_cost > 0 and "consolidated_capm_alpha_usd" in out_ticker.columns and not out_ticker.empty else np.nan,
        "portfolio_capm_alpha_raw": float(out_ticker["consolidated_capm_alpha_usd"].sum() / total_cost) if total_cost > 0 and "consolidated_capm_alpha_usd" in out_ticker.columns and not out_ticker.empty else np.nan,
    }
    return out_positions, out_ticker, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forward pipeline then compute portfolio-level returns/alpha.")
    parser.add_argument("--run-forward", dest="run_forward", action="store_true", help="Run the 0007 forward pipeline first")
    parser.add_argument("--no-run-forward", dest="run_forward", action="store_false", help="Do not run forward pipeline (use existing CSV)")
    parser.set_defaults(run_forward=False)
    parser.add_argument("--forward-path", default=str(DEFAULT_FORWARD), help="Path to 0007_1forward.py")
    parser.add_argument("--input-dir", default=str(FORWARD_INPUT_DIR), help="Input directory where forward results are read from")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory where tracker results are written")
    parser.add_argument("--capital-per-trade", type=float, default=1000.0)
    parser.add_argument("--holding-days", type=int, default=20)
    parser.add_argument("--max-buys-per-day", type=int, default=DEFAULT_MAX_BUYS_PER_DAY)
    parser.add_argument("--no-live-refresh", action="store_true", help="Do not refresh prices via yfinance")
    args = parser.parse_args()

    forward_path = Path(args.forward_path).resolve()
    input_dir = Path(args.input_dir).resolve()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.run_forward:
        print("Running forward pipeline...")
        subprocess.run([sys.executable, "-u", str(forward_path)], cwd=str(BASE), check=True)

    tracked_recs, source_csv = load_tracker_recommendations(input_dir)
    if tracked_recs.empty:
        print(f"Could not find tracked recommendations CSV in {input_dir}. Aborting tracker run.")
        return

    if source_csv is None:
        print(f"Could not find a source CSV in {outdir}. Aborting tracker run.")
        return

    final_csv = find_final_csv(input_dir)
    if final_csv is None:
        print(f"Could not find final ensemble CSV in {input_dir}. Aborting tracker run.")
        return

    print(f"Using tracked recommendations file: {source_csv}")
    print(f"Using final predictions file for enrichment: {final_csv}")

    df_preds = pd.read_csv(final_csv)
    all_recs = map_forward_to_tracker_df(df_preds)
    recs = all_recs.merge(tracked_recs[["Date", "Ticker"]], on=["Date", "Ticker"], how="inner")
    before_liquidity = len(recs)
    recs = recs[recs["Dollar_Turnover"].fillna(0) >= MIN_TURNOVER_DOLLARS].copy()
    filtered_liquidity = int(before_liquidity - len(recs))
    recs_limited, filtered = apply_daily_buy_limit(recs, args.max_buys_per_day)

    benchmark_prices = load_price_history(str(BASE / "001_final_db_daily"), "SPY", use_live_refresh=not args.no_live_refresh)
    trades_df, stats = build_trade_table(
        recs_df=recs_limited,
        price_dir=str(BASE / "001_final_db_daily"),
        benchmark_prices=benchmark_prices,
        holding_days=args.holding_days,
        capital_per_trade=args.capital_per_trade,
        capm_lookback_days=252,
        risk_free_rate_annual=0.0,
        use_live_refresh=not args.no_live_refresh,
    )

    if trades_df.empty:
        print("No trades evaluated by tracker.")
        return

    run_timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = summarize_trade_table(trades_df)
    current_positions_df, current_ticker_df, current_summary = build_current_equity_snapshot(
        trades_df,
        str(BASE / "001_final_db_daily"),
        benchmark_prices,
        capm_lookback_days=252,
        risk_free_rate_annual=0.0,
        use_live_refresh=not args.no_live_refresh,
    )

    snapshot_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    current_positions_history_out = append_daily_open_positions_history(outdir, snapshot_date, current_positions_df.assign(snapshot_timestamp_utc=run_timestamp_utc))

    trades_out = outdir / "merged_forward_trades.csv"
    summary_out = outdir / "merged_forward_summary.csv"
    current_positions_out = outdir / "merged_forward_current_positions.csv"
    current_ticker_out = outdir / "merged_forward_current_by_ticker.csv"

    trades_df = trades_df.copy()
    trades_df["run_timestamp_utc"] = run_timestamp_utc
    trades_df["snapshot_date"] = snapshot_date

    current_positions_df = current_positions_df.copy()
    current_positions_df["snapshot_timestamp_utc"] = run_timestamp_utc
    current_positions_df["snapshot_date"] = snapshot_date

    current_ticker_df = current_ticker_df.copy()
    current_ticker_df["snapshot_timestamp_utc"] = run_timestamp_utc
    current_ticker_df["snapshot_date"] = snapshot_date

    current_summary = dict(current_summary)
    current_summary["snapshot_timestamp_utc"] = run_timestamp_utc
    current_summary["snapshot_date"] = snapshot_date

    summary = dict(summary)
    summary["run_timestamp_utc"] = run_timestamp_utc
    summary["snapshot_date"] = snapshot_date

    trades_df.to_csv(trades_out, index=False)
    pd.DataFrame([summary]).to_csv(summary_out, index=False)
    
    # Write readable summary format (Metric | Value)
    summary_readable = transpose_summary_to_readable(summary)
    summary_readable_out = outdir / "merged_forward_summary_readable.csv"
    summary_readable.to_csv(summary_readable_out, index=False)
    
    pd.DataFrame(current_summary, index=[0]).to_csv(outdir / "merged_forward_current_summary.csv", index=False)
    
    # Write readable current summary format
    current_summary_readable = transpose_summary_to_readable(current_summary)
    current_summary_readable_out = outdir / "merged_forward_current_summary_readable.csv"
    current_summary_readable.to_csv(current_summary_readable_out, index=False)
    
    current_positions_df.to_csv(current_positions_out, index=False)
    current_ticker_df.to_csv(current_ticker_out, index=False)

    print("Tracker outputs written to:")
    print(trades_out)
    print(summary_out)
    print(f"  → Readable format: {summary_readable_out}")
    print(current_positions_out)
    print(current_positions_history_out)
    print(current_ticker_out)
    print(f"  → Current summary readable: {current_summary_readable_out}")
    print("Coverage")
    print(f"- input_signals: {stats['input_signals']}")
    print(f"- filtered_by_turnover_floor_{int(MIN_TURNOVER_DOLLARS)}: {filtered_liquidity}")
    print(f"- filtered_by_daily_buy_limit: {filtered}")
    print(f"- evaluated_trades: {stats['evaluated_trades']}")
    print(f"- skipped_missing_price_file_or_close: {stats['skipped_missing_price_file_or_close']}")
    print(f"- skipped_no_price_on_or_after_signal_date: {stats['skipped_no_price_on_or_after_signal_date']}")


if __name__ == "__main__":
    main()
