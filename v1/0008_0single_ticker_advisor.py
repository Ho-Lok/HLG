#!/usr/bin/env python3
"""Single-ticker advisor for position management.

This script asks for a ticker and the date you bought it, then reuses the
existing Sentinel model stack and TradingAgents cross-check to answer the
practical question: should you hold it or sell it now?

The model horizon is 20 trading days, not 20 calendar days.

Outputs:
- Console summary
- Markdown report in 0008_forward_output/
- Optional CSV/JSON-style summary file if you extend it later
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import requests
import torch
import xgboost as xgb
import yfinance as yf
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FORWARD_MODULE_PATH = os.path.join(SCRIPT_DIR, "0007_1forward.py")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "0008_forward_output")

# Edit these first if you want to run the script without CLI arguments.
USER_TICKER = "NASA" # A ticker you want advice on.
USER_BUY_DATE = "" # Leave blank to ask "should I buy?"; set YYYY-MM-DD if you already hold it and want hold/sell advice.
USER_BUY_PRICE = None # Optional: exact price you paid (float). If None, uses market close on buy date.
USER_OUTPUT_PATH = ""
USER_RUNS = 5 # Set to 3 (or more) if you want a built-in multi-run consistency report.
USER_LLM_MODEL = "deepseek-v4-pro" # Options: deepseek-v4-pro, deepseek-v4-flash


def _apply_model_override(model: str | None) -> None:
    """Apply a local model override before loading the forward module."""
    if model and str(model).strip():
        os.environ["TRADINGAGENTS_DEEP_MODEL"] = str(model).strip()


def _load_forward_module():
    spec = importlib.util.spec_from_file_location("sentinel_forward", FORWARD_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load forward module from {FORWARD_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def _get_ticker_output_dir(ticker: str) -> str:
    """Create and return the ticker-specific output directory."""
    ticker_normalized = _normalize_ticker(ticker)
    ticker_dir = os.path.join(OUTPUT_DIR, ticker_normalized)
    os.makedirs(ticker_dir, exist_ok=True)
    return ticker_dir


def _shorten_text(text: object, max_len: int) -> str:
    value = "" if text is None else str(text)
    value = value.strip()
    if len(value) <= max_len:
        return value
    return value[: max(0, max_len - 3)].rstrip() + "..."


def _extract_verdict_from_markdown(filepath: str) -> dict:
    """Extract verdict, decision, and confidence from a generated markdown report."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Priority: "Final Recommendation" (actual conclusion), then "Should Buy Now" (intermediate field)
        final_rec_match = re.search(r"\*\*Final Recommendation\*\*:\s*\*\*([A-Z_]+)\*\*", content)
        decision_match = re.search(r"-?\s*(?:Should Buy Now|Decision|Hold or Sell):\s*([A-Z]+)", content)
        
        # Match both "Verdict:" and "Current Verdict:" forms
        rec_match = re.search(r"-?\s*(?:Current\s+)?Verdict:\s*([A-Z_]+)", content)
        # Match both "Confidence:" and "Current Confidence:" forms
        conf_match = re.search(r"-?\s*(?:Current\s+)?Confidence:\s*([\d.]+)", content)

        # Use Final Recommendation if available, otherwise Decision, otherwise Verdict
        final_verdict = final_rec_match.group(1) if final_rec_match else (decision_match.group(1) if decision_match else (rec_match.group(1) if rec_match else "N/A"))

        return {
            "Verdict": final_verdict,
            "Decision": decision_match.group(1) if decision_match else "N/A",
            "Confidence": float(conf_match.group(1)) if conf_match else 0.0,
            "Raw_Report": content,
        }
    except Exception as exc:
        print(f"Error reading {filepath}: {exc}")
        return {"Verdict": "ERROR", "Decision": "ERROR", "Confidence": 0.0, "Raw_Report": ""}


def _run_multi_run_report(ticker: str, buy_date: str | None, output_path: str | None, buy_price: float | None, runs: int, model: str | None = None) -> str:
    """Run the advisor multiple times and write one combined markdown report."""
    ticker_dir = _get_ticker_output_dir(ticker)
    _apply_model_override(model or USER_LLM_MODEL)

    us_now = datetime.now(timezone(timedelta(hours=-4)))
    today_str = us_now.strftime("%Y-%m-%d")
    generated_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    buy_date_value = str(buy_date).strip() if buy_date is not None else ""
    is_buy_mode = not buy_date_value
    is_cost_mode = buy_price is not None and not buy_date_value
    if is_cost_mode:
        suffix = "Cost_Advice"
    elif is_buy_mode:
        suffix = "Buy_Advice"
    else:
        suffix = "Advice"

    if output_path is None or not output_path.strip():
        output_path = os.path.join(ticker_dir, f"Sentinel_v1_{_normalize_ticker(ticker)}_{suffix}_MultiRun_{today_str}.md")

    print(f"📌 Ticker: {_normalize_ticker(ticker)}")
    print(f"🔄 Multi-run mode enabled: {runs} runs")

    run_results: list[dict] = []
    run_reports: list[str] = []

    for run_num in range(1, runs + 1):
        run_output_path = os.path.join(
            ticker_dir,
            f"Sentinel_v1_{_normalize_ticker(ticker)}_{suffix}_{today_str}_Run{run_num:02d}.md",
        )
        print(f"[{run_num}/{runs}] Running advisor...")
        run_single_ticker_advisor(ticker, buy_date, run_output_path, buy_price, model=model)
        run_result = _extract_verdict_from_markdown(run_output_path)
        run_result["Output_Path"] = run_output_path
        run_results.append(run_result)
        run_reports.append(run_result.get("Raw_Report", ""))
        print(
            f"  -> Verdict: {run_result['Verdict']} | Decision: {run_result['Decision']} | Confidence: {run_result['Confidence']:.2f}"
        )

    verdicts = [result["Verdict"] for result in run_results]
    decisions = [result["Decision"] for result in run_results]
    confidences = [result["Confidence"] for result in run_results if result["Confidence"] is not None]

    summary_lines = []
    summary_lines.append("# Sentinel AI - Multi-Run Advisor Summary")
    summary_lines.append(f"**Ticker:** {_normalize_ticker(ticker)}")
    is_cost_mode = buy_price is not None and not buy_date_value
    if is_cost_mode:
        mode_text = "Cost-based review"
    elif is_buy_mode:
        mode_text = "Buy review"
    else:
        mode_text = "Position review"
    summary_lines.append(f"**Mode:** {mode_text}")
    summary_lines.append(f"**Runs:** {runs}")
    if buy_date_value:
        summary_lines.append(f"**Buy Date:** {buy_date_value}")
        summary_lines.append(f"**Generated At (UTC):** {generated_at_utc}")
        generated_at_us = us_now.strftime("%Y-%m-%d %H:%M:%S EDT")
        summary_lines.append(f"**Generated At (US Eastern):** {generated_at_us}")
        summary_lines.append("")
        summary_lines.append(f"**Report Date:** {today_str}")
    summary_lines.append(f"**Generated At (UTC):** {generated_at_utc}")
    generated_at_us = us_now.strftime("%Y-%m-%d %H:%M:%S EDT")
    summary_lines.append(f"**Generated At (US Eastern):** {generated_at_us}")
    summary_lines.append(f"**Report Date:** {today_str}")
    summary_lines.append("")
    summary_lines.append("## Consensus")
    summary_lines.append(f"- Unique Verdicts: {', '.join(sorted(set(verdicts)))}")
    summary_lines.append(f"- Unique Decisions: {', '.join(sorted(set(decisions)))}")
    if confidences:
        summary_lines.append(f"- Confidence Range: {min(confidences):.2f} - {max(confidences):.2f}")
        summary_lines.append(f"- Average Confidence: {sum(confidences) / len(confidences):.2f}")
    if len(set(verdicts)) == 1:
        summary_lines.append(f"- Verdict Consensus: {verdicts[0]}")
    else:
        consensus = max(set(verdicts), key=verdicts.count)
        summary_lines.append(f"- Verdict Consensus: {consensus} ({verdicts.count(consensus)}/{runs})")
    summary_lines.append("")
    summary_lines.append("## Run Index")
    for idx, result in enumerate(run_results, 1):
        summary_lines.append(f"- Run {idx}: Verdict={result['Verdict']} | Decision={result['Decision']} | Confidence={result['Confidence']:.2f}")
    summary_lines.append("")
    summary_lines.append("## Full Run Reports")
    for idx, report_text in enumerate(run_reports, 1):
        summary_lines.append(f"### Run {idx}")
        summary_lines.append(report_text.rstrip())
        summary_lines.append("")

    combined_report = "\n".join(summary_lines).rstrip() + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined_report)

    print("\n=== Multi-Run Recommendation ===")
    print(f"Ticker: {_normalize_ticker(ticker)}")
    print(f"Runs: {runs}")
    print(f"Verdicts: {verdicts}")
    print(f"Decisions: {decisions}")
    print(f"Report saved to: {output_path}")
    return output_path


def _parse_buy_date(buy_date: str) -> pd.Timestamp:
    parsed = pd.to_datetime(buy_date, errors="coerce")
    if pd.isna(parsed):
        raise ValueError("buy-date must be a valid date in YYYY-MM-DD format")
    return pd.Timestamp(parsed).normalize()


def _load_entry_price(price_history: pd.DataFrame, buy_date: pd.Timestamp, buy_price: float | None = None) -> tuple[pd.Timestamp, float]:
    # If buy_price is explicitly provided and valid, use it directly.
    if buy_price is not None and not np.isnan(buy_price) and buy_price > 0:
        return buy_date, float(buy_price)
    
    if price_history.empty:
        raise ValueError("No price history available to determine the entry price.")

    df = price_history.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    df = df.dropna(how="all")

    if "close" not in df.columns:
        raise ValueError("Price history does not contain a close column.")

    after_buy = df[df.index >= buy_date]
    if not after_buy.empty:
        entry_row = after_buy.iloc[0]
        entry_date = pd.Timestamp(after_buy.index[0]).normalize()
        return entry_date, float(entry_row["close"])

    before_buy = df[df.index < buy_date]
    if not before_buy.empty:
        entry_row = before_buy.iloc[-1]
        entry_date = pd.Timestamp(before_buy.index[-1]).normalize()
        return entry_date, float(entry_row["close"])

    raise ValueError("Could not locate a trading day near the provided buy date.")


def _build_single_ticker_row(
    forward,
    ticker: str,
    price_history: pd.DataFrame,
    df_macro: pd.DataFrame,
    news_corpus: str,
    finbert_mod,
    finbert_tok,
    pca,
    pca_scaler,
    global_scaler,
    schema_cols,
    pt_model,
    xgb_model,
    macro_vec: np.ndarray,
    macro_sents: np.ndarray,
    today_str: str,
) -> pd.Series:
    alpaca = forward.tradeapi.REST(forward.ALPACA_API_KEY, forward.ALPACA_SECRET_KEY, forward.ALPACA_BASE_URL)
    ticker_corpus = forward.enhance_corpus_with_openbb(ticker, news_corpus)
    if isinstance(ticker_corpus, tuple):
        ticker_corpus = ticker_corpus[-1] if ticker_corpus else ""

    finbert_vectors = forward.batch_finbert_vectors(
        [(ticker, ticker_corpus)],
        finbert_mod,
        finbert_tok,
        pca,
        pca_scaler,
    )
    if ticker not in finbert_vectors:
        raise ValueError(f"No FinBERT vector could be built for {ticker}")

    finbert_data = finbert_vectors[ticker]
    finbert_vec = finbert_data["pca"]
    sents = finbert_data["sents"]

    df_price = price_history.copy()
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = [c[0] for c in df_price.columns]
    df_price.columns = [str(c).lower() for c in df_price.columns]
    df_price.index = pd.to_datetime(df_price.index, errors="coerce")
    if getattr(df_price.index, "tz", None) is not None:
        df_price.index = df_price.index.tz_convert(None)
    df_price = df_price.dropna(how="all")
    if df_price.empty or len(df_price) < 62:
        raise ValueError(f"Not enough price history for {ticker}; need at least 62 rows.")

    if "adj close" not in df_price.columns and "close" in df_price.columns:
        df_price["adj close"] = df_price["close"]
    if "open" in df_price.columns and "close" in df_price.columns and "adj close" in df_price.columns:
        df_price["adj open"] = (df_price["open"] / df_price["close"]) * df_price["adj close"]

    df_price = forward.add_technical_features(df_price)
    if not df_macro.empty:
        df_macro = df_macro.copy()
    df = df_price.join(df_macro, how="left").ffill()

    for i in range(forward.VECTOR_DIM):
        df[f"dim_{i+1}"] = finbert_vec[i]
        df[f"market_dim_{i+1}"] = macro_vec[i]
    df["sent_pos"] = sents[0]
    df["sent_neg"] = sents[1]
    df["sent_neu"] = sents[2]
    df["market_sent_pos"] = macro_sents[0]
    df["market_sent_neg"] = macro_sents[1]
    df["market_sent_neu"] = macro_sents[2]

    num_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_pct = [c for c in num_cols if any(x in c.lower() for x in ["close", "open", "high", "low", "cpi", "m2", "volume"])]
    cols_to_diff = [c for c in num_cols if any(x in c.lower() for x in ["fed", "vix", "rsi", "macd"])]
    if cols_to_pct:
        df[cols_to_pct] = df[cols_to_pct].pct_change().fillna(0.0)
    if cols_to_diff:
        df[cols_to_diff] = df[cols_to_diff].diff().fillna(0.0)

    df_aligned = df.reindex(columns=schema_cols, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    recent_60 = df_aligned.values[-60:]
    if len(recent_60) != 60:
        raise ValueError(f"Could not build a 60-day window for {ticker}")

    data_scaled = global_scaler.transform(recent_60)
    x_3d = torch.tensor(np.array([data_scaled]), dtype=torch.float32).to(forward.DEVICE)
    feats = np.concatenate([data_scaled[-1], data_scaled.mean(axis=0), data_scaled.std(axis=0)])
    x_1d = xgb.DMatrix(np.array([feats], dtype=np.float32))

    with torch.no_grad():
        pred_ret_raw = pt_model(x_3d).cpu().numpy().flatten()[0]
        pred_ret_model = pred_ret_raw * forward.pt_target_std + forward.pt_target_mean
        pred_rank = xgb_model.predict(x_1d)[0]

    recent_20d_return = forward.estimate_recent_return_pct(df_price, horizon=20)
    pred_ret = pred_ret_model
    used_fallback = False
    if recent_20d_return is not None and (not np.isfinite(pred_ret_model) or abs(pred_ret_model) < 0.25):
        pred_ret = 0.5 * pred_ret_model + 0.5 * recent_20d_return
        used_fallback = True

    current_price = float(df_price["close"].iloc[-1])
    current_volume = float(df_price["volume"].iloc[-1]) if "volume" in df_price.columns and pd.notna(df_price["volume"].iloc[-1]) else np.nan
    dollar_turnover = current_price * current_volume if pd.notna(current_volume) else np.nan

    result = {
        "Ticker": ticker,
        "Current_Price": current_price,
        "Current_Volume": current_volume,
        "Dollar_Turnover": dollar_turnover,
        "Pred_Return_Raw": pred_ret_raw,
        "Pred_Return_Model": pred_ret_model,
        "Pred_Return_%": pred_ret,
        "XGB_Rank_Score": pred_rank,
        "Recent_20D_Return_%": recent_20d_return,
        "Used_Fallback": used_fallback,
        "FinBERT_Pos": float(sents[0]),
        "FinBERT_Neg": float(sents[1]),
        "FinBERT_Neu": float(sents[2]),
        "Market_Sent_Pos": float(macro_sents[0]),
        "Market_Sent_Neg": float(macro_sents[1]),
        "Market_Sent_Neu": float(macro_sents[2]),
        "News_Corpus": ticker_corpus,
        "News_Corpus_Len": len(ticker_corpus),
    }
    result["AI_Input_Summary"] = forward._format_ai_input_summary(pd.Series(result), today_str)
    result["AI_Model_Output_Summary"] = forward._format_ai_output_summary(pd.Series(result))
    return pd.Series(result)


def _build_position_review_prompt(
    ticker: str,
    buy_date: str,
    entry_row: pd.Series,
    current_row: pd.Series,
) -> str:
    return f"""You are reviewing an existing position, not finding a new trade.

Ticker: {ticker}
Buy date: {buy_date}

You must compare the entry-date snapshot against the most recent snapshot and answer two questions:
1. Was the buying moment reasonable?
2. Should I hold it or sell it now?

Important rules:
- Use both snapshots.
- We care about whether the entry thesis still holds.
- If the current evidence weakened materially, say SELL.
- If the position still looks valid, say HOLD.
- Be honest about uncertainty.
- Do not use the Sentinel model alpha, XGB rank, or model output summary as inputs to your price target.
- Build your own target from the price action, liquidity, and any news or context you can infer from the snapshots.

Entry-date snapshot:
- Entry date used: {entry_row.get('Snapshot_Date', buy_date)}
- Entry price: ${float(entry_row.get('Current_Price', np.nan)):.2f}
- Entry recent 20 trading-day return: {float(entry_row.get('Recent_20D_Return_%', np.nan)):+.2f}%

Most recent snapshot:
- Current date used: {current_row.get('Snapshot_Date', buy_date)}
- Current price: ${float(current_row.get('Current_Price', np.nan)):.2f}
- Current recent 20 trading-day return: {float(current_row.get('Recent_20D_Return_%', np.nan)):+.2f}%

Task:
Return a structured analysis with these sections:
Technical Analysis
Fundamental Assessment
Risk Assessment
Price Target
Catalysts
Investment Thesis
Timeline

Final Recommendation: HOLD or SELL
Confidence: 0.0 to 1.0

In the reasoning, explicitly mention whether the buy date looked like a good entry and whether the position should be kept or sold now.
"""


def _build_buy_decision_prompt(ticker: str, current_row: pd.Series) -> str:
    return f"""You are evaluating a new potential purchase, not an existing position.

Ticker: {ticker}

Question:
Should I buy this stock now?

Use only the latest snapshot below and give a direct buy-or-wait assessment.

Do not use the Sentinel model alpha, XGB rank, or model output summary as inputs to the price target.
Build your own target from the market snapshot, price action, liquidity, and any visible context.

Latest trading-day snapshot:
- Snapshot date: {current_row.get('Snapshot_Date', 'n/a')}
- Current price: ${float(current_row.get('Current_Price', np.nan)):.2f}
- Daily dollar turnover: ${float(current_row.get('Dollar_Turnover', np.nan)):.0f}
- Current recent 20 trading-day return: {float(current_row.get('Recent_20D_Return_%', np.nan)):+.2f}%

Task:
Return a structured analysis with these sections:
Technical Analysis
Fundamental Assessment
Risk Assessment
Price Target
Catalysts
Investment Thesis
Timeline

Final Recommendation: BUY, HOLD, or SELL
Confidence: 0.0 to 1.0

In the reasoning, explicitly state whether this looks like a good new entry right now and whether you would buy it today.
"""


def _run_position_review_agent(forward, ticker: str, buy_date: str, entry_row: pd.Series, current_row: pd.Series) -> dict:
    provider_key_map = {"deepseek": "DEEPSEEK_API_KEY", "google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}
    api_key_env = provider_key_map.get(forward.LLM_PROVIDER.lower(), "DEEPSEEK_API_KEY")
    api_key_value = os.getenv(api_key_env) or getattr(forward, "DEEPSEEK_OFFICIAL_API_KEY", None)
    if not api_key_value:
        raise RuntimeError(
            "No API key available for LLM provider. Set the environment variable '{}' or place a DeepSeek key in forward.DEEPSEEK_OFFICIAL_API_KEY.".format(api_key_env)
        )
    client_kwargs = dict(
        provider=forward.LLM_PROVIDER,
        model=forward.LLM_DEEP_MODEL,
        base_url=forward.TRADINGAGENTS_BACKEND_URL,
        api_key=api_key_value,
        timeout=90,
        max_retries=1,
    )
    if forward.LLM_PROVIDER.lower() == "google":
        client_kwargs["thinking_level"] = forward.LLM_GOOGLE_THINKING_LEVEL
    if not (forward.TRADINGAGENTS_BACKEND_URL and str(forward.TRADINGAGENTS_BACKEND_URL).startswith("http")):
        raise RuntimeError(
            "Invalid TRADINGAGENTS_BACKEND_URL: '{}'. Ensure DEEPSEEK_API_URL or TRADINGAGENTS_BACKEND_URL is set to a valid https:// host.".format(forward.TRADINGAGENTS_BACKEND_URL)
        )
    llm_client = forward.create_llm_client(**client_kwargs)
    llm = llm_client.get_llm()
    prompt = _build_position_review_prompt(ticker, buy_date, entry_row, current_row)
    response_text = forward._run_tradingagents_with_retry(llm, prompt)
    parsed = forward._detailed_parse_agent_response(response_text)
    parsed["Raw_Response"] = response_text
    return parsed


def _run_buy_decision_agent(forward, ticker: str, current_row: pd.Series) -> dict:
    provider_key_map = {"deepseek": "DEEPSEEK_API_KEY", "google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}
    api_key_env = provider_key_map.get(forward.LLM_PROVIDER.lower(), "DEEPSEEK_API_KEY")
    api_key_value = os.getenv(api_key_env) or getattr(forward, "DEEPSEEK_OFFICIAL_API_KEY", None)
    if not api_key_value:
        raise RuntimeError(
            "No API key available for LLM provider. Set the environment variable '{}' or place a DeepSeek key in forward.DEEPSEEK_OFFICIAL_API_KEY.".format(api_key_env)
        )
    client_kwargs = dict(
        provider=forward.LLM_PROVIDER,
        model=forward.LLM_DEEP_MODEL,
        base_url=forward.TRADINGAGENTS_BACKEND_URL,
        api_key=api_key_value,
        timeout=90,
        max_retries=1,
    )
    if forward.LLM_PROVIDER.lower() == "google":
        client_kwargs["thinking_level"] = forward.LLM_GOOGLE_THINKING_LEVEL
    if not (forward.TRADINGAGENTS_BACKEND_URL and str(forward.TRADINGAGENTS_BACKEND_URL).startswith("http")):
        raise RuntimeError(
            "Invalid TRADINGAGENTS_BACKEND_URL: '{}'. Ensure DEEPSEEK_API_URL or TRADINGAGENTS_BACKEND_URL is set to a valid https:// host.".format(forward.TRADINGAGENTS_BACKEND_URL)
        )
    llm_client = forward.create_llm_client(**client_kwargs)
    llm = llm_client.get_llm()
    prompt = _build_buy_decision_prompt(ticker, current_row)
    response_text = forward._run_tradingagents_with_retry(llm, prompt)
    parsed = forward._detailed_parse_agent_response(response_text)
    parsed["Raw_Response"] = response_text
    return parsed


def _run_snapshot_agent(forward, row: pd.Series, date_str: str) -> dict:
    """Run TradingAgents on a single snapshot (entry or current) and return parsed result."""
    provider_key_map = {"deepseek": "DEEPSEEK_API_KEY", "google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}
    api_key_env = provider_key_map.get(forward.LLM_PROVIDER.lower(), "DEEPSEEK_API_KEY")
    api_key_value = os.getenv(api_key_env) or getattr(forward, "DEEPSEEK_OFFICIAL_API_KEY", None)
    if not api_key_value:
        raise RuntimeError(
            "No API key available for LLM provider. Set the environment variable '{}' or place a DeepSeek key in forward.DEEPSEEK_OFFICIAL_API_KEY.".format(api_key_env)
        )
    client_kwargs = dict(
        provider=forward.LLM_PROVIDER,
        model=forward.LLM_DEEP_MODEL,
        base_url=forward.TRADINGAGENTS_BACKEND_URL,
        api_key=api_key_value,
        timeout=90,
        max_retries=1,
    )
    if forward.LLM_PROVIDER.lower() == "google":
        client_kwargs["thinking_level"] = forward.LLM_GOOGLE_THINKING_LEVEL
    if not (forward.TRADINGAGENTS_BACKEND_URL and str(forward.TRADINGAGENTS_BACKEND_URL).startswith("http")):
        raise RuntimeError(
            "Invalid TRADINGAGENTS_BACKEND_URL: '{}'. Ensure DEEPSEEK_API_URL or TRADINGAGENTS_BACKEND_URL is set to a valid https:// host.".format(forward.TRADINGAGENTS_BACKEND_URL)
        )
    llm_client = forward.create_llm_client(**client_kwargs)
    llm = llm_client.get_llm()
    prompt = forward._format_detailed_agent_prompt(row, date_str)
    response_text = forward._run_tradingagents_with_retry(llm, prompt)
    parsed = forward._detailed_parse_agent_response(response_text)
    parsed["Raw_Response"] = response_text
    return parsed


def run_single_ticker_advisor(ticker: str, buy_date: str | None = None, output_path: str | None = None, buy_price: float | None = None, model: str | None = None) -> str:
    _apply_model_override(model or USER_LLM_MODEL)
    forward = _load_forward_module()
    ticker = _normalize_ticker(ticker)
    ticker_dir = _get_ticker_output_dir(ticker)
    buy_date_value = str(buy_date).strip() if buy_date is not None else ""
    is_buy_mode = not buy_date_value
    is_cost_mode = buy_price is not None and not buy_date_value
    buy_date_ts = _parse_buy_date(buy_date_value) if not is_buy_mode else None

    us_now = datetime.now(timezone(timedelta(hours=-4)))
    today_str = us_now.strftime("%Y-%m-%d")
    generated_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if output_path is None or not output_path.strip():
        if is_cost_mode:
            suffix = "Cost_Advice"
        elif is_buy_mode:
            suffix = "Buy_Advice"
        else:
            suffix = "Advice"
        output_path = os.path.join(ticker_dir, f"Sentinel_v1_{ticker}_{suffix}_{today_str}.md")

    print(f"📌 Ticker: {ticker}")
    if is_cost_mode:
        print(f"💰 Cost price: ${buy_price:.2f}")
        print("🧠 Loading Sentinel models for a cost-based hold/sell review...")
    elif is_buy_mode:
        print("📅 Buy date: not provided")
        print("🧠 Loading Sentinel models for a should-I-buy review...")
    else:
        print(f"📅 Buy date: {buy_date_ts.strftime('%Y-%m-%d')}")
        print(f"🧠 Loading Sentinel models for a buy-date vs now review...")

    alpaca = forward.tradeapi.REST(forward.ALPACA_API_KEY, forward.ALPACA_SECRET_KEY, forward.ALPACA_BASE_URL)
    finbert_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_dtype = torch.float16 if forward.DEVICE.type == "cuda" else torch.float32
    finbert_mod = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert",
        torch_dtype=finbert_dtype,
    ).to(forward.DEVICE).eval()

    pca = joblib.load(forward.PCA_PATH)
    pca_scaler = joblib.load(forward.PCA_SCALER_PATH)
    global_scaler = joblib.load(forward.SCALER_PATH)
    schema_cols = joblib.load(forward.SCHEMA_PATH)

    if os.path.exists(forward.PT_TARGET_STATS_PATH):
        with open(forward.PT_TARGET_STATS_PATH, "r", encoding="utf-8") as f:
            pt_target_stats = json.load(f)
        forward.pt_target_mean = float(pt_target_stats.get("mean", 0.0))
        forward.pt_target_std = float(pt_target_stats.get("std", 1.0)) or 1.0
    else:
        forward.pt_target_mean = 0.0
        forward.pt_target_std = 1.0

    pt_model = forward.PatchTST(num_features=len(schema_cols), seq_len=60).to(forward.DEVICE)
    pt_state = torch.load(forward.PT_MODEL_PATH, map_location=forward.DEVICE, weights_only=True)
    if list(pt_state.keys())[0].startswith("module."):
        pt_state = {k[7:]: v for k, v in pt_state.items()}
    pt_model.load_state_dict(pt_state)
    pt_model.eval()

    xgb_model = xgb.Booster()
    xgb_model.load_model(forward.XGB_MODEL_PATH)

    df_macro = forward.get_live_macro_data()
    if not df_macro.empty:
        df_macro = df_macro.copy()

    # Fetch long-term daily history for modeling, then overlay with freshest quote (pre/post preferred)
    ticker_prices = yf.download(ticker, period="8mo", threads=False, progress=False)
    latest_quote_source = "daily_close"
    try:
        latest_price, latest_volume, latest_quote_source = forward.get_latest_price_quote(ticker)
        if pd.notna(latest_price) and float(latest_price) > 0 and not ticker_prices.empty:
            if isinstance(ticker_prices.columns, pd.MultiIndex):
                ticker_prices.columns = [c[0] if isinstance(c, tuple) and len(c) > 0 else c for c in ticker_prices.columns]
            tp_cols = {c.lower(): c for c in ticker_prices.columns}
            tp_close_col = tp_cols.get("close")
            tp_vol_col = tp_cols.get("volume")
            if tp_close_col:
                ticker_prices.at[ticker_prices.index[-1], tp_close_col] = float(latest_price)
            if tp_vol_col and pd.notna(latest_volume):
                ticker_prices.at[ticker_prices.index[-1], tp_vol_col] = float(latest_volume)
    except Exception:
        pass
    if ticker_prices.empty:
        raise ValueError(f"No price history returned for {ticker}")

    _, current_news_corpus = forward.fetch_ticker_news(
        ticker,
        alpaca,
        use_alpaca=forward.ALPACA_NEWS_ENABLED,
        alpaca_semaphore=None,
    )
    current_news_corpus = forward.enhance_corpus_with_openbb(ticker, current_news_corpus)
    if isinstance(current_news_corpus, tuple):
        current_news_corpus = current_news_corpus[-1] if current_news_corpus else ""

    # Build macro sentiment vectors using SPY so both snapshots see the same market context
    _, market_news = forward.fetch_ticker_news(
        "SPY",
        alpaca,
        use_alpaca=forward.ALPACA_NEWS_ENABLED,
        alpaca_semaphore=None,
    )
    market_news = forward.enhance_corpus_with_openbb("SPY", market_news)
    if isinstance(market_news, tuple):
        market_news = market_news[-1] if market_news else ""
    market_vectors = forward.batch_finbert_vectors(
        [("SPY", market_news)],
        finbert_mod,
        finbert_tok,
        pca,
        pca_scaler,
    )
    macro_data = market_vectors.get("SPY", {"pca": np.zeros(forward.VECTOR_DIM), "sents": np.zeros(3)})
    macro_vec = macro_data["pca"]
    macro_sents = macro_data["sents"]

    current_row = _build_single_ticker_row(
        forward=forward,
        ticker=ticker,
        price_history=ticker_prices,
        df_macro=df_macro,
        news_corpus=current_news_corpus,
        finbert_mod=finbert_mod,
        finbert_tok=finbert_tok,
        pca=pca,
        pca_scaler=pca_scaler,
        global_scaler=global_scaler,
        schema_cols=schema_cols,
        pt_model=pt_model,
        xgb_model=xgb_model,
        macro_vec=macro_vec,
        macro_sents=macro_sents,
        today_str=today_str,
    )

    current_row["Snapshot_Date"] = today_str
    current_row["Current_Price_Source"] = latest_quote_source

    # Persist current news and vector files for this ticker even if it's not in the SP500 whitelist.
    try:
        # Upsert news row into NEWS_DIR
        forward.upsert_ticker_news_row(ticker, today_str, current_news_corpus)
    except Exception:
        pass

    # Build and write a single-row vector CSV into VECTOR_DIR
    try:
        finbert_vecs = forward.batch_finbert_vectors([(ticker, current_news_corpus)], finbert_mod, finbert_tok, pca, pca_scaler)
        vec_row = None
        if ticker in finbert_vecs:
            fv = finbert_vecs[ticker]
            pca_vec = fv.get("pca", np.zeros(forward.VECTOR_DIM))
            sents = fv.get("sents", np.zeros(3))
            vec_cols = {f"dim_{i+1}": float(pca_vec[i]) for i in range(forward.VECTOR_DIM)}
            vec_cols.update({f"market_dim_{i+1}": float(macro_vec[i]) for i in range(forward.VECTOR_DIM)})
            vec_cols.update({"sent_pos": float(sents[0]), "sent_neg": float(sents[1]), "sent_neu": float(sents[2])})
            vec_row = pd.DataFrame([vec_cols])
        if vec_row is not None:
            os.makedirs(forward.VECTOR_DIR, exist_ok=True)
            vec_path = os.path.join(forward.VECTOR_DIR, f"{ticker}_vec.csv")
            vec_row.to_csv(vec_path, index=False)
    except Exception:
        pass

    # Persist latest ticker stock row into PRICE_DIR (001_final_db_daily)
    try:
        df_price_save = ticker_prices.copy()
        if isinstance(df_price_save.columns, pd.MultiIndex):
            df_price_save.columns = [c[0] for c in df_price_save.columns]
        df_price_save.columns = [str(c).lower() for c in df_price_save.columns]
        df_price_save.index = pd.to_datetime(df_price_save.index, utc=True, errors="coerce").tz_convert(None)
        df_price_save = forward.add_technical_features(df_price_save)

        base_cols = [
            "close", "high", "low", "open", "volume",
            "RSI_14", "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
        ]
        for col in base_cols:
            if col not in df_price_save.columns:
                df_price_save[col] = np.nan

        macro_tail = df_macro.iloc[-1:] if isinstance(df_macro, pd.DataFrame) and not df_macro.empty else pd.DataFrame(index=df_price_save.iloc[-1:].index)
        latest_price_row = df_price_save[base_cols].iloc[-1:].join(macro_tail, how="left")
        latest_price_row.index.name = "Date"

        os.makedirs(forward.PRICE_DIR, exist_ok=True)
        price_path = os.path.join(forward.PRICE_DIR, f"{ticker}_daily.csv")
        if os.path.exists(price_path):
            df_old_price = pd.read_csv(price_path, index_col="Date")
            if not df_old_price.empty:
                old_dates = pd.to_datetime(df_old_price.index, errors="coerce").strftime("%Y-%m-%d")
                target_day = latest_price_row.index[0].strftime("%Y-%m-%d")
                df_old_price = df_old_price[old_dates != target_day]
            pd.concat([df_old_price, latest_price_row]).to_csv(price_path)
        else:
            latest_price_row.to_csv(price_path)
    except Exception:
        pass

    if is_buy_mode and not is_cost_mode:
        buy_review = _run_buy_decision_agent(forward, ticker, current_row)
        current_verdict = str(buy_review.get("Recommendation", "hold")).lower()
        current_confidence = float(buy_review.get("Confidence", 0.5)) if pd.notna(buy_review.get("Confidence")) else 0.5
        hold_or_sell = "BUY" if current_verdict == "buy" else "HOLD" if current_verdict in {"hold", "wait"} else "SELL"
        exit_analysis = None
    elif is_cost_mode:
        # Cost-mode: hold/sell based on cost price vs current price
        current_price = float(current_row["Current_Price"])
        price_change = ((current_price - buy_price) / buy_price) * 100
        
        # Run agent analysis on current data for hold/sell recommendation
        review = _run_buy_decision_agent(forward, ticker, current_row)
        current_verdict = str(review.get("Recommendation", "hold")).lower()
        current_confidence = float(review.get("Confidence", 0.5)) if pd.notna(review.get("Confidence")) else 0.5
        
        # Map verdict to hold/sell decision
        if current_verdict == "buy" or price_change > 10:
            hold_or_sell = "HOLD"  # Still bullish or up significantly
        elif current_verdict == "sell" or price_change < -10:
            hold_or_sell = "SELL"  # Bearish or down significantly
        else:
            hold_or_sell = "HOLD"  # Default to hold for wait/neutral
        
        exit_analysis = None
        buydate_agent = review  # Reuse the agent review for consistency
    else:
        entry_date, entry_price = _load_entry_price(ticker_prices, buy_date_ts, buy_price)
        entry_price_history = ticker_prices.loc[:entry_date].copy()
        if entry_price_history.empty:
            raise ValueError("Could not build entry-date history slice.")

        # Historical news is not reconstructable from this pipeline, so keep a short placeholder
        # while still using the actual buy-date price window for the entry snapshot.
        entry_news_corpus = f"Historical news unavailable for {ticker} on {entry_date.strftime('%Y-%m-%d')}; compare price/momentum and current context."

        entry_row = _build_single_ticker_row(
            forward=forward,
            ticker=ticker,
            price_history=entry_price_history,
            df_macro=df_macro,
            news_corpus=entry_news_corpus,
            finbert_mod=finbert_mod,
            finbert_tok=finbert_tok,
            pca=pca,
            pca_scaler=pca_scaler,
            global_scaler=global_scaler,
            schema_cols=schema_cols,
            pt_model=pt_model,
            xgb_model=xgb_model,
            macro_vec=macro_vec,
            macro_sents=macro_sents,
            today_str=entry_date.strftime("%Y-%m-%d"),
        )
        entry_row["Snapshot_Date"] = entry_date.strftime("%Y-%m-%d")

        # Run an independent agent analysis for the buy-date snapshot so we can store
        # the agent's original assessment separately in the advice markdown.
        buydate_agent = _run_snapshot_agent(forward, entry_row, entry_row["Snapshot_Date"])

        review = _run_position_review_agent(forward, ticker, buy_date_ts.strftime("%Y-%m-%d"), entry_row, current_row)
        current_verdict = str(review.get("Recommendation", "hold")).lower()
        current_confidence = float(review.get("Confidence", 0.5)) if pd.notna(review.get("Confidence")) else 0.5

        exit_analysis = forward._analyze_position_exit(
            entry_ticker=ticker,
            entry_date=entry_date.strftime("%Y-%m-%d"),
            entry_recommendation="buy",
            entry_price=entry_price,
            entry_confidence=0.5,
            current_row=current_row,
            current_agent_verdict=current_verdict,
            current_confidence=current_confidence,
            current_price=float(current_row["Current_Price"]),
        )

        hold_or_sell = "SELL" if current_verdict == "sell" or "EXIT" in exit_analysis["Exit_Signal"] or "REDUCE" in exit_analysis["Exit_Signal"] or "TAKE PROFITS" in exit_analysis["Exit_Signal"] or "REVERSE" in exit_analysis["Exit_Signal"] else "HOLD"

    report = []
    report.append("# Sentinel AI - Single Ticker Advisor")
    report.append(f"**Ticker:** {ticker}")
    report.append(f"**Generated At (UTC):** {generated_at_utc}")
    if is_cost_mode:
        report.append("**Mode:** Cost-based hold/sell review")
        report.append(f"**Cost Price:** ${buy_price:.2f}")
        report.append(f"**Current Price:** ${float(current_row['Current_Price']):.2f}")
        price_change_pct = ((float(current_row['Current_Price']) - buy_price) / buy_price) * 100
        report.append(f"**Gain/Loss:** {price_change_pct:+.2f}%")
    elif is_buy_mode:
        report.append("**Mode:** Buy review")
    else:
        report.append(f"**Buy Date:** {buy_date_ts.strftime('%Y-%m-%d')}")
        report.append(f"**Entry Date Used:** {entry_date.strftime('%Y-%m-%d')}")
    report.append("This report compares the latest Sentinel model snapshot with a TradingAgents opinion. If a buy date is provided, it also compares the entry snapshot against the latest snapshot.")
    report.append("")
    report.append("## Sentinel Model View - Latest Trading Date (20 trading-day horizon)")
    report.append(f"- Current Price: ${float(current_row['Current_Price']):.2f}")
    if str(current_row.get("Current_Price_Source", "")).strip():
        report.append(f"- Current Price Source: {current_row.get('Current_Price_Source')}")
    if pd.notna(current_row.get("Dollar_Turnover", np.nan)):
        report.append(f"- Daily Dollar Turnover: ${float(current_row['Dollar_Turnover']):,.0f}")
    report.append("")
    report.append("### Model Predictions")
    if pd.notna(current_row.get("Pred_Return_Raw", np.nan)):
        report.append(f"- PatchTST Raw Output (normalized): {float(current_row['Pred_Return_Raw']):.6f}")
    if pd.notna(current_row.get("Pred_Return_Model", np.nan)):
        report.append(f"- PatchTST De-normalized Output: {float(current_row['Pred_Return_Model']):+.2f}%")
    if pd.notna(current_row.get("Recent_20D_Return_%", np.nan)):
        report.append(f"- Recent 20D Return (momentum context): {float(current_row['Recent_20D_Return_%']):+.2f}%")
    report.append(f"- Predicted 20D Alpha (final blend): {float(current_row['Pred_Return_%']):+.2f}%")
    report.append(f"- XGBoost Ranking Score: {float(current_row['XGB_Rank_Score']):.3f}")
    report.append("")
    if is_cost_mode:
        report.append("## TradingAgents Analysis - Hold/Sell Decision")
        report.append(f"- Recommendation: {str(buydate_agent.get('Recommendation', 'n/a')).upper()}")
        report.append(f"- Confidence: {float(buydate_agent.get('Confidence', 0.0)):.2f}")
        report.append(f"- Technical Analysis: {_shorten_text(buydate_agent.get('Technical_Analysis', ''), 8000)}")
        report.append(f"- Fundamental Assessment: {_shorten_text(buydate_agent.get('Fundamental_Assessment', ''), 8000)}")
        report.append(f"- Risk Assessment: {_shorten_text(buydate_agent.get('Risk_Assessment', ''), 8000)}")
        report.append(f"- Investment Thesis: {_shorten_text(buydate_agent.get('Investment_Thesis', ''), 8000)}")
        report.append("")
        report.append("## Hold/Sell Decision")
        if pd.notna(buydate_agent.get('Price_Target', np.nan)) and buydate_agent.get('Price_Target'):
            report.append(f"- Price Target: {_shorten_text(buydate_agent.get('Price_Target'), 8000)}")

        price_change_pct = ((float(current_row['Current_Price']) - buy_price) / buy_price) * 100
        report.append(f"- Decision: {hold_or_sell}")
        report.append(f"- Cost Price: ${buy_price:.2f}")
        report.append(f"- Current Price: ${float(current_row['Current_Price']):.2f}")
        report.append(f"- Gain/Loss: {price_change_pct:+.2f}%")
        report.append(f"- Verdict: {current_verdict.upper()}")
        report.append(f"- Confidence: {current_confidence:.2f}")
        report.append(f"- Agent Raw Response: {_shorten_text(buydate_agent.get('Raw_Response', ''), 8000)}")
    elif is_buy_mode:
        report.append("")
        report.append("## TradingAgents Analysis - Buy Decision")
        report.append(f"- Recommendation: {str(buy_review.get('Recommendation', 'n/a')).upper()}")
        report.append(f"- Confidence: {float(buy_review.get('Confidence', 0.0)):.2f}")
        report.append(f"- Technical Analysis: {_shorten_text(buy_review.get('Technical_Analysis', ''), 8000)}")
        report.append(f"- Fundamental Assessment: {_shorten_text(buy_review.get('Fundamental_Assessment', ''), 8000)}")
        report.append(f"- Risk Assessment: {_shorten_text(buy_review.get('Risk_Assessment', ''), 8000)}")
        report.append(f"- Investment Thesis: {_shorten_text(buy_review.get('Investment_Thesis', ''), 8000)}")
        report.append("")
        report.append("## Buy Decision Summary")
        if pd.notna(buy_review.get('Price_Target', np.nan)) and buy_review.get('Price_Target'):
            report.append(f"- Price Target: {_shorten_text(buy_review.get('Price_Target'), 8000)}")

        # Present as clear BUY / DON'T BUY for buy review
        quick_buy = "BUY" if current_verdict == "buy" else "DON'T BUY"
        report.append(f"- Should Buy Now: {quick_buy}")
        report.append(f"- Current Price: ${float(current_row['Current_Price']):.2f}")
        report.append(f"- Current Verdict: {current_verdict.upper()}")
        report.append(f"- Current Confidence: {current_confidence:.2f}")
        report.append(f"- Agent Raw Response: {_shorten_text(buy_review.get('Raw_Response', ''), 8000)}")
    else:
        report.append("## Sentinel Model View - Buy-Date Snapshot (nearest trading day)")
        report.append(f"- Entry Date: {entry_date.strftime('%Y-%m-%d')}")
        report.append(f"- Entry Price: ${entry_price:.2f}")
        report.append("")
        report.append("### Model Predictions at Entry")
        if pd.notna(entry_row.get("Pred_Return_Raw", np.nan)):
            report.append(f"- PatchTST Raw Output (normalized): {float(entry_row['Pred_Return_Raw']):.6f}")
        if pd.notna(entry_row.get("Pred_Return_Model", np.nan)):
            report.append(f"- PatchTST De-normalized Output: {float(entry_row['Pred_Return_Model']):+.2f}%")
        if pd.notna(entry_row.get("Recent_20D_Return_%", np.nan)):
            report.append(f"- Recent 20D Return (momentum context): {float(entry_row['Recent_20D_Return_%']):+.2f}%")
        report.append(f"- Predicted 20D Alpha at Entry (final blend): {float(entry_row['Pred_Return_%']):+.2f}%")
        report.append(f"- XGBoost Ranking Score at Entry: {float(entry_row['XGB_Rank_Score']):.3f}")
        report.append("")
        report.append("## TradingAgents Analysis - Entry Snapshot")
        report.append(f"- Recommendation: {str(buydate_agent.get('Recommendation', 'n/a')).upper()}")
        report.append(f"- Confidence: {float(buydate_agent.get('Confidence', 0.0)):.2f}")
        report.append(f"- Technical Analysis: {_shorten_text(buydate_agent.get('Technical_Analysis', ''), 8000)}")
        report.append(f"- Fundamental Assessment: {_shorten_text(buydate_agent.get('Fundamental_Assessment', ''), 8000)}")
        report.append(f"- Risk Assessment: {_shorten_text(buydate_agent.get('Risk_Assessment', ''), 8000)}")
        report.append(f"- Investment Thesis: {_shorten_text(buydate_agent.get('Investment_Thesis', ''), 8000)}")
        report.append("")
        report.append("## TradingAgents Comparison - Entry vs Latest Trading Date")
        if pd.notna(buydate_agent.get('Price_Target', np.nan)) and buydate_agent.get('Price_Target'):
            report.append(f"- Price Target: {_shorten_text(buydate_agent.get('Price_Target'), 8000)}")

        report.append(f"- Recommendation: {str(review.get('Recommendation', 'n/a')).upper()}")
        report.append(f"- Confidence: {float(review.get('Confidence', 0.0)):.2f}")
        report.append(f"- Thesis Change / Hold-Sell Assessment: {_shorten_text(review.get('Investment_Thesis', '') or review.get('Technical_Analysis', ''), 8000)}")
        report.append("")
        report.append("## Position Decision")
        report.append(f"- Exit Signal: {exit_analysis['Exit_Signal']}")
        report.append(f"- Hold or Sell: {hold_or_sell}")
        report.append(f"- Days Held: {exit_analysis['Days_Held']}")
        report.append(f"- Price Change: {exit_analysis['Price_Change_%']:+.2f}%")
        report.append(f"- Reason: {exit_analysis['Reason']}")
        report.append("")
        report.append("## Key Details")
        report.append(f"- Entry Price (derived from the nearest trading day): ${entry_price:.2f}")
        report.append(f"- Current Price: ${float(current_row['Current_Price']):.2f}")
        report.append(f"- Current Verdict: {current_verdict.upper()}")
        report.append(f"- Current Confidence: {current_confidence:.2f}")
        report.append(f"- Agent Raw Response: {_shorten_text(review.get('Raw_Response', ''), 8000)}")

    report_text = "\n".join(report) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n=== Single Ticker Recommendation ===")
    print(f"Ticker: {ticker}")
    print(f"Decision: {hold_or_sell}")
    if is_cost_mode:
        price_change_pct = ((float(current_row['Current_Price']) - buy_price) / buy_price) * 100
        print("Signal: COST-BASED REVIEW")
        print(f"Cost Price: ${buy_price:.2f} → Current: ${float(current_row['Current_Price']):.2f} ({price_change_pct:+.2f}%)")
        print(f"Reason: {buydate_agent.get('Investment_Thesis', buydate_agent.get('Technical_Analysis', 'No raw agent reason available.'))}")
    elif is_buy_mode:
        print("Signal: BUY REVIEW")
        print(f"Reason: {buy_review.get('Investment_Thesis', buy_review.get('Technical_Analysis', 'No raw agent reason available.'))}")
    else:
        print(f"Signal: {exit_analysis['Exit_Signal']}")
        print(f"Reason: {exit_analysis['Reason']}")
    print(f"Report saved to: {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-ticker Sentinel advisor for buy, hold, or sell decisions.")
    parser.add_argument("--ticker", default=USER_TICKER, help="Ticker symbol to analyze, for example AAPL")
    parser.add_argument("--buy-date", default=USER_BUY_DATE, help="Date you bought the ticker, in YYYY-MM-DD format. Leave blank to ask should I buy?")
    parser.add_argument("--buy-price", type=float, default=USER_BUY_PRICE, help="Optional: exact price you paid. If provided without buy-date, get hold/sell advice based on cost vs current price.")
    parser.add_argument("--output", default="", help="Optional markdown output path")
    parser.add_argument("--runs", type=int, default=USER_RUNS, help="Run the advisor N times and create one combined report")
    parser.add_argument("--model", default=USER_LLM_MODEL, help="LLM model to use for TradingAgents (e.g., gemini-2.5-pro-thinking, gemini-2.5-pro, gemini-2.5-flash)")
    args = parser.parse_args()

    output_path = args.output if args.output else USER_OUTPUT_PATH
    if args.runs > 1:
        _run_multi_run_report(args.ticker, args.buy_date, output_path, args.buy_price, args.runs, model=args.model)
    else:
        run_single_ticker_advisor(args.ticker, args.buy_date, output_path, args.buy_price, model=args.model)


if __name__ == "__main__":
    main()