import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

RAW_TICKERS = """MMM,AOS,ABT,ABBV,ACN,ADBE,AMD,AES,AFL,A,APD,ABNB,AKAM,ALB,ARE,ALGN,ALLE,LNT,ALL,GOOGL,GOOG,MO,AMZN,AMCR,AEE,AEP,AXP,AIG,AMT,AWK,AMP,AME,AMGN,APH,ADI,AON,APA,APO,AAPL,AMAT,APP,APTV,ACGL,ADM,ARES,ANET,AJG,AIZ,T,ATO,ADSK,ADP,AZO,AVB,AVY,AXON,BKR,BALL,BAC,BAX,BDX,BRK.B,BBY,TECH,BIIB,BLK,BX,XYZ,BK,BA,BKNG,BSX,BMY,AVGO,BR,BRO,BF.B,BLDR,BG,BXP,CHRW,CDNS,CPT,CPB,COF,CAH,CCL,CARR,CVNA,CAT,CBOE,CBRE,CDW,COR,CNC,CNP,CF,CRL,SCHW,CHTR,CVX,CMG,CB,CHD,CIEN,CI,CINF,CTAS,CSCO,C,CFG,CLX,CME,CMS,KO,CTSH,COIN,CL,CMCSA,FIX,CAG,COP,ED,STZ,CEG,COO,CPRT,GLW,CPAY,CTVA,CSGP,COST,CTRA,CRH,CRWD,CCI,CSX,CMI,CVS,DHR,DRI,DDOG,DVA,DECK,DE,DELL,DAL,DVN,DXCM,FANG,DLR,DG,DLTR,D,DPZ,DASH,DOV,DOW,DHI,DTE,DUK,DD,ETN,EBAY,ECL,EIX,EW,EA,ELV,EME,EMR,ETR,EOG,EPAM,EQT,EFX,EQIX,EQR,ERIE,ESS,EL,EG,EVRG,ES,EXC,EXE,EXPE,EXPD,EXR,XOM,FFIV,FDS,FICO,FAST,FRT,FDX,FIS,FITB,FSLR,FE,FISV,F,FTNT,FTV,FOXA,FOX,BEN,FCX,GRMN,IT,GE,GEHC,GEV,GEN,GNRC,GD,GIS,GM,GPC,GILD,GPN,GL,GDDY,GS,HAL,HIG,HAS,HCA,DOC,HSIC,HSY,HPE,HLT,HOLX,HD,HON,HRL,HST,HWM,HPQ,HUBB,HUM,HBAN,HII,IBM,IEX,IDXX,ITW,INCY,IR,PODD,INTC,IBKR,ICE,IFF,IP,INTU,ISRG,IVZ,INVH,IQV,IRM,JBHT,JBL,JKHY,J,JNJ,JCI,JPM,KVUE,KDP,KEY,KEYS,KMB,KIM,KMI,KKR,KLAC,KHC,KR,LHX,LH,LRCX,LW,LVS,LDOS,LEN,LII,LLY,LIN,LYV,LMT,L,LOW,LULU,LYB,MTB,MPC,MAR,MRSH,MLM,MAS,MA,MTCH,MKC,MCD,MCK,MDT,MRK,META,MET,MTD,MGM,MCHP,MU,MSFT,MAA,MRNA,MOH,TAP,MDLZ,MPWR,MNST,MCO,MS,MOS,MSI,MSCI,NDAQ,NTAP,NFLX,NEM,NWSA,NWS,NEE,NKE,NI,NDSN,NSC,NTRS,NOC,NCLH,NRG,NUE,NVDA,NVR,NXPI,ORLY,OXY,ODFL,OMC,ON,OKE,ORCL,OTIS,PCAR,PKG,PLTR,PANW,PSKY,PH,PAYX,PAYC,PYPL,PNR,PEP,PFE,PCG,PM,PSX,PNW,PNC,POOL,PPG,PPL,PFG,PG,PGR,PLD,PRU,PEG,PTC,PSA,PHM,PWR,QCOM,DGX,Q,RL,RJF,RTX,O,REG,REGN,RF,RSG,RMD,RVTY,HOOD,ROK,ROL,ROP,ROST,RCL,SPGI,CRM,SNDK,SBAC,SLB,STX,SRE,NOW,SHW,SPG,SWKS,SJM,SW,SNA,SOLV,SO,LUV,SWK,SBUX,STT,STLD,STE,SYK,SMCI,SYF,SNPS,SYY,TMUS,TROW,TTWO,TPR,TRGP,TGT,TEL,TDY,TER,TSLA,TXN,TPL,TXT,TMO,TJX,TKO,TTD,TSCO,TT,TDG,TRV,TRMB,TFC,TYL,TSN,USB,UBER,UDR,ULTA,UNP,UAL,UPS,URI,UNH,UHS,VLO,VTR,VLTO,VRSN,VRSK,VZ,VRTX,VTRS,VICI,V,VST,VMC,WRB,GWW,WAB,WMT,DIS,WBD,WM,WAT,WEC,WFC,WELL,WST,WDC,WY,WSM,WMB,WTW,WDAY,WYNN,XEL,XYL,YUM,ZBRA,ZBH,ZTS"""
SP500_TICKERS = [t.strip() for t in RAW_TICKERS.replace('\n', '').split(',') if t.strip()]


def resolve_existing_path(candidates: list[str], kind_label: str) -> str | None:
    for path in candidates:
        if os.path.exists(path):
            return path
    print(f"❌ Error: Could not find {kind_label}. Tried: {candidates}")
    return None


def _load_price_history(price_file: str) -> pd.DataFrame:
    df_price = pd.read_csv(price_file, parse_dates=["Date"])
    df_price["Date"] = pd.to_datetime(df_price["Date"], errors="coerce")
    df_price = df_price.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df_price


def _compute_trade_daily_returns(df_trade: pd.DataFrame, execution_model: str) -> pd.Series | None:
    close_col = next((c for c in ["Close", "close", "adj close"] if c in df_trade.columns), "Close")
    open_col = next((c for c in ["Open", "open", "adj open"] if c in df_trade.columns), "Open")

    if execution_model == "AVG":
        high_col = next((c for c in ["High", "high"] if c in df_trade.columns), None)
        low_col = next((c for c in ["Low", "low"] if c in df_trade.columns), None)
        close_for_avg = next((c for c in ["Close", "close", "adj close"] if c in df_trade.columns), None)
        if not all([open_col, high_col, low_col, close_for_avg]):
            return None

        needed_cols = [open_col, high_col, low_col, close_for_avg, close_col]
        for col in needed_cols:
            df_trade[col] = pd.to_numeric(df_trade[col], errors="coerce")
        df_trade = df_trade.dropna(subset=needed_cols)
        if len(df_trade) == 0:
            return None

        avg_price_series = (df_trade[open_col] + df_trade[high_col] + df_trade[low_col] + df_trade[close_for_avg]) / 4
        daily_returns = avg_price_series.pct_change().fillna(0)

        start_price = avg_price_series.iloc[0]
        daily_returns.iloc[0] = (df_trade[close_col].iloc[0] - start_price) / start_price if start_price > 0 else 0.0

        if len(daily_returns) > 1:
            final_avg_exit = avg_price_series.iloc[-1]
            prev_close = df_trade[close_col].iloc[-2]
            daily_returns.iloc[-1] = (final_avg_exit - prev_close) / prev_close if prev_close > 0 else 0.0
        return daily_returns

    if execution_model == "PAIN":
        high_col = next((c for c in ["High", "high"] if c in df_trade.columns), None)
        low_col = next((c for c in ["Low", "low"] if c in df_trade.columns), None)
        if not all([high_col, low_col, close_col]):
            return None

        needed_cols = [close_col, high_col, low_col]
        for col in needed_cols:
            df_trade[col] = pd.to_numeric(df_trade[col], errors="coerce")
        df_trade = df_trade.dropna(subset=needed_cols)
        if len(df_trade) == 0:
            return None

        # Worst-case long execution: enter at first-day high and exit at last-day low.
        daily_returns = df_trade[close_col].pct_change().fillna(0)
        start_price = df_trade[high_col].iloc[0]
        end_price = df_trade[low_col].iloc[-1]

        if len(df_trade) == 1:
            only_day_return = (end_price - start_price) / start_price if start_price > 0 else 0.0
            return pd.Series([only_day_return], index=df_trade.index)

        daily_returns.iloc[0] = (df_trade[close_col].iloc[0] - start_price) / start_price if start_price > 0 else 0.0
        prev_close = df_trade[close_col].iloc[-2]
        daily_returns.iloc[-1] = (end_price - prev_close) / prev_close if prev_close > 0 else 0.0
        return daily_returns

    for col in [close_col, open_col]:
        df_trade[col] = pd.to_numeric(df_trade[col], errors="coerce")
    df_trade = df_trade.dropna(subset=[close_col, open_col])
    if len(df_trade) == 0:
        return None

    daily_returns = df_trade[close_col].pct_change().fillna(0)
    start_price = df_trade[open_col].iloc[0]
    daily_returns.iloc[0] = (df_trade[close_col].iloc[0] - start_price) / start_price if start_price > 0 else 0.0
    return daily_returns


def _calculate_risk_metrics(returns: pd.Series, confidence_level: float = 0.95) -> dict[str, float]:
    clean_returns = pd.to_numeric(returns, errors="coerce").dropna()
    if clean_returns.empty:
        return {
            "variance": 0.0,
            "historical_var": 0.0,
            "historical_cvar": 0.0,
            "parametric_var": 0.0,
            "parametric_cvar": 0.0,
        }

    alpha = 1.0 - confidence_level
    variance = float(clean_returns.var())

    historical_cutoff = clean_returns.quantile(alpha)
    historical_var = float(-historical_cutoff)
    historical_tail = clean_returns[clean_returns <= historical_cutoff]
    historical_cvar = float(-historical_tail.mean()) if not historical_tail.empty else historical_var

    mean_return = float(clean_returns.mean())
    std_return = float(clean_returns.std())
    if std_return > 0:
        z_score = float(norm.ppf(alpha))
        parametric_return_var = mean_return + std_return * z_score
        parametric_var = float(-parametric_return_var)
        parametric_return_cvar = mean_return - std_return * norm.pdf(z_score) / alpha
        parametric_cvar = float(-parametric_return_cvar)
    else:
        parametric_var = 0.0
        parametric_cvar = 0.0

    return {
        "variance": variance,
        "historical_var": historical_var,
        "historical_cvar": historical_cvar,
        "parametric_var": parametric_var,
        "parametric_cvar": parametric_cvar,
    }


def run_backtest(config: dict[str, Any]) -> None:
    variant_name = config["variant_name"]
    results_files = config["results_files"]
    out_prefix = config["out_prefix"]
    execution_model = config.get("execution_model", "OPEN")
    signal_delay_bdays = int(config.get("signal_delay_bdays", 0))

    max_trades_per_day = int(config.get("max_trades_per_day", 1))
    leverage = float(config.get("leverage", 1.0))
    cash_recycling = bool(config.get("cash_recycling", False))
    min_pred_return = float(config.get("min_pred_return", 0.5))
    min_rank_percentile = config.get("min_rank_percentile", None)
    if min_rank_percentile is not None:
        min_rank_percentile = float(min_rank_percentile)
    slippage_and_fees = float(config.get("slippage_and_fees", 0.01))
    hard_stop_loss = float(config.get("hard_stop_loss", -0.1))
    take_profit_cfg = config.get("take_profit", None)
    take_profit = float(take_profit_cfg) if take_profit_cfg is not None else None
    contradiction_exit_enabled = bool(config.get("contradiction_exit_enabled", False))
    contradiction_pred_threshold = float(config.get("contradiction_pred_threshold", 0.0))
    risk_free_rate = float(config.get("risk_free_rate", 0.051))
    starting_capital = float(config.get("starting_capital", 100000))
    active_holding_period = int(config.get("holding_period", 20))

    macro_file = config.get("macro_file", "001_macro_data.csv")
    price_dir_candidates = config.get("price_dir_candidates", ["001_final_db_daily", "final_db_daily"])
    simulation_start_date = config.get("simulation_start_date", "2024-01-01")
    simulation_end_date = config.get("simulation_end_date", "2025-12-31")

    output_root = config.get("output_root", "006_outputs")
    output_dir = os.path.join(output_root, out_prefix)
    os.makedirs(output_dir, exist_ok=True)

    trades_export = os.path.join(output_dir, f"trades_{out_prefix}.csv")
    daily_export = os.path.join(output_dir, f"daily_{out_prefix}.csv")
    monthly_export = os.path.join(output_dir, f"monthly_{out_prefix}.csv")
    plot_export = os.path.join(output_dir, f"portfolio_{out_prefix}.png")

    results_path = resolve_existing_path(results_files, "results file")
    if results_path is None:
        return
    if results_path != results_files[0]:
        print(f"⚠️ Using fallback results file: {results_path}")

    macro_path = resolve_existing_path([macro_file], "macro file")
    if macro_path is None:
        return

    price_dir = resolve_existing_path(price_dir_candidates, "price directory")
    if price_dir is None:
        return

    df = pd.read_csv(results_path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Pred_Return_%", "XGB_Rank_Score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Pred_Return_%", "XGB_Rank_Score"])

    filter_sp500 = bool(config.get("filter_sp500", True))

    if filter_sp500 and SP500_TICKERS:
        pre_filter_count = len(df)
        df = df[df["Ticker"].isin(SP500_TICKERS)]
        print(f"Applied S&P 500 filter: {pre_filter_count} -> {len(df)} rows")
    elif not filter_sp500:
        print("S&P 500 filter disabled; processing all tickers in results.")
    else:
        print("Warning: SP500_TICKERS unavailable; skipping S&P 500 filter.")

    if signal_delay_bdays > 0:
        df["Original_Signal_Date"] = df["Date"]
        df["Date"] = df["Date"] + pd.offsets.BusinessDay(signal_delay_bdays)

    df_macro = pd.read_csv(macro_path)
    df_macro["Date"] = pd.to_datetime(df_macro["Date"], errors="coerce")
    df_macro["SPY_Close"] = pd.to_numeric(df_macro["SPY_Close"], errors="coerce")
    df_macro = df_macro.dropna(subset=["Date", "SPY_Close"])
    df_macro.set_index("Date", inplace=True)

    if simulation_start_date:
        start_ts = pd.to_datetime(simulation_start_date)
        df = df[df["Date"] >= start_ts]
        df_macro = df_macro[df_macro.index >= start_ts]
    if simulation_end_date:
        end_ts = pd.to_datetime(simulation_end_date)
        df = df[df["Date"] <= end_ts]
        df_macro = df_macro[df_macro.index <= end_ts]

    if df.empty or df_macro.empty:
        print("No rows left after cleaning/date filtering.")
        return

    pred_lookup = df.groupby(["Date", "Ticker"])["Pred_Return_%"].mean()
    active_capital_per_trade = 1.0 / (max_trades_per_day * active_holding_period)
    rolling_equity = starting_capital

    min_xgb_score = config.get("min_xgb_score", None)
    if min_xgb_score is not None:
        df_longs = df[(df["Pred_Return_%"] >= min_pred_return) & (df["XGB_Rank_Score"] >= float(min_xgb_score))].copy()
    else:
        df_longs = df[df["Pred_Return_%"] >= min_pred_return].copy()
    if min_rank_percentile is not None:
        df_longs["Daily_Rank_Pct"] = df_longs.groupby("Date")["XGB_Rank_Score"].rank(pct=True, ascending=True)
        pre_rank_count = len(df_longs)
        df_longs = df_longs[df_longs["Daily_Rank_Pct"] >= min_rank_percentile].copy()
        print(f"Applied rank percentile filter >= {min_rank_percentile:.2f}: {pre_rank_count} -> {len(df_longs)} rows")
    if df_longs.empty:
        print("No trades found.")
        return

    df_trades = df_longs.sort_values(by=["Date", "XGB_Rank_Score", "Pred_Return_%"], ascending=[True, False, False])
    skip_days_with_tickers = config.get("skip_days_with_tickers", [])
    exclude_tickers = config.get("exclude_tickers", [])
    if skip_days_with_tickers or exclude_tickers:
        skip_set = set(t.upper() for t in skip_days_with_tickers)
        exclude_set = set(t.upper() for t in exclude_tickers)
        top_trades_list = []
        for date, group in df_trades.groupby("Date"):
            if skip_set and group["Ticker"].str.upper().isin(skip_set).any():
                continue
            available = group[~group["Ticker"].str.upper().isin(exclude_set)]
            top_trades_list.append(available.head(max_trades_per_day))
        top_trades = pd.concat(top_trades_list, ignore_index=False).copy() if top_trades_list else pd.DataFrame(columns=df_trades.columns)
    else:
        top_trades = df_trades.groupby("Date").head(max_trades_per_day).copy()

    portfolio = pd.DataFrame(index=df_macro.index)
    portfolio["Daily_PnL"] = 0.0
    portfolio["Active_Capital"] = 0.0
    portfolio["Cash_Flow"] = 0.0
    portfolio["Cash_Used_For_Buys"] = 0.0
    portfolio["Stock_Holding"] = 0.0
    executed_trades: list[dict[str, Any]] = []
    contradiction_exits = 0

    daily_pnl_deltas = {}
    unique_signal_dates = sorted(top_trades["Date"].unique())
    signal_date_to_trading_day = {date: idx for idx, date in enumerate(unique_signal_dates)}
    
    # =========== PRE-PASS: Calculate exit dates, returns, and opening capital ===========
    # Store: (exit_date, signal_date_index) -> list of (trading_capital, return)
    # We need signal_date_index to know which trade size was used when the trade opened
    exit_date_to_trade_info: dict[pd.Timestamp, list[tuple[float, float]]] = {}
    
    if cash_recycling:
        print(f"📊 Pre-calculating exit dates for cash recycling...")
        
        for signal_date, group in top_trades.groupby("Date"):
            trading_day_index = signal_date_to_trading_day[signal_date]
            is_ramp_up_period = trading_day_index < active_holding_period
            
            for _, row in group.iterrows():
                ticker = row["Ticker"]
                price_file = os.path.join(price_dir, f"{ticker}_daily.csv")
                if not os.path.exists(price_file):
                    continue
                
                df_price = _load_price_history(price_file)
                # Include the signal date so same-day executions are possible
                df_price = df_price[df_price.index >= signal_date]
                if len(df_price) == 0:
                    continue
                
                df_trade = df_price.iloc[:active_holding_period].copy()
                if len(df_trade) < active_holding_period:
                    continue
                
                daily_returns = _compute_trade_daily_returns(df_trade, execution_model)
                if daily_returns is None or len(daily_returns) < active_holding_period:
                    continue
                
                daily_returns.iloc[0] -= slippage_and_fees
                prev_cum_return = 0.0
                exit_idx = len(daily_returns) - 1
                
                for i in range(len(daily_returns)):
                    trade_day = df_trade.index[i]
                    curr_ret = daily_returns.iloc[i]
                    curr_cum_return = (1 + prev_cum_return) * (1 + curr_ret) - 1
                    
                    if curr_cum_return <= hard_stop_loss:
                        daily_returns.iloc[i] = hard_stop_loss - prev_cum_return
                        daily_returns.iloc[i + 1:] = 0.0
                        exit_idx = i
                        break

                    if take_profit is not None and curr_cum_return >= take_profit:
                        daily_returns.iloc[i] = take_profit - prev_cum_return
                        daily_returns.iloc[i + 1:] = 0.0
                        exit_idx = i
                        break
                    
                    if contradiction_exit_enabled and curr_cum_return < 0:
                        pred_today = pred_lookup.get((trade_day, ticker), np.nan)
                        if pd.notna(pred_today) and pred_today < contradiction_pred_threshold:
                            daily_returns.iloc[i + 1:] = 0.0
                            exit_idx = i
                            break
                    
                    prev_cum_return = curr_cum_return
                
                trade_total_return = (1 + daily_returns).prod() - 1
                exit_date = df_trade.index[exit_idx]
                
                # Store: we'll determine the capital in the main loop based on rolling_equity at opening
                # For now, store None as placeholder for capital (we'll fill it in the main loop)
                if exit_date not in exit_date_to_trade_info:
                    exit_date_to_trade_info[exit_date] = []
                # Store (signal_date_index, trade_return) - we'll get capital from main loop
                exit_date_to_trade_info[exit_date].append((trading_day_index, trade_total_return))
        
        print(f"✅ Pre-calculated exits for {sum(len(v) for v in exit_date_to_trade_info.values())} trades")
    
    # =========== MAIN BACKTEST LOOP ===========
    # Track trade capitals as they're calculated so we can use exact amounts for rolling_equity updates
    trade_capital_by_signal_idx_and_count: dict[int, list[float]] = {}
    
    for signal_date, group in tqdm(top_trades.groupby("Date"), desc=f"Simulating {variant_name}"):
        rolling_equity_for_day = rolling_equity
        day_trades = []
        
        trading_day_index = signal_date_to_trading_day[signal_date]
        is_ramp_up_period = trading_day_index < active_holding_period
        
        # Update rolling_equity for trades that EXIT on this signal_date
        # We track which trades (by opening signal_date_index) are exiting now
        if cash_recycling and signal_date in exit_date_to_trade_info:
            for opening_signal_index, trade_total_return in exit_date_to_trade_info[signal_date]:
                # Get the actual trade capital that was used when this trade opened
                if opening_signal_index in trade_capital_by_signal_idx_and_count and len(trade_capital_by_signal_idx_and_count[opening_signal_index]) > 0:
                    exit_trade_capital = trade_capital_by_signal_idx_and_count[opening_signal_index].pop(0)
                    rolling_equity += exit_trade_capital * trade_total_return
            
            rolling_equity_for_day = rolling_equity  # Use updated rolling_equity for new trades
        
        for _, row in group.iterrows():
            ticker = row["Ticker"]
            price_file = os.path.join(price_dir, f"{ticker}_daily.csv")
            if not os.path.exists(price_file):
                continue

            df_price = _load_price_history(price_file)
            # Include the signal date so same-day executions are possible
            df_price = df_price[df_price.index >= signal_date]
            if len(df_price) == 0:
                continue

            df_trade = df_price.iloc[:active_holding_period].copy()
            if len(df_trade) < active_holding_period:
                continue

            close_col = next((c for c in ["Close", "close", "adj close"] if c in df_trade.columns), "Close")
            open_col = next((c for c in ["Open", "open", "adj open"] if c in df_trade.columns), "Open")
            high_col = next((c for c in ["High", "high"] if c in df_trade.columns), None)
            low_col = next((c for c in ["Low", "low"] if c in df_trade.columns), None)
            close_for_avg_col = next((c for c in ["Close", "close", "adj close"] if c in df_trade.columns), None)

            entry_date = df_trade.index[0]
            if execution_model == "AVG":
                if not all([open_col, high_col, low_col, close_for_avg_col]):
                    continue
                entry_price = pd.to_numeric(
                    pd.Series(
                        [
                            (
                                df_trade[open_col].iloc[0]
                                + df_trade[high_col].iloc[0]
                                + df_trade[low_col].iloc[0]
                                + df_trade[close_for_avg_col].iloc[0]
                            )
                            / 4
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]
            elif execution_model == "PAIN":
                if high_col is None:
                    continue
                entry_price = pd.to_numeric(pd.Series([df_trade[high_col].iloc[0]]), errors="coerce").iloc[0]
            else:
                entry_price = pd.to_numeric(pd.Series([df_trade[open_col].iloc[0]]), errors="coerce").iloc[0]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            if cash_recycling and not is_ramp_up_period:
                trade_capital = rolling_equity_for_day / (max_trades_per_day * active_holding_period)
                trade_capital_fraction = trade_capital / starting_capital
            else:
                trade_capital = starting_capital * active_capital_per_trade
                trade_capital_fraction = active_capital_per_trade
            
            # Track this trade's capital for later rolling_equity updates
            if trading_day_index not in trade_capital_by_signal_idx_and_count:
                trade_capital_by_signal_idx_and_count[trading_day_index] = []
            trade_capital_by_signal_idx_and_count[trading_day_index].append(trade_capital)

            daily_returns = _compute_trade_daily_returns(df_trade, execution_model)
            if daily_returns is None or len(daily_returns) < active_holding_period:
                continue

            daily_returns.iloc[0] -= slippage_and_fees
            prev_cum_return = 0.0
            contradiction_exit_triggered = False
            hard_stop_triggered = False
            take_profit_triggered = False
            exit_idx = len(daily_returns) - 1
            exit_reason = "Time Exit"
            for i in range(len(daily_returns)):
                trade_day = df_trade.index[i]
                curr_ret = daily_returns.iloc[i]
                curr_cum_return = (1 + prev_cum_return) * (1 + curr_ret) - 1

                if curr_cum_return <= hard_stop_loss:
                    daily_returns.iloc[i] = hard_stop_loss - prev_cum_return
                    daily_returns.iloc[i + 1:] = 0.0
                    exit_idx = i
                    exit_reason = "Hard Stop-Loss"
                    hard_stop_triggered = True
                    break

                if take_profit is not None and curr_cum_return >= take_profit:
                    daily_returns.iloc[i] = take_profit - prev_cum_return
                    daily_returns.iloc[i + 1:] = 0.0
                    exit_idx = i
                    exit_reason = "Take Profit"
                    take_profit_triggered = True
                    break

                if contradiction_exit_enabled and curr_cum_return < 0:
                    pred_today = pred_lookup.get((trade_day, ticker), np.nan)
                    if pd.notna(pred_today) and pred_today < contradiction_pred_threshold:
                        daily_returns.iloc[i + 1:] = 0.0
                        contradiction_exit_triggered = True
                        contradiction_exits += 1
                        exit_idx = i
                        exit_reason = "Contradiction Exit"
                        break

                prev_cum_return = curr_cum_return

            trade_total_return = (1 + daily_returns).prod() - 1
            exit_date = df_trade.index[exit_idx]
            if hard_stop_triggered:
                exit_price = entry_price * (1 + hard_stop_loss)
            elif take_profit_triggered and take_profit is not None:
                exit_price = entry_price * (1 + take_profit)
            elif execution_model == "AVG":
                exit_price = pd.to_numeric(
                    pd.Series(
                        [
                            (
                                df_trade[open_col].iloc[exit_idx]
                                + df_trade[high_col].iloc[exit_idx]
                                + df_trade[low_col].iloc[exit_idx]
                                + df_trade[close_for_avg_col].iloc[exit_idx]
                            )
                            / 4
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]
            elif execution_model == "PAIN" and (exit_idx == len(daily_returns) - 1) and not contradiction_exit_triggered:
                exit_price = pd.to_numeric(pd.Series([df_trade[low_col].iloc[exit_idx]]), errors="coerce").iloc[0]
            else:
                exit_price = pd.to_numeric(pd.Series([df_trade[close_col].iloc[exit_idx]]), errors="coerce").iloc[0]
            if pd.isna(exit_price) or exit_price <= 0:
                continue

            trade_meta = {
                "Date": signal_date,
                "Ticker": ticker,
                "Side": 1,
                "Pred_Return_%": row["Pred_Return_%"],
                "XGB_Rank_Score": row["XGB_Rank_Score"],
                "Strategy_Return": trade_total_return,
                "Cash_Used": trade_capital,
                "Cash_Gained": trade_capital * trade_total_return,
                "Cash_Final": trade_capital * (1 + trade_total_return),
                "Entry_Date": entry_date,
                "Entry_Price": entry_price,
                "Exit_Date": exit_date,
                "Exit_Price": exit_price,
                "Exit_Reason": exit_reason,
            }
            if contradiction_exit_enabled:
                trade_meta["Contradiction_Exit"] = contradiction_exit_triggered
            if signal_delay_bdays > 0 and "Original_Signal_Date" in row:
                trade_meta["Original_Signal_Date"] = row["Original_Signal_Date"]
            executed_trades.append(trade_meta)

            valid_dates = df_trade.index.intersection(portfolio.index)
            portfolio.loc[valid_dates, "Daily_PnL"] += daily_returns.loc[valid_dates] * trade_capital_fraction
            portfolio.loc[valid_dates, "Active_Capital"] += trade_capital_fraction

            if len(valid_dates) > 0:
                active_stock_dates = valid_dates[:-1]
                if len(active_stock_dates) > 0:
                    trade_stock_values = trade_capital * (1 + daily_returns.loc[active_stock_dates]).cumprod()
                    portfolio.loc[active_stock_dates, "Stock_Holding"] += trade_stock_values

                if entry_date in portfolio.index:
                    portfolio.loc[entry_date, "Cash_Flow"] -= trade_capital
                    portfolio.loc[entry_date, "Cash_Used_For_Buys"] += trade_capital
                if exit_date in portfolio.index:
                    portfolio.loc[exit_date, "Cash_Flow"] += trade_capital * (1 + trade_total_return)
            
            day_trades.append((trade_capital, trade_total_return))
        
    if len(executed_trades) == 0:
        print("No trades were successfully simulated after price alignment checks.")
        return

    portfolio["AI_Daily_Return"] = (portfolio["Daily_PnL"] * leverage) + (1.0 - portfolio["Active_Capital"].clip(upper=1.0)) * (risk_free_rate / 252.0)
    portfolio["SPY_Daily_Return"] = df_macro["SPY_Close"].pct_change().fillna(0)
    portfolio["Invested_Ratio"] = portfolio["Active_Capital"].clip(upper=1.0)
    portfolio["AI_Equity"] = starting_capital * (1 + portfolio["AI_Daily_Return"]).cumprod()
    portfolio["SPY_Equity"] = starting_capital * (1 + portfolio["SPY_Daily_Return"]).cumprod()
    portfolio["AI_Equity"] = portfolio["AI_Equity"].fillna(starting_capital)
    portfolio["SPY_Equity"] = portfolio["SPY_Equity"].fillna(starting_capital)
    portfolio["Cash_Holding"] = starting_capital + portfolio["Cash_Flow"].cumsum()
    portfolio["Cash_Not_Used_For_Buys"] = portfolio["Cash_Holding"]

    ai_roi = ((portfolio["AI_Equity"].iloc[-1] - starting_capital) / starting_capital) * 100
    spy_roi = ((portfolio["SPY_Equity"].iloc[-1] - starting_capital) / starting_capital) * 100

    portfolio["AI_Peak"] = portfolio["AI_Equity"].cummax()
    portfolio["AI_Drawdown"] = (portfolio["AI_Equity"] - portfolio["AI_Peak"]) / portfolio["AI_Peak"]
    max_drawdown = portfolio["AI_Drawdown"].min() * 100
    avg_invested = portfolio["Invested_Ratio"].mean() * 100

    ai_std = portfolio["AI_Daily_Return"].std()
    spy_std = portfolio["SPY_Daily_Return"].std()
    ai_sharpe = (portfolio["AI_Daily_Return"].mean() / ai_std) * np.sqrt(252) if ai_std != 0 else 0
    spy_sharpe = (portfolio["SPY_Daily_Return"].mean() / spy_std) * np.sqrt(252) if spy_std != 0 else 0
    risk_confidence_level = float(config.get("risk_confidence_level", 0.95))
    ai_risk_metrics = _calculate_risk_metrics(portfolio["AI_Daily_Return"], risk_confidence_level)
    daily_rf_rate = risk_free_rate / 252.0
    excess_ai = portfolio["AI_Daily_Return"] - daily_rf_rate
    excess_mkt = portfolio["SPY_Daily_Return"] - daily_rf_rate
    mkt_var = excess_mkt.var()
    capm_beta = excess_ai.cov(excess_mkt) / mkt_var if mkt_var != 0 else 0.0
    capm_alpha_daily = excess_ai.mean() - (capm_beta * excess_mkt.mean())
    capm_alpha_annual_pct = capm_alpha_daily * 252 * 100
    active_daily_return = portfolio["AI_Daily_Return"] - portfolio["SPY_Daily_Return"]
    tracking_error = active_daily_return.std()
    info_ratio = (active_daily_return.mean() / tracking_error) * np.sqrt(252) if tracking_error != 0 else 0
    correlation = portfolio["AI_Daily_Return"].corr(portfolio["SPY_Daily_Return"])

    down_mask = portfolio["SPY_Daily_Return"] < 0
    up_mask = portfolio["SPY_Daily_Return"] > 0
    ai_down = portfolio.loc[down_mask, "AI_Daily_Return"]
    ai_up = portfolio.loc[up_mask, "AI_Daily_Return"]
    spy_down = portfolio.loc[down_mask, "SPY_Daily_Return"]
    spy_up = portfolio.loc[up_mask, "SPY_Daily_Return"]
    ai_avg_down_day = ai_down.mean() * 100 if len(ai_down) > 0 else 0.0
    ai_avg_up_day = ai_up.mean() * 100 if len(ai_up) > 0 else 0.0
    spy_avg_down_day = spy_down.mean() * 100 if len(spy_down) > 0 else 0.0
    spy_avg_up_day = spy_up.mean() * 100 if len(spy_up) > 0 else 0.0
    excess_avg_down_day = ai_avg_down_day - spy_avg_down_day
    excess_avg_up_day = ai_avg_up_day - spy_avg_up_day

    export_trades = pd.DataFrame(executed_trades)
    trade_cols = [
        "Date",
        "Ticker",
        "Side",
        "Pred_Return_%",
        "XGB_Rank_Score",
        "Entry_Date",
        "Entry_Price",
        "Exit_Date",
        "Exit_Price",
        "Exit_Reason",
        "Strategy_Return",
        "Cash_Used",
        "Cash_Gained",
        "Cash_Final",
    ]
    if signal_delay_bdays > 0 and "Original_Signal_Date" in export_trades.columns:
        trade_cols = [
            "Date",
            "Original_Signal_Date",
            "Ticker",
            "Side",
            "Pred_Return_%",
            "XGB_Rank_Score",
            "Entry_Date",
            "Entry_Price",
            "Exit_Date",
            "Exit_Price",
            "Exit_Reason",
            "Strategy_Return",
            "Cash_Used",
            "Cash_Gained",
            "Cash_Final",
        ]
    export_trades = export_trades[trade_cols].copy()
    export_trades["Date"] = pd.to_datetime(export_trades["Date"]).dt.strftime("%Y-%m-%d")
    export_trades["Entry_Date"] = pd.to_datetime(export_trades["Entry_Date"]).dt.strftime("%Y-%m-%d")
    export_trades["Exit_Date"] = pd.to_datetime(export_trades["Exit_Date"]).dt.strftime("%Y-%m-%d")
    if "Original_Signal_Date" in export_trades.columns:
        export_trades["Original_Signal_Date"] = pd.to_datetime(export_trades["Original_Signal_Date"]).dt.strftime("%Y-%m-%d")
    export_trades["Side"] = export_trades["Side"].map({1: "LONG"})
    export_trades["Entry_Price"] = pd.to_numeric(export_trades["Entry_Price"], errors="coerce").round(4)
    export_trades["Exit_Price"] = pd.to_numeric(export_trades["Exit_Price"], errors="coerce").round(4)
    export_trades["Strategy_Return_%"] = round(export_trades["Strategy_Return"] * 100, 2)
    export_trades["Cash_Used"] = pd.to_numeric(export_trades["Cash_Used"], errors="coerce").round(2)
    export_trades["Cash_Gained"] = pd.to_numeric(export_trades["Cash_Gained"], errors="coerce").round(2)
    export_trades["Cash_Final"] = pd.to_numeric(export_trades["Cash_Final"], errors="coerce").round(2)
    export_trades.to_csv(trades_export, index=False)

    portfolio["Equity_Diff"] = portfolio["AI_Equity"] - portfolio["SPY_Equity"]
    invested_ratio = portfolio["Invested_Ratio"].clip(lower=0, upper=1)
    portfolio["Gross_Exposure"] = invested_ratio * leverage * 100
    portfolio["Net_Exposure"] = ((invested_ratio * leverage) - (1.0 - invested_ratio)) * 100

    daily_export_df = portfolio[["AI_Daily_Return", "SPY_Daily_Return", "AI_Equity", "SPY_Equity", "Equity_Diff", "Stock_Holding", "Cash_Holding", "Cash_Used_For_Buys", "Cash_Not_Used_For_Buys", "AI_Drawdown", "Invested_Ratio", "Gross_Exposure", "Net_Exposure"]].copy()
    daily_export_df.index = daily_export_df.index.strftime("%Y-%m-%d")
    daily_export_df.to_csv(daily_export)

    monthly_data = portfolio[["AI_Daily_Return", "SPY_Daily_Return", "AI_Equity", "SPY_Equity", "Equity_Diff", "AI_Drawdown", "Invested_Ratio", "Gross_Exposure", "Net_Exposure"]].copy()
    monthly_data.index = pd.to_datetime(monthly_data.index)
    monthly_export_df = monthly_data.resample("ME").agg(
        {
            "AI_Daily_Return": lambda x: (np.prod(1 + x) - 1) * 100,
            "SPY_Daily_Return": lambda x: (np.prod(1 + x) - 1) * 100,
            "AI_Equity": "last",
            "SPY_Equity": "last",
            "Equity_Diff": "last",
            "AI_Drawdown": "min",
            "Invested_Ratio": "mean",
            "Gross_Exposure": "mean",
            "Net_Exposure": "mean",
        }
    )

    monthly_export_df["AI_Ret_%"] = monthly_export_df["AI_Daily_Return"].round(2)
    monthly_export_df["SPY_Ret_%"] = monthly_export_df["SPY_Daily_Return"].round(2)
    monthly_export_df["Alpha_%"] = (monthly_export_df["AI_Ret_%"] - monthly_export_df["SPY_Ret_%"]).round(2)
    monthly_export_df["Max_DD_%"] = (monthly_export_df["AI_Drawdown"] * 100).round(2)
    monthly_export_df["Exposure_%"] = (monthly_export_df["Invested_Ratio"] * 100).round(2)
    monthly_export_df["Gross_Exposure_%"] = monthly_export_df["Gross_Exposure"].round(2)
    monthly_export_df["Net_Exposure_%"] = monthly_export_df["Net_Exposure"].round(2)
    monthly_export_df["AI_Equity"] = monthly_export_df["AI_Equity"].round(2)
    monthly_export_df["SPY_Equity"] = monthly_export_df["SPY_Equity"].round(2)
    monthly_export_df["Equity_Diff"] = monthly_export_df["Equity_Diff"].round(2)
    monthly_export_df.index = monthly_export_df.index.strftime("%Y-%m")
    monthly_export_df.index.name = "Date"
    column_order = ["AI_Ret_%", "SPY_Ret_%", "Alpha_%", "Max_DD_%", "Exposure_%", "Gross_Exposure_%", "Net_Exposure_%", "AI_Equity", "SPY_Equity", "Equity_Diff"]
    monthly_export_df = monthly_export_df[column_order]
    monthly_export_df.to_csv(monthly_export)

    winning_trades = export_trades[export_trades["Strategy_Return"] > 0]
    losing_trades = export_trades[export_trades["Strategy_Return"] <= 0]
    win_rate = (len(winning_trades) / len(export_trades)) * 100 if len(export_trades) > 0 else 0
    avg_win = winning_trades["Strategy_Return_%"].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades["Strategy_Return_%"].mean() if not losing_trades.empty else 0
    gross_profit = winning_trades["Strategy_Return"].sum()
    gross_loss = abs(losing_trades["Strategy_Return"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

    print("\n" + "=" * 50)
    print(f"📈 SENTINEL AI - 20-DAY S&P 500 COMPOUNDER ({variant_name}) 📈")
    print("=" * 50)
    if signal_delay_bdays > 0:
        print(f"Signal Delay:           {signal_delay_bdays} trading day")
    if contradiction_exit_enabled:
        print(f"Contradiction Exit:     ON (Pred < {contradiction_pred_threshold:.2f} while losing)")
    if min_rank_percentile is not None:
        print(f"Min Rank Percentile:    {min_rank_percentile:.2f}")
    print(f"Total Trades Executed:  {len(export_trades):,}")
    print(f"Avg Base Capital Dep:   {avg_invested:.1f}%")
    print(f"Hard Stop-Loss:         ACTIVE (Cut at {hard_stop_loss*100}%)")
    if take_profit is not None:
        print(f"Take-Profit:            ACTIVE (Cap at {take_profit*100}%)")
    print(f"Slippage Assumption:    {slippage_and_fees*10000} BPS")
    print("-" * 50)
    print(f"Final AI Total Asset:   ${portfolio['AI_Equity'].iloc[-1]:,.2f}")
    print(f"Final SPY Benchmark:    ${portfolio['SPY_Equity'].iloc[-1]:,.2f}")
    print("-" * 50)
    print(f"AI Total ROI:           {ai_roi:+.2f}%")
    print(f"SPY Total ROI:          {spy_roi:+.2f}%")
    print(f"ROI Difference:         {ai_roi - spy_roi:+.2f}%")
    print("-" * 50)
    print(f"AI Sharpe Ratio:        {ai_sharpe:.2f}")
    print(f"SPY Sharpe Ratio:       {spy_sharpe:.2f}")
    print(f"Market Correlation:     {correlation:.2f}")
    print(f"CAPM Alpha (Ann.):      {capm_alpha_annual_pct:+.2f}%")
    print(f"Information Ratio:      {info_ratio:.2f}")
    print(f"Max Drawdown:           {max_drawdown:.2f}%")
    print("-" * 50)
    print(f"Risk Confidence Level:  {risk_confidence_level:.0%}")
    print(f"Variance (daily):       {ai_risk_metrics['variance']:.6f}")
    print(f"Historical VaR:         {ai_risk_metrics['historical_var'] * 100:.2f}%")
    print(f"Historical CVaR:        {ai_risk_metrics['historical_cvar'] * 100:.2f}%")
    print(f"Parametric VaR:         {ai_risk_metrics['parametric_var'] * 100:.2f}%")
    print(f"Parametric CVaR:        {ai_risk_metrics['parametric_cvar'] * 100:.2f}%")
    print("-" * 50)
    print("🌦️ MARKET REGIME ATTRIBUTION 🌦️")
    print(f"AI Avg on SPY Down Day: {ai_avg_down_day:+.3f}% ({len(ai_down)} days)")
    print(f"AI Avg on SPY Up Day:   {ai_avg_up_day:+.3f}% ({len(ai_up)} days)")
    print(f"Excess vs SPY on Down:  {excess_avg_down_day:+.3f}%/day")
    print(f"Excess vs SPY on Up:    {excess_avg_up_day:+.3f}%/day")
    print("-" * 50)
    print("🏆 TRADE ANALYTICS 🏆")
    print(f"Win Rate:               {win_rate:.1f}% ({len(winning_trades)} Wins / {len(losing_trades)} Losses)")
    print(f"Average Win:            +{avg_win:.2f}%")
    print(f"Average Loss:           {avg_loss:.2f}%")
    print(f"Profit Factor:          {profit_factor:.2f}")
    if contradiction_exit_enabled:
        print(f"Contradiction Exits:    {contradiction_exits}")
    print("=" * 50)
    print(f"Saved outputs to:       {output_dir}")

    summary_lines = [
        "BASE PERFORMANCE",
        f"Variant:  {variant_name}",
        f"AI ROI:   {ai_roi:+.1f}%",
        f"SPY ROI:  {spy_roi:+.1f}%",
        f"ROI Diff: {ai_roi - spy_roi:+.1f}%",
        f"Final $:  ${portfolio['AI_Equity'].iloc[-1]:,.0f}",
        f"Max DD:   {max_drawdown:.1f}%",
        f"AI Sharpe:  {ai_sharpe:.2f}",
        f"SPY Sharpe: {spy_sharpe:.2f}",
        f"MKT Corr:  {correlation:.2f}",
        f"CAPM Alpha (Ann): {capm_alpha_annual_pct:+.1f}%",
        f"Info Ratio: {info_ratio:.2f}",
        f"VaR95 Hist: {ai_risk_metrics['historical_var'] * 100:.2f}%",
        f"CVaR95 Hist: {ai_risk_metrics['historical_cvar'] * 100:.2f}%",
        f"VaR95 Param: {ai_risk_metrics['parametric_var'] * 100:.2f}%",
        f"CVaR95 Param: {ai_risk_metrics['parametric_cvar'] * 100:.2f}%",
        "",
        "MARKET REGIME",
        f"Avg@SPY Down: {ai_avg_down_day:+.3f}%",
        f"Avg@SPY Up:   {ai_avg_up_day:+.3f}%",
        f"Excess@Down:  {excess_avg_down_day:+.3f}%",
        f"Excess@Up:    {excess_avg_up_day:+.3f}%",
        "",
        "TRADE ANALYTICS",
        f"Win Rate:  {win_rate:.1f}%",
        f"PF Factor: {profit_factor:.2f}",
        f"Avg Win:   +{avg_win:.1f}%",
        f"Avg Loss:  {avg_loss:.1f}%",
        f"Trades:    {len(export_trades):,}",
        f"Exposure:  {avg_invested:.1f}%",
        "",
        "STRATEGY SETTINGS",
        f"Start Cap: ${starting_capital:,.0f}",
        f"Max Trades/Day: {max_trades_per_day}",
        f"Slippage: {slippage_and_fees*10000:.0f} BPS",
        f"Stop-Loss: {hard_stop_loss*100:.0f}%",
    ]
    if take_profit is not None:
        summary_lines.append(f"Take-Profit: {take_profit*100:.0f}%")
    if signal_delay_bdays > 0:
        summary_lines.append(f"Delay:    {signal_delay_bdays} Days")
    if min_rank_percentile is not None:
        summary_lines.append(f"Min Rank: {min_rank_percentile:.2f}")
    if contradiction_exit_enabled:
        summary_lines.append(f"Contra Exits: {contradiction_exits}")

    fig = plt.figure(figsize=(19, 10.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.3, 2.2], height_ratios=[3.2, 1], wspace=0.06, hspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 1])
    ax1.plot(portfolio.index, portfolio["AI_Equity"], label=f"Sentinel 20-Day (ROI: {ai_roi:.1f}%)", color="#0a7f45", linewidth=2.8)
    ax1.plot(portfolio.index, portfolio["SPY_Equity"], label=f"S&P 500 (ROI: {spy_roi:.1f}%)", color="#2f3a45", linestyle="--", linewidth=2.4)
    ax1.set_title(f"Sentinel AI vs Benchmark ({variant_name})", fontsize=16, fontweight="bold", pad=8)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=13)
    ax1.tick_params(axis="both", labelsize=11)
    ax1.legend(loc="upper left", fontsize=12, frameon=True, facecolor="white", edgecolor="#bbbbbb")
    ax1.grid(color="#b3b3b3", linestyle=":", alpha=0.7)

    ax2.fill_between(portfolio.index, portfolio["AI_Drawdown"] * 100, 0, color="#d9534f", alpha=0.45)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.tick_params(axis="both", labelsize=10)
    ax2.grid(color="#b3b3b3", linestyle=":", alpha=0.7)
    ax3.axis("off")
    ax3.text(
        0.02,
        0.99,
        "\n".join(summary_lines),
        transform=ax3.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        linespacing=1.1,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8f8f8", "edgecolor": "#b8b8b8", "alpha": 0.98},
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.085)
    plt.savefig(plot_export, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.2)

    exposure_export = os.path.join(output_dir, f"exposure_{out_prefix}.png")
    exposure_fig = plt.figure(figsize=(18, 8))
    exposure_ax = exposure_fig.add_subplot(111)
    exposure_ax.plot(portfolio.index, portfolio["Gross_Exposure"], label="Gross Exposure", color="#8b0000", linewidth=2.2)
    exposure_ax.plot(portfolio.index, portfolio["Net_Exposure"], label="Net Exposure", color="#0a7f45", linewidth=2.2)
    exposure_ax.set_title("Historical Gross and Net Exposure", fontsize=18, fontweight="bold", pad=10)
    exposure_ax.set_ylabel("Exposure (%)", fontsize=14)
    exposure_ax.set_xlabel("Date", fontsize=14)
    exposure_ax.tick_params(axis="both", labelsize=11)
    exposure_ax.grid(color="#b3b3b3", linestyle=":", alpha=0.7)
    exposure_ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=12)
    exposure_fig.text(
        0.08,
        0.03,
        "Gross exposure is total long notional; net exposure subtracts idle cash.",
        fontsize=9,
        color="#555555",
        style="italic",
    )
    exposure_fig.subplots_adjust(left=0.08, right=0.83, top=0.90, bottom=0.15)
    exposure_fig.savefig(exposure_export, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(exposure_fig)
    print(f"Saved exposure chart to: {exposure_export}")
