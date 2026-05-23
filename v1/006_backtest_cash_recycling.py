from backtest_common_006 import run_backtest
# ==============================================================================
# 🎯 SENTINEL AI: CASH RECYCLING BACKTEST TUNER
# Instructions: This variant reinvests realized gains into future trade sizing.
# ==============================================================================

# 1. CORE STRATEGY TOGGLES
USE_DEEPSEEK = False        # True: Uses 'final_ensemble_results_20.csv' | False: Uses '...nodis_20.csv'
EXECUTION_MODEL = "AVG"     # "AVG": Daily average price | "OPEN": Next-day open price
FILTER_SP500 = True         # True: S&P 500 components only | False: All tickers in database
SIGNAL_DELAY_BDAYS = 0      # 0: Execute on same day | 1: Execute on next-day signal

# 2. TRADE FILTERS & LIMITS
MAX_TRADES_PER_DAY = 1      # Maximum number of concurrent new trades per day
MIN_XGB_SCORE = -0.5       # Minimum XGBoost rank score to enter a trade
MIN_PRED_RETURN = 1       # Minimum PatchTST predicted return (%) to enter
MIN_RANK_PERCENTILE = None  # (Optional) e.g., 0.9 to only buy the top 10% daily scores

# 3. RISK MANAGEMENT
HOLDING_PERIOD = 15         # Days until automatic exit
LEVERAGE = 1.0              # Portfolio leverage (1.0 = no leverage)
HARD_STOP_LOSS = -0.4       # Exit immediately if position drops this much (-0.1 = -10%)
TAKE_PROFIT = None           # Exit immediately if position reaches this gain (+0.1 = +10%)
SLIPPAGE_AND_FEES = 0.01    # (1.00%) Cost per trade (both ways)

# 4. SIMULATION SETTINGS
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"
STARTING_CAPITAL = 100000

# 5. CASH RECYCLING
CASH_RECYCLING = True       # Reinvest realized gains into the next trade sizes

# ==============================================================================
# 🚀 EXECUTION LOGIC (Do not change)
# ==============================================================================

results_file = "final_ensemble_results_20.csv"
variant_tag = ("DS" if USE_DEEPSEEK else "NoDS") + f"_{EXECUTION_MODEL}" + ("_All" if not FILTER_SP500 else "")

CONFIG = {
    "variant_name": f"Sentinel_CashRecycle_{variant_tag}",
    "results_files": [results_file],
    "out_prefix": f"CashRecycle_Max{MAX_TRADES_PER_DAY}_delay{SIGNAL_DELAY_BDAYS}_{EXECUTION_MODEL}",
    "execution_model": EXECUTION_MODEL,
    "filter_sp500": FILTER_SP500,
    "signal_delay_bdays": SIGNAL_DELAY_BDAYS,
    "max_trades_per_day": MAX_TRADES_PER_DAY,
    "min_xgb_score": MIN_XGB_SCORE,
    "min_pred_return": MIN_PRED_RETURN,
    "min_rank_percentile": MIN_RANK_PERCENTILE,
    "cash_recycling": CASH_RECYCLING,
    "leverage": LEVERAGE,
    "slippage_and_fees": SLIPPAGE_AND_FEES,
    "hard_stop_loss": HARD_STOP_LOSS,
    "take_profit": TAKE_PROFIT,
    "starting_capital": STARTING_CAPITAL,
    "holding_period": HOLDING_PERIOD,
    "simulation_start_date": START_DATE,
    "simulation_end_date": END_DATE,
    # System Paths
    "macro_file": "001_macro_data.csv",
    "price_dir_candidates": ["001_final_db_daily", "final_db_daily"],
    "output_root": "006_outputs",
    "risk_free_rate": 0.0,
}

if __name__ == "__main__":
    print(f"--- Launching Cash Recycling Backtest: {variant_tag} ---")
    run_backtest(CONFIG)
