import os
import glob
import re

# ==========================================
# CLEANUP CONFIGURATION
# ==========================================
# We include all possible data folders to ensure consistency
folders_to_clean = [
    "001_ticker_news_train",
    "001_ticker_news_backtest",
    "001_final_db_daily",
    # "002_final_vectors_train",
    # "002_final_vectors_backtest",
]

# CRITICAL: These are S&P 500 stocks that use hyphens. DO NOT DELETE.
# BRK.B -> BRK-B, BF.B -> BF-B in yfinance/pipeline logic
PROTECTED_TICKERS = ["BRK-A", "BRK-B", "BF-A", "BF-B"]

# Patterns to identify "Impurities":
# 1. Crypto: Ends in -USD
# 2. International: Ends in -SZ (Shenzhen), -HK (Hong Kong), -KS (Korea), -SS (Shanghai), -TO (Toronto), -AX (ASX), etc.
# 3. Numeric tickers: e.g., 000001-SZ
IMPURITY_PATTERNS = [
    r".*-USD\.csv$",   # Crypto
    r".*-[A-Z]{2}\.csv$", # International market suffixes (SZ, HK, KS, TO, AX, etc.)
    r"^[0-9].*\.csv$"    # Numeric-start tickers (common in Asian markets)
]

deleted_count = 0
protected_count = 0

print("--- 🧹 SENTINEL: Non-Equity Cleanup Started ---")

for folder in folders_to_clean:
    if not os.path.exists(folder):
        continue
    
    print(f"Checking folder: {folder}...")
    
    # Get all CSV files in the folder
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        ticker = filename.replace("_news.csv", "").replace("_daily.csv", "").replace("_hourly.csv", "").replace("_vec.csv", "").replace("_deepseek.csv", "").upper()
        
        # Check if it's protected
        if ticker in PROTECTED_TICKERS:
            protected_count += 1
            continue
            
        # Check against impurity patterns
        is_impure = False
        for pattern in IMPURITY_PATTERNS:
            if re.match(pattern, filename, re.IGNORECASE):
                is_impure = True
                break
        
        if is_impure:
            try:
                os.remove(file_path)
                print(f"  [DELETED] {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"  [ERROR] Could not delete {filename}: {e}")

print("\n--- Cleanup Complete! ---")
print(f"Total Deleted:   {deleted_count}")
print(f"Total Protected: {protected_count} (e.g., BRK-B, BF-B)")
print("--------------------------")
