import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
import pandas as pd
import yfinance as yf
import time
import requests
import csv
import re
import numpy as np
from io import StringIO
from datasets import load_dataset 
import json
import warnings
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import shutil
import contextlib
import hashlib

HF_TOKEN = os.getenv("HF_TOKEN")

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow is deprecated.*")

NEWS_SOURCES = [
    {"name": "FNSPID_Nasdaq", "huggingface_id": "Zihan1004/FNSPID", "data_files": "Stock_news/nasdaq_exteral_data.csv", "split": "train"},
    {"name": "Financial_News_2024", "huggingface_id": "m-ric/financial-news-2024", "data_files": "**/*.parquet", "split": "train"},
    {"name": "Financial_News_MultiSource_2025", "huggingface_id": "Brianferrell787/financial-news-multisource", "data_files": "data/**/*.parquet", "split": "train"}
]

COLUMN_ALIASES = {
    'ticker':  ['stock_symbol', 'ticker', 'symbol', 'stock', 'code'],
    'date':    ['date', 'timestamp', 'time', 'published_at', 'datetime', 'day'],
    'title':   ['article_title', 'title', 'headline', 'header', 'subject'],
    'content': ['article', 'content', 'body', 'text', 'summary', 'news', 'description'],
    'url':     ['url', 'link', 'original_url', 'href', 'source_url'],
    'extra':   ['extra_fields', 'meta', 'metadata', 'info'] 
}

TRAIN_NEWS_DIR = "001_ticker_news_train"
BACKTEST_NEWS_DIR = "001_ticker_news_backtest"
DAILY_DIR = "001_final_db_daily"
MARKET_TICKER = "^GSPC"
MACRO_FILE = "001_macro_data.csv"
WS_PATTERN = re.compile(r'[\r\n\t\x0b\x0c]+')

# =========================================================
# --- UNLEASHED BROAD MARKET CONFIGURATION ---
# =========================================================

# 1. 🟢 DISABLED: Extracts news for ALL available market tickers
RESTRICT_TO_WHITELIST = False 

# We still provide the S&P 500 list to guarantee they are caught in raw text scanning fallback
RAW_TICKERS = [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
    "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK",
    "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT",
    "APP", "APTV", "ACGL", "ADM", "ARES", "ANET", "AJG", "AIZ", "T", "ATO",
    "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX",
    "BDX", "BRK.B", "BBY", "TECH", "BIIB", "BLK", "BX", "XYZ", "BK", "BA",
    "BKNG", "BSX", "BMY", "AVGO", "BR", "BRO", "BF.B", "BLDR", "BG", "BXP",
    "CHRW", "CDNS", "CPT", "CPB", "COF", "CAH", "CCL", "CARR", "CVNA", "CAT",
    "CBOE", "CBRE", "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR",
    "CVX", "CMG", "CB", "CHD", "CIEN", "CI", "CINF", "CTAS", "CSCO", "C",
    "CFG", "CLX", "CME", "CMS", "KO", "CTSH", "COIN", "CL", "CMCSA", "FIX",
    "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT", "GLW", "CPAY", "CTVA",
    "CSGP", "COST", "CTRA", "CRH", "CRWD", "CCI", "CSX", "CMI", "CVS", "DHR",
    "DRI", "DDOG", "DVA", "DECK", "DE", "DELL", "DAL", "DVN", "DXCM", "FANG",
    "DLR", "DG", "DLTR", "D", "DPZ", "DASH", "DOV", "DOW", "DHI", "DTE",
    "DUK", "DD", "ETN", "EBAY", "ECL", "EIX", "EW", "EA", "ELV", "EME",
    "EMR", "ETR", "EOG", "EPAM", "EQT", "EFX", "EQIX", "EQR", "ERIE", "ESS",
    "EL", "EG", "EVRG", "ES", "EXC", "EXE", "EXPE", "EXPD", "EXR", "XOM",
    "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB", "FSLR", "FE",
    "FISV", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT",
    "GE", "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS", "GM", "GPC", "GILD",
    "GPN", "GL", "GDDY", "GS", "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC",
    "HSY", "HPE", "HLT", "HOLX", "HD", "HON", "HRL", "HST", "HWM", "HPQ",
    "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX", "ITW", "INCY", "IR",
    "PODD", "INTC", "IBKR", "ICE", "IFF", "IP", "INTU", "ISRG", "IVZ", "INVH",
    "IQV", "IRM", "JBHT", "JBL", "JKHY", "J", "JNJ", "JCI", "JPM", "KVUE",
    "KDP", "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KLAC", "KHC", "KR",
    "LHX", "LH", "LRCX", "LW", "LVS", "LDOS", "LEN", "LII", "LLY", "LIN",
    "LYV", "LMT", "L", "LOW", "LULU", "LYB", "MTB", "MPC", "MAR", "MRSH",
    "MLM", "MAS", "MA", "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "META",
    "MET", "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "MOH", "TAP",
    "MDLZ", "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP",
    "NFLX", "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS",
    "NOC", "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL",
    "OMC", "ON", "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PLTR", "PANW", "PSKY",
    "PH", "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX",
    "PNW", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU",
    "PEG", "PTC", "PSA", "PHM", "PWR", "QCOM", "DGX", "Q", "RL", "RJF", "RTX",
    "O", "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "HOOD", "ROK", "ROL",
    "ROP", "ROST", "RCL", "SPGI", "CRM", "SNDK", "SBAC", "SLB", "STX", "SRE",
    "NOW", "SHW", "SPG", "SWKS", "SJM", "SW", "SNA", "SOLV", "SO", "LUV",
    "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS", "SYY",
    "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TER", "TSLA",
    "TXN", "TPL", "TXT", "TMO", "TJX", "TKO", "TTD", "TSCO", "TT", "TDG",
    "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UBER", "UDR", "ULTA", "UNP",
    "UAL", "UPS", "URI", "UNH", "UHS", "VLO", "VTR", "VLTO", "VRSN", "VRSK",
    "VZ", "VRTX", "VTRS", "VICI", "V", "VST", "VMC", "WRB", "GWW", "WAB",
    "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC",
    "WY", "WSM", "WMB", "WTW", "WDAY", "WYNN", "XEL", "XYL", "YUM", "ZBRA",
    "ZBH", "ZTS"
]

TICKER_WHITELIST = [t.replace('.', '-').strip().upper() for t in RAW_TICKERS]
if RESTRICT_TO_WHITELIST:
    print(f"✅ Restricted extraction to {len(TICKER_WHITELIST)} hardcoded S&P 500 tickers.")
else:
    print(f"🚀 UNLEASHED MODE: Extracting all available market tickers from datasets.")

if not HF_TOKEN:
    print("Warning: HF_TOKEN is not set. Public datasets will still load, but private gated datasets may fail.")

# 2. 🟢 ERA FILTER: Start 2015 to give 2016-01-01 ML models a 1-year warmup buffer
START_YEAR = 2015

# 3. ENABLE general/macro market news
EXTRACT_GENERAL_NEWS = True 
MAX_ROWS_PER_SOURCE = None 
# =========================================================

for folder in [TRAIN_NEWS_DIR, BACKTEST_NEWS_DIR, DAILY_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def map_columns(actual_columns):
    mapping = {}
    lower_cols = {c.lower(): c for c in actual_columns}
    for standard_field, aliases in COLUMN_ALIASES.items():
        found = next((lower_cols[a] for a in aliases if a in lower_cols), None)
        if not found:
            for alias in aliases:
                for col_name in lower_cols.keys():
                    if alias in col_name:
                        found = lower_cols[col_name]
                        break
                if found: break
        mapping[standard_field] = found
    return mapping

def sanitize_text(text):
    if text is None: return ""
    text = WS_PATTERN.sub(' ', str(text)).strip().replace('"', "'")
    return text[:29997] + "..." if len(text) > 30000 else text

def download_macro_context():
    print("--- Downloading Macro Context (SPY, VIX & FRED) ---")
    try:
        spy = yf.download("SPY", start=f"{START_YEAR}-01-01", progress=False)
        vix = yf.download("^VIX", start=f"{START_YEAR}-01-01", progress=False)
        
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = [c[0] for c in spy.columns]
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = [c[0] for c in vix.columns]
            
        spy_df = spy[['Open', 'High', 'Low', 'Close', 'Volume']].add_prefix('SPY_')
        vix_df = vix[['Open', 'High', 'Low', 'Close']].add_prefix('VIX_')
        
        df_yfinance = spy_df.join(vix_df, how='outer')
        df_yfinance.index = pd.to_datetime(df_yfinance.index, utc=True).tz_convert(None)
        
        dfs_fred = []
        for s_id in ['FEDFUNDS', 'CPIAUCSL', 'M2SL']:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s_id}&cosd={START_YEAR}-01-01"
            try:
                r = requests.get(url, timeout=10)
                fred_df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
                # Physical delay to simulate real-world publication lag
                fred_df.index = pd.to_datetime(fred_df.index).tz_localize(None) + pd.Timedelta(days=45)
                dfs_fred.append(fred_df)
            except Exception as e: 
                print(f"    Warning: Failed to fetch {s_id}: {e}")
                
        if dfs_fred:
            df_fred = pd.concat(dfs_fred, axis=1)
            df_macro = df_yfinance.join(df_fred, how='outer')
        else:
            df_macro = df_yfinance
        
        df_macro.index.name = 'Date'
        # FIX: ffill is safe (past to future), but bfill is LEAKAGE (future to past).
        # We allow the first 45 days of the dataset to be NaN to remain honest.
        df_macro.ffill(inplace=True) 
        
        df_macro.to_csv(MACRO_FILE)
        print(f"Success! Saved {MACRO_FILE} with {len(df_macro.columns)} macro features.")
        return df_macro
    except Exception as e:
        print(f"Macro Download Failed: {e}")
        return pd.DataFrame()

def process_shard(args):
    source, shard_id, num_shards, allowed_tickers, train_dir, backtest_dir, market_ticker, market_keywords, hf_token, shared_counter, lock, start_time, ticker_whitelist, start_year, extract_general, max_rows = args
    if hf_token: os.environ["HF_TOKEN"] = hf_token
    
    ticker_set = set(allowed_tickers)
    if ticker_whitelist:
        ticker_set.update(set(t.upper() for t in ticker_whitelist))

    ticker_regex = None
    if ticker_set:
        sorted_allowed = sorted(list(ticker_set), key=len, reverse=True)
        pattern = r'\b(' + '|'.join(map(re.escape, [t for t in sorted_allowed if len(t) > 1])) + r')\b'
        ticker_regex = re.compile(pattern)

    temp_writers = {}
    last_log_time = time.time()
    
    ticker_safe_cache = {}
    seen_hashes = set() 
    
    try:
        ds = load_dataset(source['huggingface_id'], data_files=source['data_files'], split=source['split'], streaming=True)
        
        try:
            work_ds = ds.shard(num_shards=num_shards, index=shard_id)
            is_manual_shard = False
        except (IndexError, ValueError):
            work_ds = ds
            is_manual_shard = True
            print(f"Shard {shard_id}: Using manual skip-sharding.")
        
        col_map = None
        row_count = 0
        processed_count = 0
        
        for row in work_ds:
            if is_manual_shard:
                if row_count % num_shards != shard_id:
                    row_count += 1
                    continue
            
            if col_map is None:
                col_map = map_columns(list(row.keys()))
            
            tickers = []
            if col_map['ticker'] and row.get(col_map['ticker']):
                t_raw = row.get(col_map['ticker'])
                tickers = t_raw if isinstance(t_raw, list) else [t.strip() for t in str(t_raw).replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",")]
            
            if not tickers and col_map['extra'] and row.get(col_map['extra']):
                try:
                    data = json.loads(row.get(col_map['extra'])) if isinstance(row.get(col_map['extra']), str) else row.get(col_map['extra'])
                    tickers = data.get('stocks') or data.get('tickers') or []
                except: pass

            date_val = str(row.get(col_map['date'], ""))
            title_val = sanitize_text(row.get(col_map['title'], ""))
            content_val = sanitize_text(row.get(col_map['content'], ""))
            full_text = f"{title_val} {content_val}".strip()

            if not tickers and ticker_regex:
                found = ticker_regex.findall(full_text.upper())
                if found: tickers = list(set(found))

            if not tickers: tickers = [market_ticker]

            try:
                year_str = date_val[:4]
                if year_str.isdigit():
                    year = int(year_str)
                else:
                    year = pd.to_datetime(date_val).year
            except: year = 0
            
            if year < start_year and year != 0:
                row_count += 1
                continue
            
            article_hash = hashlib.md5(f"{date_val}_{title_val}_{full_text[:200]}".encode('utf-8')).hexdigest()
            if article_hash in seen_hashes:
                row_count += 1
                continue
            seen_hashes.add(article_hash)
            
            target_dir = backtest_dir if year >= 2024 else train_dir
            is_macro = any(k in full_text.lower() for k in market_keywords)
            
            processed_for_row = set()
            for t in tickers:
                if not t or len(str(t)) > 10: continue
                t = str(t).upper().replace('.', '-')
                
                # Check whitelist constraint if active
                if RESTRICT_TO_WHITELIST and t not in ticker_set and t != market_ticker: 
                    continue

                if t in processed_for_row: continue
                processed_for_row.add(t)
                
                if t not in ticker_safe_cache:
                    ticker_safe_cache[t] = re.sub(r'[^\w\-. ]', '_', t)
                safe_t = ticker_safe_cache[t]
                
                shard_file = os.path.join(target_dir, f"__shard_{safe_t}_{shard_id}.tmp")
                if shard_file not in temp_writers:
                    temp_writers[shard_file] = open(shard_file, "a", encoding="utf-8", newline="")
                
                csv.writer(temp_writers[shard_file], quoting=csv.QUOTE_ALL).writerow([source['name'], date_val, title_val, full_text, row.get(col_map['url'], "")])
            
            if is_macro and market_ticker not in processed_for_row:
                if extract_general: 
                    if market_ticker not in ticker_safe_cache:
                        ticker_safe_cache[market_ticker] = re.sub(r'[^\w\-. ]', '_', market_ticker)
                    safe_t = ticker_safe_cache[market_ticker]
                    
                    shard_file = os.path.join(target_dir, f"__shard_{safe_t}_{shard_id}.tmp")
                    if shard_file not in temp_writers:
                        temp_writers[shard_file] = open(shard_file, "a", encoding="utf-8", newline="")
                    csv.writer(temp_writers[shard_file], quoting=csv.QUOTE_ALL).writerow([source['name'], date_val, title_val, full_text, row.get(col_map['url'], "")])

            row_count += 1
            processed_count += 1
            
            if max_rows and row_count >= max_rows:
                break
            
            if processed_count % 5000 == 0:
                with lock:
                    shared_counter.value += 5000
                    total_so_far = shared_counter.value
                
                now = time.time()
                shard_elapsed = now - last_log_time
                total_elapsed = now - start_time
                
                if shard_elapsed > 0 and total_elapsed > 0:
                    shard_speed = 5000 / shard_elapsed
                    total_speed = total_so_far / total_elapsed
                    print(f"[{total_so_far:,} total] Shard {shard_id}: {shard_speed:.1f} art/sec | COMBINED SPEED: {total_speed:.1f} art/sec")
                
                last_log_time = time.time()
                
    except Exception as e:
        print(f"Error in shard {shard_id}: {e}")
    finally:
        for f in temp_writers.values(): f.close()
    
    return row_count

def get_news_by_ticker():
    print("--- Executing Phase 1: Optimized Parallel Extraction ---")
    allowed_tickers = set()
    if os.path.exists(TRAIN_NEWS_DIR):
        for f in os.listdir(TRAIN_NEWS_DIR):
            if f.endswith('_news.csv'): allowed_tickers.add(f.replace('_news.csv', '').upper())
    
    market_keywords = [
        "s&p 500", "sp500", "nasdaq 100", "dow jones", "fomc", "federal reserve", 
        "interest rates", "inflation", "cpi", "macroeconomic", "recession", 
        "geopolitical", "global market", "central bank", "nonfarm payrolls"
    ]
    
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        num_cpus = int(slurm_cpus)
        print(f"Detected Slurm environment: Using {num_cpus} CPUs as requested.")
    else:
        num_cpus = mp.cpu_count()
        print(f"No Slurm detected: Using all {num_cpus} available CPU cores.")
    
    for source in NEWS_SOURCES:
        print(f"--- Processing source: {source['name']} ---")
        
        is_shardable = "*" in str(source.get('data_files', ""))
        current_shards = num_cpus if is_shardable else 1
        
        print(f"  Extraction with {current_shards} worker(s)...")
        
        manager = mp.Manager()
        shared_counter = manager.Value('i', 0)
        lock = manager.Lock()
        
        start_time = time.time()
        args_list = [(source, i, current_shards, allowed_tickers, TRAIN_NEWS_DIR, BACKTEST_NEWS_DIR, MARKET_TICKER, market_keywords, HF_TOKEN, shared_counter, lock, start_time, TICKER_WHITELIST, START_YEAR, EXTRACT_GENERAL_NEWS, MAX_ROWS_PER_SOURCE) for i in range(current_shards)]
        with ProcessPoolExecutor(max_workers=current_shards) as executor:
            futures = [executor.submit(process_shard, args) for args in args_list]
            total_rows = 0
            with tqdm.tqdm(total=current_shards, desc=f"Workers for {source['name']}", unit="shard") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if isinstance(res, int): total_rows += res
                    pbar.update(1)
        
        total_time = time.time() - start_time
        avg_speed = total_rows / total_time if total_time > 0 else 0
        print(f"--- Completed {source['name']} ---")
        print(f"  Total Extracted: {total_rows:,} articles")
        print(f"  Average Throughput: {avg_speed:.2f} articles/second (Combined)")
        
        new_count = 0
        for d in [TRAIN_NEWS_DIR, BACKTEST_NEWS_DIR]:
            for f in os.listdir(d):
                if f.startswith("__shard_") and f.endswith(".tmp"):
                    t = f.split("__shard_")[1].rsplit("_", 1)[0]
                    if t.upper() not in allowed_tickers:
                        allowed_tickers.add(t.upper())
                        new_count += 1
        if new_count > 0:
            print(f"  Discovered {new_count} new tickers. Total universe: {len(allowed_tickers)}")

    print("--- Merging Shards ---")
    for d in [TRAIN_NEWS_DIR, BACKTEST_NEWS_DIR]:
        shard_files = [f for f in os.listdir(d) if f.startswith("__shard_") and f.endswith(".tmp")]
        ticker_groups = {}
        for f in shard_files:
            ticker = f.split("__shard_")[1].rsplit("_", 1)[0]
            ticker_groups.setdefault(ticker, []).append(f)
        
        for ticker, files in tqdm.tqdm(ticker_groups.items(), desc=f"Merging {d}"):
            dest_path = os.path.join(d, f"{ticker}_news.csv")
            file_exists = os.path.exists(dest_path)
            with open(dest_path, "a", encoding="utf-8", newline="") as f_out:
                writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
                if not file_exists:
                    writer.writerow(["source_dataset", "date", "title", "content", "original_url"])
                for f_in_name in files:
                    f_in_path = os.path.join(d, f_in_name)
                    with open(f_in_path, "r", encoding="utf-8") as f_in:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(f_in_path)

    all_symbols = set()
    for d in [TRAIN_NEWS_DIR, BACKTEST_NEWS_DIR]:
        for f in os.listdir(d):
            if f.endswith("_news.csv"): all_symbols.add(f.replace("_news.csv", "").upper())
            
    return sorted(list(all_symbols))

def deduplicate_file(path):
    if not os.path.exists(path): return
    temp_path = path + ".tmp"
    seen_hashes = set()
    try:
        with open(path, 'r', encoding='utf-8', newline='') as f_in, open(temp_path, 'w', encoding='utf-8', newline='') as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
            try:
                header = next(reader)
                writer.writerow(header)
                header_lower = [h.lower() for h in header]
                date_idx = header_lower.index("date") if "date" in header_lower else -1
                title_idx = header_lower.index("title") if "title" in header_lower else -1
                content_idx = header_lower.index("content") if "content" in header_lower else -1
            except StopIteration: return

            if date_idx == -1 or title_idx == -1: return

            for row in reader:
                if not row: continue
                content_snippet = row[content_idx][:200] if content_idx != -1 and len(row) > content_idx else ""
                row_hash = hashlib.md5(f"{row[date_idx]}_{row[title_idx]}_{content_snippet}".encode('utf-8')).hexdigest()
                if row_hash not in seen_hashes:
                    writer.writerow(row)
                    seen_hashes.add(row_hash)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path): os.remove(temp_path)

def deduplicate_directory(directory):
    print(f"--- Parallel Deduplicating: {directory} ---")
    if not os.path.exists(directory): return
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('_news.csv')]
    if files:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            list(tqdm.tqdm(executor.map(deduplicate_file, files), total=len(files), desc=f"Deduplicating {os.path.basename(directory)}"))

def get_stock_data(ticker, macro_df):
    try:
        if not ticker or ticker.startswith('_') or len(str(ticker)) > 10: return "SKIPPED_INVALID"
        safe_ticker = re.sub(r'[^\w\-. ]', '_', str(ticker))
        daily_path = os.path.join(DAILY_DIR, f"{safe_ticker}_daily.csv")
        status_parts = []

        # 1. DAILY
        if os.path.exists(daily_path): status_parts.append("Daily:Exists")
        else:
            try:
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    d = yf.download(ticker, start=f"{START_YEAR}-01-01", progress=False)
                
                if not d.empty:
                    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                    d.columns = [c.lower() for c in d.columns]
                    
                    # FIX: Create 'adj open' for accurate split-adjusted multi-day targeting
                    if 'open' in d.columns and 'close' in d.columns and 'adj close' in d.columns:
                        d['adj open'] = (d['open'] / d['close']) * d['adj close']
                    
                    # Removed lookahead survivorship bias (tail 20 days mean)
                    # Institutional liquidity gates should be handled dynamically in training/inference.

                    d.index = pd.to_datetime(d.index, utc=True).tz_convert(None)
                    
                    try:
                        import pandas_ta_classic as ta
                        d.ta.rsi(length=14, append=True)
                        d.ta.macd(append=True)
                    except: pass
                    
                    macro_aligned = macro_df.reindex(d.index, method='ffill')
                    final_d = d.join(macro_aligned, how="left").dropna()
                    final_d.to_csv(daily_path)
                    status_parts.append("Daily:Saved")
                else: status_parts.append("Daily:NoData")
            except: status_parts.append("Daily:Error")

        return "|".join(status_parts)
    except Exception: return "FAILED_EXCEPTION"

if __name__ == "__main__":
    symbols = []
    symbols = get_news_by_ticker()
    deduplicate_directory(TRAIN_NEWS_DIR)
    deduplicate_directory(BACKTEST_NEWS_DIR)

    if not symbols and os.path.exists(TRAIN_NEWS_DIR):
        files = [f for f in os.listdir(TRAIN_NEWS_DIR) if f.endswith('_news.csv')]
        symbols = sorted([f.replace('_news.csv', '') for f in files])

    macro_data = download_macro_context()

    if symbols and not macro_data.empty:
        print(f"📊 Resuming Price Download for {len(symbols)} identified tickers...")
        counters = {"Daily": 0, "Skipped": 0, "Errors": 0}
        
        for i, sym in enumerate(symbols):
            status = get_stock_data(sym, macro_data)
            if "Saved" in status: counters["Daily"] += 1
            if "Exists" in status: counters["Skipped"] += 1
            if "Error" in status or "FAILED" in status: counters["Errors"] += 1
            if "NoData" in status: counters.setdefault("NoData/Delisted", 0); counters["NoData/Delisted"] += 1

            print(f"[{i+1}/{len(symbols)}] {sym}: {status} | Stats: {counters}", end='\r')
            if "Saved" in status or "FAILED" in status: time.sleep(0.5) 

    print("\nSENTINEL FULL PIPELINE COMPLETE.")
