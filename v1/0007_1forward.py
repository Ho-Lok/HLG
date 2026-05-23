"""
FORWARD PREDICTION PIPELINE (0007_1forward.py)

USAGE:
    Run individually (generates predictions):
        python 0007_1forward.py
  
    With options:
        python 0007_1forward.py --help
  
    Run as part of daily workflow (via orchestrator):
        python 0007_0run_daily.py

OUTPUTS:
    - final_ensemble_results_live_v1.csv (all ~500 S&P tickers ranked by score)
    - Sentinel_v1_Recommendations_History.csv (top picks + AI analysis)
    - Individual ticker news files in 0001_ticker_news/

DEPENDENCIES:
    - PatchTST model predictions
    - XGBoost ranking scores
    - LLM analysis (TradingAgents optional)
"""
import os
import json
import argparse
from datetime import datetime, timedelta, timezone
import re
import xml.etree.ElementTree as ET
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import joblib
import yfinance as yf
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import alpaca_trade_api as tradeapi
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
import warnings
import concurrent.futures
from functools import partial
import logging
import threading
from zoneinfo import ZoneInfo
from tradingagents.llm_clients import create_llm_client

warnings.filterwarnings("ignore")
logging.getLogger("alpaca_trade_api").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Alpaca news endpoint is rate-limited. Keep this conservative to avoid retry storms.
ALPACA_NEWS_ENABLED = True
# 0 (or negative) means use Alpaca for all tickers in the scrape target list.
ALPACA_MAX_NEWS_TICKERS = 0
ALPACA_MAX_CONCURRENT_REQUESTS = 4
# FutuBull news scraping (good for Chinese stocks and fintech)
FUTUBULL_NEWS_ENABLED = True

MIN_PRED_RETURN       = 1
MIN_XGB_SCORE         = -0.25
# Hard minimum liquidity filter: never allow turnover threshold below $5M.
MIN_TURNOVER_DOLLARS  = max(5_000_000, float(os.getenv("MIN_TURNOVER_DOLLARS", "5000000")))
TOP_N_RECOMMENDATIONS = 3
TOP_N_CROSSCHECK = 3
# Enabled: use a direct Gemini cross-check call for an independent second opinion.
TRADINGAGENTS_CROSSCHECK_ENABLED = True

# OpenBB adds extra network calls and slows the forward run down.
# Keep it off by default; flip this to True only if you want enrichment.
OPENBB_ENABLED = False
if OPENBB_ENABLED:
    try:
        from openbb import obb
        OPENBB_AVAILABLE = True
    except Exception:
        obb = None
        OPENBB_AVAILABLE = False
else:
    obb = None
    OPENBB_AVAILABLE = False

try:
    # tradingagents import is attempted after environment keys are set below.
    TRADINGAGENTS_AVAILABLE = False
except Exception:
    TRADINGAGENTS_AVAILABLE = False

# ----------------------
# API KEYS / CREDENTIALS
# Prefer environment variables. You can paste keys below into
# `TRADINGAGENTS_API_KEYS` for quick local testing (not recommended
# for production). The script will set those into `os.environ` so
# TradingAgents and other libs can pick them up at runtime.
# ----------------------

# Alpaca (used for news endpoints)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PKODLUQU32XQ2LEIYG2TGQJV4W")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "rBr77s9mRfvshPoRENgS1MLkw2NnSkEF8Jv3u7TyTWE")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# NYT key already read from env above (NYT_API_KEY)

# TradingAgents / LLM provider keys
# For runtime convenience you may paste real keys into the values below
# or set them in your environment (recommended). Empty string = not set.
TRADINGAGENTS_API_KEYS = {
    # Primary OpenAI-compatible key (keeps compatibility for OpenAI-like providers)
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    # Google key (kept for Google provider usage)
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
    "XAI_API_KEY": os.getenv("XAI_API_KEY", ""),
    # DeepSeek-specific key (alias, in case provider is set to 'deepseek')
    "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY", ""),
    "DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY", ""),
    "ZHIPU_API_KEY": os.getenv("ZHIPU_API_KEY", ""),
    "ALPHA_VANTAGE_API_KEY": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
}

# ----------------------------
# DeepSeek official defaults (editable here)
# Replace the string below with your official DeepSeek API key or set the
# DEEPSEEK_API_KEY / DEEPSEEK_API_URL environment variables instead.
# ----------------------------
# NOTE: prefer the official DeepSeek host (docs show api.deepseek.com). Do not include a /v1
# path in the default here; we'll append or normalize paths for specific clients as needed.
DEEPSEEK_OFFICIAL_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-e5b962ae760042878a1b09bdb09b795d")
# Default DeepSeek API base URL — edit if DeepSeek publishes a different endpoint.
DEEPSEEK_OFFICIAL_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com")

# Ensure the TRADINGAGENTS dict uses the DeepSeek key by default if available
if not TRADINGAGENTS_API_KEYS.get("DEEPSEEK_API_KEY"):
    TRADINGAGENTS_API_KEYS["DEEPSEEK_API_KEY"] = DEEPSEEK_OFFICIAL_API_KEY


def _apply_tradingagents_keys_to_env(keys_dict: dict):
    """Apply provided TradingAgents keys into os.environ if not already set.

    Use-case: paste keys into `TRADINGAGENTS_API_KEYS` above and the
    import/initialization below will pick them up.
    """
    for k, v in (keys_dict or {}).items():
        if not v:
            continue
        if os.getenv(k):
            # Do not overwrite existing environment variables
            continue
        os.environ[k] = v

# Apply any keys the user placed in the TRADINGAGENTS_API_KEYS dict.
_apply_tradingagents_keys_to_env(TRADINGAGENTS_API_KEYS)

# Default LLM provider and model — editable in-code per user preference.
# Default to the DeepSeek provider and official API.
LLM_PROVIDER = os.getenv("TRADINGAGENTS_LLM_PROVIDER", "deepseek")
# Default deep model (editable). Prefer official DeepSeek model names.
LLM_DEEP_MODEL = os.getenv("TRADINGAGENTS_DEEP_MODEL", "deepseek-v4-pro")

# Backwards-compatibility mapping: map legacy or user-friendly names to
# official DeepSeek model identifiers used by the API.
_MODEL_NAME_MAP = {
    "deepseek-r1-0528": "deepseek-v4-pro",
    "deepseek-r1": "deepseek-v4-pro",
    "deepseek-r1_0528": "deepseek-v4-pro",
}
_normalized = str(LLM_DEEP_MODEL or "").strip().lower()
if _normalized in _MODEL_NAME_MAP:
    LLM_DEEP_MODEL = _MODEL_NAME_MAP[_normalized]
LLM_QUICK_MODEL = os.getenv("TRADINGAGENTS_QUICK_MODEL", "gemini-2.5-flash-lite")
LLM_OPENAI_REASONING_EFFORT = os.getenv("TRADINGAGENTS_OPENAI_REASONING_EFFORT", "low")
LLM_ANTHROPIC_EFFORT = os.getenv("TRADINGAGENTS_ANTHROPIC_EFFORT", "low")
TRADINGAGENTS_BACKEND_URL = os.getenv("TRADINGAGENTS_BACKEND_URL", DEEPSEEK_OFFICIAL_API_URL)


def _resolve_google_thinking_level(model: str, requested_level: str) -> str:
    """Normalize Google thinking settings to match the selected model."""
    model_lower = (model or "").lower()
    level = (requested_level or "").strip().lower()

    if "gemini-3" in model_lower:
        # Gemini 3 Pro does not accept "minimal".
        if "pro" in model_lower and level == "minimal":
            return "low"
        return level or "low"

    if "thinking" in model_lower:
        # Gemini 2.5 thinking variants reject the zero-budget path.
        if level not in {"high", "dynamic"}:
            return "high"
        return level

    return level or "low"


LLM_GOOGLE_THINKING_LEVEL = _resolve_google_thinking_level(
    LLM_DEEP_MODEL,
    os.getenv("TRADINGAGENTS_GOOGLE_THINKING_LEVEL", "low"),
)  # Upgraded from minimal for better analysis depth


def _normalize_google_backend_url(url: str) -> str:
    """Keep provider host clean; Google client appends versioned path internally."""
    cleaned = (url or "").strip().rstrip("/")
    if cleaned.lower().endswith("/v1beta"):
        cleaned = cleaned[:-7]
    return cleaned


# Prefer an explicit DeepSeek URL when provided; otherwise normalize the existing URL.
if os.getenv("DEEPSEEK_API_URL"):
    TRADINGAGENTS_BACKEND_URL = _normalize_google_backend_url(os.getenv("DEEPSEEK_API_URL"))
else:
    # fallback to the configured backend URL, normalized
    TRADINGAGENTS_BACKEND_URL = _normalize_google_backend_url(TRADINGAGENTS_BACKEND_URL or DEEPSEEK_OFFICIAL_API_URL)


def _validate_and_propagate_backend_url(url: str) -> str:
    """Ensure the backend URL is a usable HTTP(S) URL and export env vars used by downstream clients.

    Returns a cleaned URL (no trailing slash) and sets `TRADINGAGENTS_BACKEND_URL` and
    `OPENAI_API_BASE` in the environment to avoid ambiguous empty-host errors.
    """
    candidate = (url or "").strip()
    if not candidate:
        candidate = DEEPSEEK_OFFICIAL_API_URL
    # If user provided a host without scheme, assume https
    if not candidate.startswith("http://") and not candidate.startswith("https://"):
        candidate = "https://" + candidate
    # remove trailing slash
    candidate = candidate.rstrip("/")

    # Propagate into environment for libraries that read OPENAI_API_BASE / TRADINGAGENTS_BACKEND_URL
    os.environ["TRADINGAGENTS_BACKEND_URL"] = candidate
    # Set OpenAI-compatible base variables without a versioned path (no /v1)
    openai_base = candidate
    # If candidate contains an API version path like /v1, strip it for OPENAI_API_BASE
    if openai_base.lower().endswith("/v1"):
        openai_base = openai_base[: -3]
    os.environ.setdefault("OPENAI_API_BASE", openai_base)
    os.environ.setdefault("OPENAI_API_BASE_URL", openai_base)
    # Also ensure DEEPSEEK_API_URL env is present for other codepaths
    os.environ.setdefault("DEEPSEEK_API_URL", candidate)
    return candidate


# Validate and ensure backend URL envs are present to avoid httpx connect errors when empty.
TRADINGAGENTS_BACKEND_URL = _validate_and_propagate_backend_url(TRADINGAGENTS_BACKEND_URL)

try:
    from tradingagents.default_config import DEFAULT_CONFIG
    TRADINGAGENTS_AVAILABLE = True
except Exception:
    DEFAULT_CONFIG = None
    TRADINGAGENTS_AVAILABLE = False


WORLDMONITOR_URL = "http://localhost:5173/api"
NYT_API_KEY = os.getenv("NYT_API_KEY", "")

SCALER_PATH = "0003_scalers_20/global_scaler.pkl"
SCHEMA_PATH = "0003_scalers_20/schema_cols.pkl"
PCA_PATH = "pca_model_live.pkl"
PCA_SCALER_PATH = "scaler_model_live.pkl"
PT_MODEL_PATH = "0004_saved_models_20/patchtst_live_20.pth"
PT_TARGET_STATS_PATH = "0004_saved_models_20/patchtst_target_stats.json"
XGB_MODEL_PATH = "0004_saved_models_20/xgboost_live_20.json"

PRICE_DIR  = "001_final_db_daily"
VECTOR_DIR = "0002_final_vectors"
NEWS_DIR   = "0001_ticker_news"
MACRO_FILE = "001_macro_data.csv"
OUTPUT_DIR = os.getenv("FORWARD_OUTPUT_DIR", "0007_forward_output/0007_1")
AI_IO_LOG_FILENAME = "Sentinel_v1_AI_IO_Log.csv"
RECOMMENDATIONS_HISTORY_FILENAME = "Sentinel_v1_Recommendations_History.csv"
POSITIONS_MONITOR_FILENAME = "Sentinel_v1_20day_Positions_Monitor.csv"
DOCUMENTED_20DAY_FILENAME = "Sentinel_v1_20day_Detailed_History.csv"
TRACKING_DAYS = 20

DEBUG_FORWARD = os.getenv("DEBUG_FORWARD", "0") == "1"
DEBUG_FORWARD_TICKERS = {
    t.strip().upper()
    for t in os.getenv("DEBUG_FORWARD_TICKERS", "AAPL,MSFT,XOM,SPGI").split(",")
    if t.strip()
}


VECTOR_DIM            = 28
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_TICKERS = """MMM,AOS,ABT,ABBV,ACN,ADBE,AMD,AES,AFL,A,APD,ABNB,AKAM,ALB,ARE,ALGN,ALLE,LNT,ALL,GOOGL,GOOG,MO,AMZN,AMCR,AEE,AEP,AXP,AIG,AMT,AWK,AMP,AME,AMGN,APH,ADI,AON,APA,APO,AAPL,AMAT,APP,APTV,ACGL,ADM,ARES,ANET,AJG,AIZ,T,ATO,ADSK,ADP,AZO,AVB,AVY,AXON,BKR,BALL,BAC,BAX,BDX,BRK.B,BBY,TECH,BIIB,BLK,BX,XYZ,BK,BA,BKNG,BSX,BMY,AVGO,BR,BRO,BF.B,BLDR,BG,BXP,CHRW,CDNS,CPT,CPB,COF,CAH,CCL,CARR,CVNA,CAT,CBOE,CBRE,CDW,COR,CNC,CNP,CF,CRL,SCHW,CHTR,CVX,CMG,CB,CHD,CIEN,CI,CINF,CTAS,CSCO,C,CFG,CLX,CME,CMS,KO,CTSH,COIN,CL,CMCSA,FIX,CAG,COP,ED,STZ,CEG,COO,CPRT,GLW,CPAY,CTVA,CSGP,COST,CTRA,CRH,CRWD,CCI,CSX,CMI,CVS,DHR,DRI,DDOG,DVA,DECK,DE,DELL,DAL,DVN,DXCM,FANG,DLR,DG,DLTR,D,DPZ,DASH,DOV,DOW,DHI,DTE,DUK,DD,ETN,EBAY,ECL,EIX,EW,EA,ELV,EME,EMR,ETR,EOG,EPAM,EQT,EFX,EQIX,EQR,ERIE,ESS,EL,EG,EVRG,ES,EXC,EXE,EXPE,EXPD,EXR,XOM,FFIV,FDS,FICO,FAST,FRT,FDX,FIS,FITB,FSLR,FE,FISV,F,FTNT,FTV,FOXA,FOX,BEN,FCX,GRMN,IT,GE,GEHC,GEV,GEN,GNRC,GD,GIS,GM,GPC,GILD,GPN,GL,GDDY,GS,HAL,HIG,HAS,HCA,DOC,HSIC,HSY,HPE,HLT,HOLX,HD,HON,HRL,HST,HWM,HPQ,HUBB,HUM,HBAN,HII,IBM,IEX,IDXX,ITW,INCY,IR,PODD,INTC,IBKR,ICE,IFF,IP,INTU,ISRG,IVZ,INVH,IQV,IRM,JBHT,JBL,JKHY,J,JNJ,JCI,JPM,KVUE,KDP,KEY,KEYS,KMB,KIM,KMI,KKR,KLAC,KHC,KR,LHX,LH,LRCX,LW,LVS,LDOS,LEN,LII,LLY,LIN,LYV,LMT,L,LOW,LULU,LYB,MTB,MPC,MAR,MRSH,MLM,MAS,MA,MTCH,MKC,MCD,MCK,MDT,MRK,META,MET,MTD,MGM,MCHP,MU,MSFT,MAA,MRNA,MOH,TAP,MDLZ,MPWR,MNST,MCO,MS,MOS,MSI,MSCI,NDAQ,NTAP,NFLX,NEM,NWSA,NWS,NEE,NKE,NI,NDSN,NSC,NTRS,NOC,NCLH,NRG,NUE,NVDA,NVR,NXPI,ORLY,OXY,ODFL,OMC,ON,OKE,ORCL,OTIS,PCAR,PKG,PLTR,PANW,PSKY,PH,PAYX,PAYC,PYPL,PNR,PEP,PFE,PCG,PM,PSX,PNW,PNC,POOL,PPG,PPL,PFG,PG,PGR,PLD,PRU,PEG,PTC,PSA,PHM,PWR,QCOM,DGX,Q,RL,RJF,RTX,O,REG,REGN,RF,RSG,RMD,RVTY,HOOD,ROK,ROL,ROP,ROST,RCL,SPGI,CRM,SNDK,SBAC,SLB,STX,SRE,NOW,SHW,SPG,SWKS,SJM,SW,SNA,SOLV,SO,LUV,SWK,SBUX,STT,STLD,STE,SYK,SMCI,SYF,SNPS,SYY,TMUS,TROW,TTWO,TPR,TRGP,TGT,TEL,TDY,TER,TSLA,TXN,TPL,TXT,TMO,TJX,TKO,TTD,TSCO,TT,TDG,TRV,TRMB,TFC,TYL,TSN,USB,UBER,UDR,ULTA,UNP,UAL,UPS,URI,UNH,UHS,VLO,VTR,VLTO,VRSN,VRSK,VZ,VRTX,VTRS,VICI,V,VST,VMC,WRB,GWW,WAB,WMT,DIS,WBD,WM,WAT,WEC,WFC,WELL,WST,WDC,WY,WSM,WMB,WTW,WDAY,WYNN,XEL,XYL,YUM,ZBRA,ZBH,ZTS"""
SP500_TICKERS = [t.strip().replace('.', '-') for t in RAW_TICKERS.replace('\n', '').split(',') if t.strip()]

# ==========================================
# NEURAL NETWORK ARCHITECTURE
# ==========================================
class PatchTST(nn.Module):
    def __init__(self, num_features, seq_len, patch_len=10, stride=10, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super(PatchTST, self).__init__()
        self.seq_len, self.patch_len, self.stride = seq_len, patch_len, stride
        self.num_patches = int((seq_len - patch_len) / stride) + 1
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True), num_layers=num_layers)
        self.flatten = nn.Flatten(start_dim=1)
        self.head = nn.Sequential(nn.Linear(num_features * self.num_patches * d_model, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 1))

    def forward(self, x):
        b, s, f = x.shape
        x = x.permute(0, 2, 1).reshape(b * f, s) 
        enc_in = self.patch_embedding(x.unfold(1, self.patch_len, self.stride)) + self.pos_embedding
        return self.head(self.flatten(self.encoder(enc_in).reshape(b, f, self.num_patches, -1)))

# ==========================================
# PIPELINE FUNCTIONS
# ==========================================
def get_live_macro_data():
    print("🌍 Fetching Live Macro Data...")
    end = datetime.now()
    start = end - timedelta(days=120) 
    
    spy = yf.download("SPY", start=start, end=end, progress=False)
    vix = yf.download("^VIX", start=start, end=end, progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = [c[0] for c in spy.columns]
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = [c[0] for c in vix.columns]
        
    df_macro = spy[['Open', 'High', 'Low', 'Close', 'Volume']].add_prefix('SPY_').join(
               vix[['Open', 'High', 'Low', 'Close']].add_prefix('VIX_'), how='outer')
    df_macro.index = pd.to_datetime(df_macro.index, utc=True).tz_convert(None)
    
    df_macro['FEDFUNDS'], df_macro['CPIAUCSL'], df_macro['M2SL'] = 5.3, 305.0, 20800.0 
    df_macro.ffill(inplace=True)
    return df_macro


def add_technical_features(df_price: pd.DataFrame) -> pd.DataFrame:
    """Add RSI/MACD features expected by the training schema."""
    out = df_price.copy()
    if 'close' not in out.columns:
        return out

    close = pd.to_numeric(out['close'], errors='coerce')

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14.0, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14.0, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # MACD(12,26,9)
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd - macd_signal

    out['RSI_14'] = rsi.fillna(50.0)
    out['MACD_12_26_9'] = macd.fillna(0.0)
    out['MACDh_12_26_9'] = macd_hist.fillna(0.0)
    out['MACDs_12_26_9'] = macd_signal.fillna(0.0)
    return out


def estimate_recent_return_pct(df_price: pd.DataFrame, horizon: int = 20) -> float | None:
    close_series = df_price.get('close')
    if close_series is None:
        return None
    close = pd.to_numeric(close_series, errors='coerce').dropna()
    if len(close) <= horizon:
        return None
    base = float(close.iloc[-(horizon + 1)])
    latest = float(close.iloc[-1])
    if not np.isfinite(base) or not np.isfinite(latest) or base == 0.0:
        return None
    return (latest / base - 1.0) * 100.0


def fetch_futubull_news(ticker: str) -> list:
    """Fetch news from FutuBull web terminal.

    Returns list of (title, timestamp) tuples.
    """
    if not FUTUBULL_NEWS_ENABLED or not BS4_AVAILABLE:
        return []
    
    news_items = []
    try:
        # Map common ticker formats to FutuBull URL format
        # FUTU -> FUTU-US, or handle other formats as needed
        fb_ticker = ticker.upper()
        if "-" not in fb_ticker:
            if fb_ticker in {"FUTU", "BABA", "NIO", "XPE", "JD", "BILI", "DIDI"}:
                fb_ticker = f"{fb_ticker}-US"
        
        url = f"https://www.futunn.com/stock/{fb_ticker}/news"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Try multiple selector strategies for robustness (FutuBull may change HTML structure)
        news_containers = soup.find_all("div", class_="news-item")
        if not news_containers:
            news_containers = soup.select(".news-list-item, .news-card, [class*='news'][class*='item']")
        
        for item in news_containers[:15]:
            try:
                title_el = item.find("p", class_="title") or item.select_one(".title, h3, h4")
                time_el = item.find("span", class_="time") or item.select_one(".time, .date, .publish-time")
                
                if title_el:
                    title = title_el.get_text(strip=True)
                    timestamp = time_el.get_text(strip=True) if time_el else None
                    
                    if title and len(title) > 10:
                        news_items.append((title, timestamp))
            except Exception:
                continue

        # Fallback: extract from embedded JSON blobs in script tags if DOM cards are dynamic.
        if not news_items:
            scripts = soup.find_all("script")
            title_patterns = [
                re.compile(r'"title"\s*:\s*"([^"\\]{12,}.*?)"'),
                re.compile(r'"newsTitle"\s*:\s*"([^"\\]{12,}.*?)"'),
            ]
            time_pattern = re.compile(r'"(?:publishTime|publishedAt|time|date)"\s*:\s*"([^"\\]+)"')
            for s in scripts:
                raw = s.string or s.get_text("", strip=False)
                if not raw or "title" not in raw:
                    continue
                collected = []
                for p in title_patterns:
                    collected.extend(p.findall(raw))
                if not collected:
                    continue
                times = time_pattern.findall(raw)
                for i, title in enumerate(collected[:20]):
                    clean_title = re.sub(r"\\\\u[0-9a-fA-F]{4}", "", title)
                    clean_title = clean_title.replace('\\"', '"').strip()
                    if len(clean_title) < 12:
                        continue
                    ts = times[i] if i < len(times) else None
                    news_items.append((clean_title, ts))
    except Exception:
        pass

    # De-duplicate in order.
    out = []
    seen = set()
    for title, ts in news_items:
        key = f"{title}||{ts}"
        if key in seen:
            continue
        seen.add(key)
        out.append((title, ts))
    return out[:20]


def fetch_yahoo_rss_news(ticker: str, max_items: int = 12) -> list:
    """Fallback news feed that is usually fresher than ticker.info snapshots."""
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        items = []
        for it in root.findall(".//item")[:max_items]:
            title = (it.findtext("title") or "").strip()
            pub = (it.findtext("pubDate") or "").strip()
            if title:
                items.append((title, pub if pub else None))
        return items
    except Exception:
        return []


def fetch_google_news_rss(ticker: str, max_items: int = 12) -> list:
    """Additional fallback news feed for fresher headlines by ticker."""
    try:
        q = f"{ticker} stock"
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        items = []
        for it in root.findall(".//item")[:max_items]:
            title = (it.findtext("title") or "").strip()
            pub = (it.findtext("pubDate") or "").strip()
            if title:
                items.append((title, pub if pub else None))
        return items
    except Exception:
        return []


def fetch_bing_news_rss(ticker: str, max_items: int = 12) -> list:
    """Fetch Bing News RSS headlines with short descriptions."""
    try:
        q = f"{ticker} stock"
        url = f"https://www.bing.com/news/search?q={requests.utils.quote(q)}&format=rss"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=6)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        items = []
        for it in root.findall(".//item")[:max_items]:
            title = (it.findtext("title") or "").strip()
            pub = (it.findtext("pubDate") or "").strip()
            desc_html = (it.findtext("description") or "").strip()
            snippet = ""
            if desc_html:
                if BS4_AVAILABLE:
                    snippet = BeautifulSoup(desc_html, "html.parser").get_text(" ", strip=True)
                else:
                    snippet = re.sub(r"<[^>]+>", " ", desc_html)
                    snippet = re.sub(r"\s+", " ", snippet).strip()

            text = title
            if snippet and snippet.lower() != title.lower():
                text = f"{title} - {snippet}"
            if text:
                items.append((text, pub if pub else None))
        return items
    except Exception:
        return []


def _market_session_now_et(now_et: datetime) -> str:
    """Return market session label: premarket, regular, postmarket, or closed."""
    t = now_et.time()
    if t >= datetime.strptime("04:00", "%H:%M").time() and t < datetime.strptime("09:30", "%H:%M").time():
        return "premarket"
    if t >= datetime.strptime("09:30", "%H:%M").time() and t < datetime.strptime("16:00", "%H:%M").time():
        return "regular"
    if t >= datetime.strptime("16:00", "%H:%M").time() and t <= datetime.strptime("20:00", "%H:%M").time():
        return "postmarket"
    return "closed"


def _filter_intraday_by_session(intraday: pd.DataFrame, session: str, tz_name: str = "America/New_York") -> pd.DataFrame:
    if intraday is None or intraday.empty:
        return intraday

    idx = intraday.index
    if getattr(idx, "tz", None) is None:
        idx_et = pd.to_datetime(idx, utc=True, errors="coerce").tz_convert(tz_name)
    else:
        idx_et = pd.to_datetime(idx, errors="coerce").tz_convert(tz_name)

    local_times = idx_et.time
    if session == "premarket":
        mask = [(datetime.strptime("04:00", "%H:%M").time() <= t < datetime.strptime("09:30", "%H:%M").time()) for t in local_times]
    elif session == "regular":
        mask = [(datetime.strptime("09:30", "%H:%M").time() <= t < datetime.strptime("16:00", "%H:%M").time()) for t in local_times]
    elif session == "postmarket":
        mask = [(datetime.strptime("16:00", "%H:%M").time() <= t <= datetime.strptime("20:00", "%H:%M").time()) for t in local_times]
    else:
        return intraday

    return intraday.loc[mask]


def get_latest_price_quote(ticker):
    """Return the freshest available quote as (price, volume, source).

    Source priority (session-aware):
    1) Yahoo 1m intraday filtered to current session (pre/regular/post)
    2) Yahoo quote metadata matching current session
    3) Yahoo fast_info fields
    """
    try:
        tk = yf.Ticker(ticker)
    except Exception:
        return np.nan, np.nan, "unavailable"

    now_et = datetime.now(ZoneInfo("America/New_York"))
    session = _market_session_now_et(now_et)

    # 1) 1m intraday with extended hours enabled, filtered by current session first.
    try:
        intraday = tk.history(period="2d", interval="1m", prepost=True, auto_adjust=False)
        if intraday is not None and not intraday.empty:
            close_col = "Close" if "Close" in intraday.columns else ("close" if "close" in intraday.columns else None)
            vol_col = "Volume" if "Volume" in intraday.columns else ("volume" if "volume" in intraday.columns else None)
            if close_col is not None:
                scoped = _filter_intraday_by_session(intraday, session)
                candidate = scoped if scoped is not None and not scoped.empty else intraday
                close_vals = pd.to_numeric(candidate[close_col], errors="coerce").dropna()
                if not close_vals.empty:
                    price = float(close_vals.iloc[-1])
                    if price > 0:
                        volume = np.nan
                        if vol_col is not None:
                            try:
                                volume = float(pd.to_numeric(candidate[vol_col], errors="coerce").fillna(0).iloc[-1])
                            except Exception:
                                volume = np.nan
                        return price, volume, f"yf_intraday_1m_{session}"
    except Exception:
        pass

    # 2) Quote metadata from info (often includes pre/post market prices)
    try:
        info = tk.info if hasattr(tk, "info") else {}
    except Exception:
        info = {}

    if isinstance(info, dict) and info:
        key_order = []
        if session == "premarket":
            key_order = [
                ("preMarketPrice", "yf_info_premarket"),
                ("regularMarketPrice", "yf_info_regular"),
                ("currentPrice", "yf_info_current"),
                ("postMarketPrice", "yf_info_postmarket"),
            ]
        elif session == "postmarket":
            key_order = [
                ("postMarketPrice", "yf_info_postmarket"),
                ("regularMarketPrice", "yf_info_regular"),
                ("currentPrice", "yf_info_current"),
                ("preMarketPrice", "yf_info_premarket"),
            ]
        else:
            key_order = [
                ("regularMarketPrice", "yf_info_regular"),
                ("currentPrice", "yf_info_current"),
                ("preMarketPrice", "yf_info_premarket"),
                ("postMarketPrice", "yf_info_postmarket"),
            ]

        for key, src in key_order:
            val = info.get(key)
            if val is not None and pd.notna(val):
                try:
                    fval = float(val)
                    if fval > 0:
                        vol = info.get("regularMarketVolume", np.nan)
                        try:
                            vol = float(vol) if vol is not None and pd.notna(vol) else np.nan
                        except Exception:
                            vol = np.nan
                        return fval, vol, f"{src}_{session}"
                except Exception:
                    pass

    # 3) fast_info fallback
    try:
        fi = getattr(tk, "fast_info", None)
        if fi:
            for key, src in [
                ("lastPrice", "yf_fast_last"),
                ("regularMarketPrice", "yf_fast_regular"),
                ("previousClose", "yf_fast_prev_close"),
            ]:
                value = fi.get(key, None) if hasattr(fi, "get") else None
                if value is not None and pd.notna(value):
                    fval = float(value)
                    if fval > 0:
                        return fval, np.nan, f"{src}_{session}"
    except Exception:
        pass

    return np.nan, np.nan, "unavailable"

def fetch_ticker_news(ticker, alpaca_client, use_alpaca=True, alpaca_semaphore=None):
    """Phase 1: Fetch raw news corpus for a ticker (Network Bound)."""
    corpus_parts = []
    timed_parts = []

    def _to_ts(value):
        try:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            ts = pd.to_datetime(value, utc=True, errors='coerce')
            if pd.isna(ts):
                return None
            return ts.to_pydatetime()
        except Exception:
            return None

    def _push(text, ts=None):
        if not text:
            return
        t = _to_ts(ts) or datetime.now(timezone.utc)
        timed_parts.append((t, str(text).strip()))

    try:
        # 1. World Monitor
        try:
            wm_url = f"{WORLDMONITOR_URL}/finance/intel?symbol={ticker}"
            wm_data = requests.get(wm_url, timeout=3).json()
            if isinstance(wm_data, dict) and 'brief' in wm_data:
                _push(f"WorldMonitor: {wm_data['brief']}")
            elif isinstance(wm_data, list):
                for item in wm_data[:5]:
                    _push(
                        f"WM: {item.get('title','')} - {item.get('summary','')}",
                        item.get('published_at') or item.get('date') or item.get('time'),
                    )
        except: pass

        # 2. Yahoo Finance
        try:
            yf_news = yf.Ticker(ticker).news
            if yf_news:
                for article in yf_news[:20]:
                    # yfinance now nests article text under article["content"].
                    if isinstance(article, dict):
                        content = article.get("content", {})
                        if isinstance(content, dict):
                            title = (content.get("title") or article.get("title") or "").strip()
                            summary = (content.get("summary") or content.get("description") or article.get("summary") or "").strip()
                            ts = (
                                content.get("pubDate")
                                or content.get("displayTime")
                                or article.get("providerPublishTime")
                                or article.get("published")
                            )
                        else:
                            title = (article.get("title") or "").strip()
                            summary = (article.get("summary") or "").strip()
                            ts = article.get("providerPublishTime") or article.get("published")

                        text = " - ".join([x for x in [title, summary] if x])
                        if text:
                            _push(f"YF: {text}", ts)
        except: pass

        # 3. Alpaca 
        try:
            if use_alpaca and alpaca_client is not None:
                def _fetch_alpaca():
                    start_news = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                    return alpaca_client.get_news(ticker, start=start_news, limit=10)

                if alpaca_semaphore is not None:
                    with alpaca_semaphore:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as alp_ex:
                            future = alp_ex.submit(_fetch_alpaca)
                            alpaca_news = future.result(timeout=5)
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as alp_ex:
                        future = alp_ex.submit(_fetch_alpaca)
                        alpaca_news = future.result(timeout=5)

                for article in alpaca_news:
                    headline = (getattr(article, "headline", "") or "").strip()
                    summary = (getattr(article, "summary", "") or "").strip()
                    text = " - ".join([x for x in [headline, summary] if x])
                    if text:
                        _push(
                            f"Alpaca: {text}",
                            getattr(article, "updated_at", None) or getattr(article, "created_at", None),
                        )
        except: pass

        # 4. New York Times
        try:
            if NYT_API_KEY:
                # Top 3 most recent articles regarding the ticker
                nyt_url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q={ticker}&sort=newest&api-key={NYT_API_KEY}"
                res = requests.get(nyt_url, timeout=4).json()
                if "response" in res and "docs" in res["response"]:
                    for doc in res["response"]["docs"][:3]:
                        headline = doc.get("headline", {}).get("main", "")
                        lead = doc.get("lead_paragraph", "")
                        _push(f"NYT: {headline} - {lead}", doc.get("pub_date"))
        except: pass

        # 5. FutuBull (good for Chinese/fintech stocks)
        try:
            fb_news = fetch_futubull_news(ticker)
            for title, timestamp in fb_news:
                _push(f"FutuBull: {title}", timestamp)
        except: pass

        # 6. Yahoo RSS fallback (stable public feed; useful when page scraping yields nothing)
        try:
            rss_news = fetch_yahoo_rss_news(ticker)
            for title, timestamp in rss_news:
                _push(f"YahooRSS: {title}", timestamp)
        except: pass

        # 7. Google News RSS fallback (often very fresh intraday headlines)
        try:
            gnews = fetch_google_news_rss(ticker)
            for title, timestamp in gnews:
                _push(f"GoogleNews: {title}", timestamp)
        except: pass

        # 8. Bing News RSS fallback (adds short snippets, not only headlines)
        try:
            bnews = fetch_bing_news_rss(ticker)
            for title, timestamp in bnews:
                _push(f"BingNews: {title}", timestamp)
        except: pass

        # Keep only the freshest items first.
        if timed_parts:
            timed_parts.sort(key=lambda x: x[0], reverse=True)
            corpus_parts = [txt for _, txt in timed_parts[:20]]

        # Final sanitize: drop placeholder/empty fragments and deduplicate.
        cleaned_parts = []
        seen = set()
        for part in corpus_parts:
            p = str(part).strip()
            if not p:
                continue
            # Remove known low-information placeholders from older parsers.
            low_info = p.replace("YF:", "").replace("Alpaca:", "").replace("NYT:", "").replace("WM:", "").replace("WorldMonitor:", "")
            low_info = low_info.replace("-", "").replace("|", "").strip()
            if len(low_info) < 5:
                continue
            if p not in seen:
                seen.add(p)
                cleaned_parts.append(p)

        corpus = " | ".join(cleaned_parts)
        return ticker, corpus
    except:
        return ticker, ""

def batch_finbert_vectors(ticker_news_list, model, tokenizer, pca, scaler, batch_size=32):
    """Phase 2: Batch process corpuses through FinBERT & Extract Softmax Probs."""
    all_vecs = {}
    valid_items = [(t, c) for t, c in ticker_news_list if len(c.strip()) > 20]
    
    print(f"🚀 Batching FinBERT for {len(valid_items)} corpuses (Batch Size: {batch_size})...")
    total_batches = max((len(valid_items) + batch_size - 1) // batch_size, 1)
    for batch_idx, i in enumerate(range(0, len(valid_items), batch_size), start=1):
        batch = valid_items[i : i + batch_size]
        tickers = [b[0] for b in batch]
        texts = [b[1][:8000] for b in batch]
        
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            cls_embs = outputs.hidden_states[-1][:, 0, :].to(torch.float32).cpu().numpy()
            probs = torch.nn.functional.softmax(outputs.logits.to(torch.float32), dim=-1).cpu().numpy()
            pca_vecs = pca.transform(scaler.transform(cls_embs))
            for t, vec, prob in zip(tickers, pca_vecs, probs): 
                all_vecs[t] = {'pca': vec, 'sents': prob}
        print(f"🧠 FinBERT batch {batch_idx}/{total_batches} done | tickers {len(all_vecs)}/{len(valid_items)}", end='\r')
    print()
    return all_vecs


def _openbb_result_to_df(result):
    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return result
    if hasattr(result, "to_df"):
        try:
            return result.to_df()
        except Exception:
            pass
    if hasattr(result, "to_dataframe"):
        try:
            return result.to_dataframe()
        except Exception:
            pass
    if hasattr(result, "results"):
        rows = result.results
        if isinstance(rows, pd.DataFrame):
            return rows
        if isinstance(rows, list):
            try:
                return pd.DataFrame(rows)
            except Exception:
                return None
    return None


def fetch_openbb_context(ticker: str) -> str:
    """Fetch optional OpenBB profile and recent price context for a ticker."""
    if not OPENBB_AVAILABLE:
        return ""

    parts = []

    try:
        profile = obb.equity.profile(symbol=ticker, provider="yfinance")
        profile_df = _openbb_result_to_df(profile)
        if profile_df is not None and not profile_df.empty:
            profile_row = profile_df.iloc[0].to_dict()
            sector = str(profile_row.get("sector", "")).strip()
            industry = str(profile_row.get("industry", "")).strip()
            description = (
                profile_row.get("longBusinessSummary")
                or profile_row.get("long_business_summary")
                or profile_row.get("description")
                or profile_row.get("summary")
                or ""
            )
            website = str(profile_row.get("website", "")).strip()
            info_bits = []
            if sector:
                info_bits.append(f"sector={sector}")
            if industry:
                info_bits.append(f"industry={industry}")
            if website:
                info_bits.append(f"website={website}")
            if description:
                info_bits.append(f"summary={str(description)[:8000]}")
            if info_bits:
                parts.append("OpenBB Profile: " + "; ".join(info_bits))
    except Exception:
        pass

    try:
        price_hist = obb.equity.price.historical(symbol=ticker, provider="yfinance")
        price_df = _openbb_result_to_df(price_hist)
        if price_df is not None and not price_df.empty:
            price_df = price_df.copy()
            if "date" in price_df.columns:
                price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
                price_df = price_df.sort_values("date")
            elif price_df.index.name is not None:
                price_df = price_df.sort_index()

            close_col = next((c for c in ["close", "Close"] if c in price_df.columns), None)
            if close_col:
                close = pd.to_numeric(price_df[close_col], errors="coerce").dropna()
                if len(close) >= 5:
                    last_close = float(close.iloc[-1])
                    ret_5d = float((close.iloc[-1] / close.iloc[-5] - 1.0) * 100.0) if len(close) >= 5 else np.nan
                    ret_20d = float((close.iloc[-1] / close.iloc[-20] - 1.0) * 100.0) if len(close) >= 20 else np.nan
                    vol_20d = float(close.pct_change().tail(20).std() * np.sqrt(252) * 100.0) if len(close) >= 20 else np.nan
                    parts.append(
                        "OpenBB Price Context: "
                        f"last_close={last_close:.2f}; "
                        f"ret_5d={ret_5d:.2f}%; "
                        f"ret_20d={ret_20d:.2f}%; "
                        f"vol_20d={vol_20d:.2f}%"
                    )
    except Exception:
        pass

    return " | ".join(parts)


def upsert_ticker_news_row(ticker: str, date_str: str, corpus: str) -> None:
    """Upsert one ticker news row for the given date immediately."""
    if not ticker:
        return
    news_path = os.path.join(NEWS_DIR, f"{ticker}_news.csv")
    news_row = {
        "source_dataset": "Live_v1",
        "date": date_str,
        "title": f"v1 Scan {ticker}",
        "content": corpus,
        "original_url": "live",
    }
    df_new_news = pd.DataFrame([news_row])
    if os.path.exists(news_path):
        df_old_news = pd.read_csv(news_path)
        if 'date' in df_old_news.columns:
            df_old_news['date'] = df_old_news['date'].astype(str)
            df_old_news = df_old_news[df_old_news['date'] != date_str]
        pd.concat([df_old_news, df_new_news], ignore_index=True).to_csv(news_path, index=False)
    else:
        os.makedirs(NEWS_DIR, exist_ok=True)
        df_new_news.to_csv(news_path, index=False)


def enhance_corpus_with_openbb(ticker: str, corpus: str) -> str:
    openbb_context = fetch_openbb_context(ticker)
    if openbb_context:
        corpus = " | ".join([part for part in [corpus, openbb_context] if part])
    return corpus


def _format_detailed_agent_prompt(row: pd.Series, date_str: str) -> str:
    """Generate an independent analysis prompt that does not expose model outputs."""
    ticker = row.get("Ticker", "n/a")
    price = float(row.get("Current_Price", np.nan)) if pd.notna(row.get("Current_Price", np.nan)) else None
    recent_20d = float(row.get("Recent_20D_Return_%", np.nan)) if pd.notna(row.get("Recent_20D_Return_%", np.nan)) else None
    turnover = float(row.get("Dollar_Turnover", np.nan)) if pd.notna(row.get("Dollar_Turnover", np.nan)) else None
    finbert_pos = float(row.get("FinBERT_Pos", np.nan)) if pd.notna(row.get("FinBERT_Pos", np.nan)) else None
    finbert_neg = float(row.get("FinBERT_Neg", np.nan)) if pd.notna(row.get("FinBERT_Neg", np.nan)) else None
    xgb_score = float(row.get("XGB_Rank_Score", np.nan)) if pd.notna(row.get("XGB_Rank_Score", np.nan)) else None
    pred_return = float(row.get("Pred_Return_%", np.nan)) if pd.notna(row.get("Pred_Return_%", np.nan)) else None
    news = _shorten_text(str(row.get("News_Corpus", "")).strip(), 8000)
    sentiment = "bullish" if (finbert_pos is not None and finbert_neg is not None and finbert_pos > finbert_neg) else "bearish" if finbert_neg is not None and finbert_neg > (finbert_pos or 0) else "neutral"

    parts = [
        "You are an independent equity analyst providing a rigorous second opinion.",
        "Analyze the raw data provided and form your own conviction. Do NOT echo, restate verbatim, or reference any model.",
        f"Ticker: {ticker}",
        f"Analysis Date: {date_str}",
    ]

    if price is not None:
        parts.append(f"Current Price: ${price:.2f}")
    if turnover is not None:
        parts.append(f"Daily Dollar Turnover: ${turnover:,.0f}")
    if recent_20d is not None:
        parts.append(f"Recent 20D Price Change: {recent_20d:+.2f}%")
    if pred_return is not None:
        parts.append(f"Model Forecast (forward return): {pred_return:+.2f}%")
    if xgb_score is not None:
        parts.append(f"Relative Strategy Score: {xgb_score:.2f}")
    if finbert_pos is not None and finbert_neg is not None:
        parts.append(f"News Sentiment: {sentiment.upper()} (positive: {finbert_pos:.1%}, negative: {finbert_neg:.1%})")
    if news:
        parts.append(f"News / Context Summary: {news}")
    else:
        parts.append("News / Context Summary: No recent news available")

    parts.extend([
        "",
        "=== REQUIRED ANALYSIS FRAMEWORK ===",
        "",
        "Technical Analysis: Assess price momentum, recent trading patterns, and key support/resistance levels.",
        "Fundamental Assessment: Evaluate business quality, growth prospects, and valuation relative to peers.",
        "Risk Assessment: Identify key downside risks, macro headwinds, and execution challenges.",
        "Price Target: Provide a specific 12-month price target with reasoning (e.g., 'Target: $185, based on 15x forward earnings').",
        "Catalysts: List 2-3 key events or milestones that could impact the stock in the next 6-12 months.",
        "Investment Thesis: Concise summary of your bull/bear/neutral case in 2-3 sentences.",
        "Timeline: When do you expect your thesis to play out? (3mo, 6mo, 12mo, or longer)",
        "",
        "Final Recommendation: BUY, SELL, or HOLD (choose one based on risk-reward and conviction)",
        "Confidence: Rate your conviction 0.0 to 1.0 (0.9+ = very high, 0.7-0.9 = high, 0.5-0.7 = moderate, <0.5 = low)",
        "",
        "=== CRITICAL RULES ===",
        "1. Make an independent call. Do not defer to the forecast data provided.",
        "2. Provide concrete, specific reasoning (not generic platitudes).",
        "3. If evidence is split, HOLD is the honest call—do not force a BUY or SELL.",
        "4. Confidence should reflect your actual conviction, not just a random number.",
        "5. ANTI-GLAZING: Challenge the bullish case. Identify specific downsides and execution risks even if you lean BUY.",
        "6. Do not agree with sentiment merely because it exists. Only cite news/sentiment if it directly supports your thesis.",
        "7. If the stock looks mediocre, say HOLD or SELL. Do not manufacture reasons to buy.",
    ])

    if LLM_PROVIDER.lower() == "deepseek":
        parts.extend([
            "",
            "=== DEEPSEEK RESPONSE STYLE ===",
            "Write a more detailed answer than usual.",
            "Use clear section headings and concrete evidence.",
            "Aim for at least 250 words unless the data are truly sparse.",
            "When possible, include 4 or more bullets under Technical Analysis, Fundamental Assessment, Risk Assessment, and Catalysts.",
            "Be specific about what would change your recommendation.",
        ])

    return "\n".join(parts)


def _detailed_parse_agent_response(response_text: str) -> dict:
    """Parse detailed agent response and extract all analysis sections with improved confidence calibration."""
    sections = {
        "Technical_Analysis": "",
        "Fundamental_Assessment": "",
        "Risk_Assessment": "",
        "Price_Target": "",
        "Catalysts": "",
        "Investment_Thesis": "",
        "Timeline": "",
        "Recommendation": "unknown",
        "Confidence": np.nan,
    }
    
    import re
    response_text_orig = str(response_text)
    text = response_text_orig.lower()

    # Extract recommendation from the explicit final recommendation section first.
    # Use DOTALL to match across newlines, look for the LAST occurrence (most recent recommendation)
    rec_matches = list(re.finditer(r"final\s+recommendation\s*[:\-]*\s*(?:\*\*|\*|#)?\s*([a-z]+)\s*(?:\*\*|\*)?", response_text_orig, re.IGNORECASE | re.DOTALL))
    if rec_matches:
        # Use the LAST match (most recent recommendation)
        rec_value = rec_matches[-1].group(1).strip().lower()
        if rec_value in {"buy", "sell", "hold", "long", "short", "neutral", "wait"}:
            sections["Recommendation"] = rec_value
    
    # If no explicit match, try "Recommendation:" field (common after "Final Recommendation")
    if sections.get("Recommendation", "unknown") == "unknown":
        rec_match = re.search(r"(?:final\s+)?recommendation\s*[:\-]*\s*(?:\*\*|\*|#)?\s*([a-z]+)\s*(?:\*\*|\*)?", response_text_orig, re.IGNORECASE | re.DOTALL)
        if rec_match:
            rec_value = rec_match.group(1).strip().lower()
            if rec_value in {"buy", "sell", "hold", "long", "short", "neutral", "wait"}:
                sections["Recommendation"] = rec_value
    
    # Fallback: scan for explicit patterns in sections (Technical/Fundamental/Risk/Thesis), prioritize HOLD over BUY to avoid over-buying
    if sections.get("Recommendation", "unknown") == "unknown":
        # Check for "hold" or "neutral" FIRST (these are honest recommendations)
        if "hold" in text or "neutral" in text:
            sections["Recommendation"] = "hold"
        elif "sell" in text or "short" in text:
            sections["Recommendation"] = "sell"
        elif "buy" in text or "long" in text:
            sections["Recommendation"] = "buy"

    # Extract confidence from explicit section, with better parsing for varied formats.
    conf_match = re.search(r"confidence[^0-9]{0,40}([0-9]*\.?[0-9]+)", response_text_orig, re.IGNORECASE | re.DOTALL)
    if not conf_match:
        conf_match = re.search(r"confidence\s*[:\-]*\s*([0-9]*\.?[0-9]+)", response_text_orig, re.IGNORECASE)
    if conf_match:
        try:
            conf_val = float(conf_match.group(1))
            # If confidence is given as percentage (e.g., "75"), normalize to 0-1 scale
            if conf_val > 1.0:
                conf_val = conf_val / 100.0
            sections["Confidence"] = max(0.0, min(1.0, conf_val))  # Clamp to [0, 1]
        except Exception:
            pass
    
    # If no explicit confidence given, calibrate based on analysis depth and conviction language
    if pd.isna(sections["Confidence"]):
        analysis_text = (sections.get("Technical_Analysis", "") + " " + 
                        sections.get("Fundamental_Assessment", "") + " " +
                        sections.get("Investment_Thesis", "")).lower()
        high_conviction_words = ["strong", "clear", "compelling", "convincing", "robust", "definitive", "high conviction"]
        low_conviction_words = ["weak", "uncertain", "mixed", "unclear", "marginal", "risk", "concern"]
        
        high_count = sum(1 for word in high_conviction_words if word in analysis_text)
        low_count = sum(1 for word in low_conviction_words if word in analysis_text)
        
        if sections["Recommendation"] == "hold" or "hold" in analysis_text:
            base_conf = 0.55  # Hold is honest, moderate confidence
        elif high_count > low_count:
            base_conf = 0.75  # High conviction language detected
        elif low_count > high_count:
            base_conf = 0.45  # Low conviction language detected
        else:
            base_conf = 0.60  # Balanced analysis
        
        # Adjust based on analysis length (longer, more detailed analysis = higher confidence)
        if len(analysis_text) > 300:
            base_conf = min(0.95, base_conf + 0.15)
        elif len(analysis_text) < 100:
            base_conf = max(0.35, base_conf - 0.15)
        
        sections["Confidence"] = max(0.2, min(0.95, base_conf))
    
    # Extract sections by parsing structured response.
    # Each section stops at the next known heading so content does not bleed into later sections.
    section_names = [
        "Technical Analysis",
        "Fundamental Assessment",
        "Risk Assessment",
        "Price Target",
        "Catalysts",
        "Investment Thesis",
        "Timeline",
        "Final Recommendation",
        "Confidence",
    ]

    def _extract_section_text(text_value: str, section_name: str) -> str:
        next_names = [name for name in section_names if name != section_name]
        next_pattern = "|".join(re.escape(name) for name in next_names)
        pattern = rf"(?:^|\n)\s*(?:\*\*|##?\s*)?{re.escape(section_name)}(?:\*\*)?\s*[:\-]?\s*(.*?)(?=\n\s*(?:\*\*|##?\s*)?(?:{next_pattern})(?:\*\*)?\s*[:\-]?\s*|\Z)"
        match = re.search(pattern, text_value, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        return match.group(1).strip() if match else ""

    section_map = {
        "Technical_Analysis": "Technical Analysis",
        "Fundamental_Assessment": "Fundamental Assessment",
        "Risk_Assessment": "Risk Assessment",
        "Price_Target": "Price Target",
        "Catalysts": "Catalysts",
        "Investment_Thesis": "Investment Thesis",
        "Timeline": "Timeline",
    }

    for key, section_name in section_map.items():
        section_content = _extract_section_text(response_text_orig, section_name)
        if section_content:
            # Preserve up to 8000 chars for better detail capture
            sections[key] = _shorten_text(section_content, 8000)
    
    return sections


def _safe_extract_agent_fields(decision):
    """Extract stable fields from a TradingAgents decision object/string (legacy)."""
    action = "unknown"
    confidence = np.nan
    summary = ""

    try:
        if isinstance(decision, dict):
            action = str(decision.get("action", decision.get("decision", "unknown")))
            confidence_raw = decision.get("confidence", decision.get("conviction_score", np.nan))
            confidence = float(confidence_raw) if confidence_raw is not None else np.nan
            summary = str(decision.get("reasoning", decision.get("summary", "")))
        else:
            decision_text = str(decision)
            summary = decision_text[:8000]
            lower = decision_text.lower()
            if "buy" in lower or "long" in lower:
                action = "buy"
            elif "sell" in lower or "short" in lower:
                action = "sell"
            elif "hold" in lower or "neutral" in lower:
                action = "hold"
    except Exception:
        pass

    return action, confidence, summary


def _format_agent_reason(row: pd.Series) -> str:
    """Create a concise narrative reason for the daily report."""
    action = str(row.get("Agent_Action", "n/a")).strip().lower()
    summary = str(row.get("Agent_Summary", "")).strip()
    status = str(row.get("Agent_Status", "n/a")).strip()
    confidence = row.get("Agent_Confidence", np.nan)

    if summary:
        return summary

    if status != "ok":
        return f"Agent did not complete successfully ({status})."

    if action in {"buy", "long"}:
        base = "Agent leaned bullish and passed the cross-check."
    elif action in {"sell", "short"}:
        base = "Agent leaned bearish and failed the cross-check."
    elif action in {"hold", "neutral"}:
        base = "Agent was neutral and did not produce a trade signal."
    else:
        base = f"Agent returned action='{action}'."

    if pd.notna(confidence):
        base += f" Confidence={float(confidence):.3f}."
    return base


def _format_agent_crosscheck_verdict(row: pd.Series) -> str:
    """Convert the agent's raw action into a buy-focused cross-check verdict."""
    action = str(row.get("Agent_Recommendation", row.get("Agent_Action", "n/a"))).strip().lower()
    status = str(row.get("Agent_Status", "n/a")).strip().lower()

    if status != "ok":
        return "ERROR"
    if action in {"buy", "long"}:
        return "BUY CONFIRMED"
    if action in {"sell", "short"}:
        return "AVOID / SELL"
    if action in {"hold", "neutral"}:
        return "NO BUY"
    if action in {"skipped", "not_available", "init_failed"}:
        return action.replace("_", " ").upper()
    return str(action).replace("_", " ").upper()


def _shorten_text(value, max_len=8000):
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return ""
    text = str(value).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _is_transient_tradingagents_error(error: Exception) -> bool:
    """Return True for temporary network/server failures from TradingAgents."""
    message = str(error).lower()
    error_name = type(error).__name__.lower()
    transient_markers = (
        "server disconnected without sending a response",
        "connection aborted",
        "connection reset",
        "read timeout",
        "request timed out",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "502",
        "503",
        "504",
    )
    return error_name in {"apitimeouterror", "readtimeouterror", "timeouterror"} or any(
        marker in message for marker in transient_markers
    )


def _is_retryable_tradingagents_refusal(error: Exception) -> bool:
    """Return True for refusal-style bad requests that deserve one re-ask."""
    message = str(error).lower()
    refusal_markers = (
        "text_au",
        "很抱歉",
        "无法提供相应的信息",
        "cannot provide",
        "unable to provide",
        "refusal",
    )
    return any(marker in message for marker in refusal_markers)


def _is_token_exhausted_tradingagents_error(error: Exception) -> bool:
    """Return True when the backend has no available model tokens."""
    message = str(error).lower()
    token_markers = (
        "没有可用token",
        "no available token",
        "resource has been exhausted",
        "quota",
        "resource_exhausted",
    )
    return any(marker in message for marker in token_markers)


def _extract_llm_text(response) -> str:
    """Return the best-effort text payload from a chat model response."""
    text = getattr(response, "content", None)
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        pieces = []
        for item in text:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    pieces.append(str(item["text"]))
            elif hasattr(item, "type") and getattr(item, "type", None) == "text" and getattr(item, "text", None):
                pieces.append(str(getattr(item, "text")))
        if pieces:
            return "\n".join(pieces).strip()

    raw_text = getattr(response, "text", None)
    if raw_text:
        return str(raw_text)

    return str(response)


def _run_tradingagents_with_retry(llm, prompt: str, max_attempts: int = 2, backoff_seconds: float = 3.0):
    """Run a single TauricResearch Google-client cross-check with transient retries."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.invoke(prompt)
            text = _extract_llm_text(response)
            print(f"✅ TradingAgents LLM call succeeded (attempt {attempt})")
            return text
        except Exception as error:
            last_error = error
            if _is_token_exhausted_tradingagents_error(error):
                raise RuntimeError(
                    "TradingAgents backend token pool is exhausted (no available token). "
                    "This is separate from your account credit balance."
                ) from error
            is_transient = _is_transient_tradingagents_error(error)
            print(f"⚠️  TradingAgents LLM attempt {attempt}: {type(error).__name__}: {str(error)[:100]} [Transient: {is_transient}]")

            retry_prompt = prompt
            if _is_retryable_tradingagents_refusal(error):
                retry_prompt = prompt + (
                    "\n\nRETRY INSTRUCTIONS: Please answer the equity-analysis request directly, in detail, "
                    "using the required sections. Do not refuse. Keep the response structured, specific, and complete."
                )
            
            if attempt < max_attempts and (is_transient or retry_prompt != prompt):
                wait_time = backoff_seconds * attempt
                print(f"   → Retrying in {wait_time}s...")
                time.sleep(wait_time)
                prompt = retry_prompt
                continue
            raise
    raise last_error


def _generate_agent_reasoning(row: pd.Series, action: str) -> str:
    """Generate detailed reasoning for agent's buy/sell decision based on available data."""
    ticker = row.get('Ticker', 'Stock')
    pred_ret = float(row.get('Pred_Return_%', 0))
    recent_return = float(row.get('Recent_20D_Return_%', 0))
    xgb_score = float(row.get('XGB_Rank_Score', 0))
    sent_pos = float(row.get('FinBERT_Pos', 0))
    sent_neg = float(row.get('FinBERT_Neg', 0))
    market_sent_pos = float(row.get('Market_Sent_Pos', 0))
    market_sent_neg = float(row.get('Market_Sent_Neg', 0))
    used_fallback = bool(row.get('Used_Fallback', False))
    
    # Build reasoning narrative based on decision and factors
    factors = []
    
    # Technical momentum analysis
    if recent_return > 5:
        factors.append(f"strong 20D momentum (+{recent_return:.1f}%)")
    elif recent_return > 2:
        factors.append(f"positive momentum (+{recent_return:.1f}%)")
    elif recent_return < -5:
        factors.append(f"weak momentum ({recent_return:.1f}%)")
    else:
        factors.append(f"neutral momentum ({recent_return:.1f}%)")
    
    # Model prediction
    if abs(pred_ret) > 10:
        factors.append(f"significant predicted alpha ({pred_ret:+.1f}%)")
    elif abs(pred_ret) > 3:
        factors.append(f"material prediction ({pred_ret:+.1f}%)")
    
    # Sentiment analysis
    if sent_pos > 0.6 and sent_neg < 0.2:
        factors.append("strong positive sentiment")
    elif sent_pos > 0.4:
        factors.append("positive sentiment bias")
    elif sent_neg > 0.6:
        factors.append("strong negative sentiment")
    elif sent_neg > 0.4:
        factors.append("negative sentiment bias")
    else:
        factors.append("mixed sentiment")
    
    # Market sentiment
    if market_sent_pos > market_sent_neg:
        factors.append("favorable broad market sentiment")
    else:
        factors.append("cautious broad market sentiment")
    
    # Relative ranking
    if xgb_score > 0.05:
        factors.append(f"strong relative ranking ({xgb_score:.3f})")
    elif xgb_score > -0.05:
        factors.append("neutral ranking")
    else:
        factors.append(f"weak relative ranking ({xgb_score:.3f})")
    
    # Build action-specific message
    if action.lower() in {"buy", "long"}:
        base = f"BUY {ticker}: "
        qualifiers = [f for f in factors if any(x in f.lower() for x in ["positive", "strong", "material"])]
        if not qualifiers:
            qualifiers = factors[:3]
        reasoning = base + " | ".join(qualifiers[:3]) + ". "
        if used_fallback:
            reasoning += "Model applied fallback blending for risk mitigation. "
        reasoning += f"Expected 20D alpha: {pred_ret:+.2f}%."
    elif action.lower() in {"sell", "short"}:
        base = f"SELL/AVOID {ticker}: "
        qualifiers = [f for f in factors if any(x in f.lower() for x in ["weak", "negative", "caution"])]
        if not qualifiers:
            qualifiers = factors[:3]
        reasoning = base + " | ".join(qualifiers[:3]) + ". "
        if used_fallback:
            reasoning += "Model applied fallback blending. "
        reasoning += f"Expected 20D alpha: {pred_ret:+.2f}%."
    else:
        base = f"HOLD {ticker}: "
        reasoning = base + " | ".join(factors[:2]) + ". Mixed signal - awaiting clarity."
    
    return reasoning


def _format_ai_input_summary(row: pd.Series, date_str: str) -> str:
    """Summarize the inputs used to produce the AI prediction and cross-check."""
    parts = [
        f"date={date_str}",
        f"ticker={row.get('Ticker', 'n/a')}",
        f"current_price={float(row.get('Current_Price', np.nan)):.2f}" if pd.notna(row.get("Current_Price", np.nan)) else "current_price=n/a",
        f"recent_20d_return={float(row.get('Recent_20D_Return_%', np.nan)):.2f}%" if pd.notna(row.get("Recent_20D_Return_%", np.nan)) else "recent_20d_return=n/a",
        f"pred_ret_raw={float(row.get('Pred_Return_Raw', np.nan)):.6f}" if pd.notna(row.get("Pred_Return_Raw", np.nan)) else "pred_ret_raw=n/a",
        f"pred_ret_model={float(row.get('Pred_Return_Model', np.nan)):.6f}" if pd.notna(row.get("Pred_Return_Model", np.nan)) else "pred_ret_model=n/a",
        f"pred_ret_final={float(row.get('Pred_Return_%', np.nan)):.2f}%" if pd.notna(row.get("Pred_Return_%", np.nan)) else "pred_ret_final=n/a",
        f"xgb_rank_score={float(row.get('XGB_Rank_Score', np.nan)):.3f}" if pd.notna(row.get("XGB_Rank_Score", np.nan)) else "xgb_rank_score=n/a",
        f"used_fallback={bool(row.get('Used_Fallback', False))}",
        f"finbert_sentiment=pos:{float(row.get('FinBERT_Pos', np.nan)):.3f},neg:{float(row.get('FinBERT_Neg', np.nan)):.3f},neu:{float(row.get('FinBERT_Neu', np.nan)):.3f}" if pd.notna(row.get("FinBERT_Pos", np.nan)) else "finbert_sentiment=n/a",
        f"market_sentiment=pos:{float(row.get('Market_Sent_Pos', np.nan)):.3f},neg:{float(row.get('Market_Sent_Neg', np.nan)):.3f},neu:{float(row.get('Market_Sent_Neu', np.nan)):.3f}" if pd.notna(row.get("Market_Sent_Pos", np.nan)) else "market_sentiment=n/a",
        f"news_excerpt={_shorten_text(row.get('News_Corpus', ''), 8000)}" if str(row.get("News_Corpus", "")).strip() else "news_excerpt=n/a",
    ]
    return " | ".join(parts)


def _format_ai_output_summary(row: pd.Series) -> str:
    """Summarize the outputs generated by the prediction model and TradingAgents."""
    parts = [
        f"pred_ret_final={float(row.get('Pred_Return_%', np.nan)):.2f}%" if pd.notna(row.get("Pred_Return_%", np.nan)) else "pred_ret_final=n/a",
        f"pred_ret_model={float(row.get('Pred_Return_Model', np.nan)):.6f}" if pd.notna(row.get("Pred_Return_Model", np.nan)) else "pred_ret_model=n/a",
        f"xgb_rank_score={float(row.get('XGB_Rank_Score', np.nan)):.3f}" if pd.notna(row.get("XGB_Rank_Score", np.nan)) else "xgb_rank_score=n/a",
        f"agent_action={row.get('Agent_Action', 'n/a')}",
        f"agent_status={row.get('Agent_Status', 'n/a')}",
        f"agent_pass={row.get('Agent_Pass', False)}",
        f"agent_reason={_shorten_text(_format_agent_reason(row), 8000)}",
    ]
    return " | ".join(parts)


def _format_agent_input_summary(row: pd.Series, date_str: str) -> str:
    """Generate a clean, decision-focused input summary for the TradingAgents agent.
    
    Unlike the model input summary, this focuses on what matters for trading decisions:
    - Ticker and current price
    - Recent momentum context
    - Sentiment and news context
    
    Excludes low-level model internals, forecast outputs, and technical details.
    """
    ticker = row.get("Ticker", "n/a")
    price = float(row.get("Current_Price", np.nan)) if pd.notna(row.get("Current_Price", np.nan)) else None
    recent_20d = float(row.get("Recent_20D_Return_%", np.nan)) if pd.notna(row.get("Recent_20D_Return_%", np.nan)) else None
    finbert_pos = float(row.get("FinBERT_Pos", np.nan)) if pd.notna(row.get("FinBERT_Pos", np.nan)) else None
    finbert_neg = float(row.get("FinBERT_Neg", np.nan)) if pd.notna(row.get("FinBERT_Neg", np.nan)) else None
    news = str(row.get("News_Corpus", "")).strip()
    
    parts = [
        f"Ticker: {ticker}",
        f"Date: {date_str}",
    ]
    
    if price:
        parts.append(f"Current Price: ${price:.2f}")
    
    if recent_20d is not None:
        parts.append(f"Recent 20D Return (Momentum): {recent_20d:+.2f}%")
    
    if finbert_pos is not None and finbert_neg is not None:
        sentiment_tone = "bullish" if finbert_pos > finbert_neg else "bearish" if finbert_neg > finbert_pos else "neutral"
        parts.append(f"Sentiment Tone: {sentiment_tone} (positive:{finbert_pos:.1%}, negative:{finbert_neg:.1%})")
    
    if news:
        news_short = _shorten_text(news, 8000)
        parts.append(f"Context: {news_short}")
    
    return " | ".join(parts)


def run_tradingagents_crosscheck(candidates_df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Run TradingAgents on top-N candidates and return detailed agent analysis.

    This is optional and never fails the pipeline.
    """
    columns = ["Ticker", "Agent_Input_Summary", "Agent_Output_Raw", "Agent_Recommendation", 
               "Agent_Confidence", "Agent_Technical_Analysis", "Agent_Fundamental_Assessment",
               "Agent_Risk_Assessment", "Agent_Price_Target", "Agent_Catalysts", 
               "Agent_Investment_Thesis", "Agent_Timeline", "Agent_Status", "Agent_Pass"]
    
    if candidates_df.empty:
        return pd.DataFrame(columns=columns)

    subset = candidates_df.head(TOP_N_CROSSCHECK).copy()
    subset["Agent_Input_Summary"] = subset.apply(lambda row: _format_agent_input_summary(row, date_str), axis=1)

    if not TRADINGAGENTS_CROSSCHECK_ENABLED:
        subset["Agent_Recommendation"] = "skipped"
        subset["Agent_Confidence"] = np.nan
        subset["Agent_Technical_Analysis"] = "TradingAgents cross-check disabled by config."
        subset["Agent_Fundamental_Assessment"] = ""
        subset["Agent_Risk_Assessment"] = ""
        subset["Agent_Price_Target"] = ""
        subset["Agent_Catalysts"] = ""
        subset["Agent_Investment_Thesis"] = ""
        subset["Agent_Timeline"] = ""
        subset["Agent_Status"] = "disabled"
        subset["Agent_Pass"] = False
        subset["Agent_Output_Raw"] = "TradingAgents cross-check disabled by config."
        return subset[columns]

    try:
        if not TRADINGAGENTS_AVAILABLE:
            raise RuntimeError("tradingagents package is not available")
        # Select API key env var name based on provider
        provider_key_map = {"google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
        api_key_env = provider_key_map.get(LLM_PROVIDER.lower(), "OPENAI_API_KEY")
        if not os.getenv(api_key_env, "").strip():
            raise RuntimeError(f"{api_key_env} is not configured")
        # Build kwargs dynamically to avoid sending None values
        client_kwargs = dict(
            provider=LLM_PROVIDER,
            model=LLM_DEEP_MODEL,
            base_url=TRADINGAGENTS_BACKEND_URL,
            api_key=os.getenv(api_key_env),
            timeout=90,
            max_retries=1,
        )
        if LLM_PROVIDER.lower() == "google":
            client_kwargs["thinking_level"] = LLM_GOOGLE_THINKING_LEVEL
        llm_client = create_llm_client(**client_kwargs)
        llm = llm_client.get_llm()
    except Exception as e:
        subset["Agent_Recommendation"] = "init_failed"
        subset["Agent_Confidence"] = np.nan
        subset["Agent_Technical_Analysis"] = f"TradingAgents init failed: {e}"
        subset["Agent_Fundamental_Assessment"] = ""
        subset["Agent_Risk_Assessment"] = ""
        subset["Agent_Price_Target"] = ""
        subset["Agent_Catalysts"] = ""
        subset["Agent_Investment_Thesis"] = ""
        subset["Agent_Timeline"] = ""
        subset["Agent_Status"] = "init_failed"
        subset["Agent_Pass"] = False
        subset["Agent_Output_Raw"] = f"TradingAgents init failed: {e}"
        return subset[columns]

    rows = []
    for idx, (_, row) in enumerate(subset.iterrows()):
        ticker = str(row["Ticker"])
        try:
            # Rate limiting: Space out calls to avoid hitting the free tier limit.
            if idx > 0:
                time.sleep(12)  # Wait 12 seconds between calls (max ~5 calls/min, safely under 6 RPM)
            prompt = _format_detailed_agent_prompt(row, date_str)
            response_text = _run_tradingagents_with_retry(llm, prompt)
            parsed = _detailed_parse_agent_response(response_text)
            action = str(parsed.get("Recommendation", "unknown")).lower()
            confidence = parsed.get("Confidence", np.nan)
            summary = parsed.get("Investment_Thesis", "") or parsed.get("Technical_Analysis", "")
            # Keep a longer analysis excerpt for storage; raw response is stored in full below.
            analysis_text = _shorten_text(summary, 8000)
            if not analysis_text:
                analysis_text = _shorten_text(response_text, 8000)

            rows.append({
                "Ticker": ticker,
                "Agent_Input_Summary": _format_agent_input_summary(row, date_str),
                # Store full raw agent response to CSV so nothing is silently truncated.
                "Agent_Output_Raw": response_text,
                "Agent_Recommendation": action,
                "Agent_Confidence": confidence,
                "Agent_Technical_Analysis": analysis_text,
                # Store parsed subfields with generous limits to preserve content while keeping CSV manageable.
                "Agent_Fundamental_Assessment": _shorten_text(parsed.get("Fundamental_Assessment", ""), 8000),
                "Agent_Risk_Assessment": _shorten_text(parsed.get("Risk_Assessment", ""), 8000),
                "Agent_Price_Target": _shorten_text(parsed.get("Price_Target", ""), 8000),
                "Agent_Catalysts": _shorten_text(parsed.get("Catalysts", ""), 8000),
                "Agent_Investment_Thesis": _shorten_text(parsed.get("Investment_Thesis", ""), 8000),
                "Agent_Timeline": _shorten_text(parsed.get("Timeline", ""), 8000),
                "Agent_Status": "ok",
                "Agent_Pass": str(action).lower() in {"buy", "long"},
            })
        except Exception as e:
            rows.append({
                "Ticker": ticker,
                "Agent_Input_Summary": _format_agent_input_summary(row, date_str),
                "Agent_Output_Raw": f"TradingAgents run failed: {e}",
                "Agent_Recommendation": "error",
                "Agent_Confidence": np.nan,
                "Agent_Technical_Analysis": f"Error: {e}",
                "Agent_Fundamental_Assessment": "",
                "Agent_Risk_Assessment": "",
                "Agent_Price_Target": "",
                "Agent_Catalysts": "",
                "Agent_Investment_Thesis": "",
                "Agent_Timeline": "",
                "Agent_Status": "error",
                "Agent_Pass": False,
            })

    return pd.DataFrame(rows)[columns]


def _load_recent_recommendations(days_back: int = 20) -> pd.DataFrame:
    """Load recommendations from the history file that are within days_back of today."""
    history_path = os.path.join(OUTPUT_DIR, RECOMMENDATIONS_HISTORY_FILENAME)
    if not os.path.exists(history_path):
        return pd.DataFrame()
    
    df = pd.read_csv(history_path)
    if df.empty or 'Date_Recommended' not in df.columns:
        return pd.DataFrame()
    
    df['Date_Recommended'] = pd.to_datetime(df['Date_Recommended'], errors='coerce')
    cutoff_date = pd.to_datetime(pd.Timestamp.today().strftime("%Y-%m-%d")) - pd.Timedelta(days=days_back)
    recent = df[df['Date_Recommended'] >= cutoff_date].copy()
    return recent.sort_values('Date_Recommended', ascending=False)


def _save_recommendation_to_history(ticker: str, date_str: str, recommendation: str, confidence: float, 
                                     price: float, thesis: str, all_fields: dict):
    """Append a new recommendation to the history file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history_path = os.path.join(OUTPUT_DIR, RECOMMENDATIONS_HISTORY_FILENAME)
    
    new_row = {
        "Date_Recommended": date_str,
        "Ticker": ticker,
        "Entry_Recommendation": recommendation,
        "Entry_Confidence": confidence,
        "Entry_Price": price,
        "Entry_Thesis": thesis,
        "Entry_Raw_Data": str(all_fields)[:500],
    }
    
    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        if not existing[(existing['Ticker'] == ticker) & (existing['Date_Recommended'] == date_str)].empty:
            return
        df_new = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_new = pd.DataFrame([new_row])
    
    df_new.to_csv(history_path, index=False)


def _analyze_position_exit(entry_ticker: str, entry_date: str, entry_recommendation: str, entry_price: float,
                           entry_confidence: float, current_row: pd.Series, current_agent_verdict: str,
                           current_confidence: float, current_price: float) -> dict:
    """Determine if a past recommendation should be exited based on current conditions."""
    days_held = (pd.to_datetime(pd.Timestamp.today().strftime("%Y-%m-%d")) - pd.to_datetime(entry_date)).days
    price_change_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else np.nan
    
    exit_signal = "HOLD"
    reason = ""
    
    if entry_recommendation.lower() == "buy":
        if current_agent_verdict.lower() == "sell":
            exit_signal = "EXIT NOW"
            reason = f"Recommendation reversed: was BUY, now SELL (confidence {current_confidence:.2f})"
        elif current_agent_verdict.lower() == "hold" and current_confidence < (entry_confidence - 0.2):
            exit_signal = "TRIM / REDUCE"
            reason = f"Weakened conviction: was {entry_confidence:.2f}, now {current_confidence:.2f}; take some profits"
        elif days_held >= TRACKING_DAYS:
            if price_change_pct and price_change_pct > 3:
                exit_signal = "TAKE PROFITS"
                reason = f"20-day thesis window closing; up {price_change_pct:.1f}%, consider exit"
            elif price_change_pct and price_change_pct < -5:
                exit_signal = "EXIT / STOP LOSS"
                reason = f"20-day thesis window closing; down {price_change_pct:.1f}%, cut loss"
            else:
                exit_signal = "EXIT (thesis complete)"
                reason = f"20-day forecast window expired; reassess"
        else:
            # Make reason explicit: show agent verdict separately from portfolio action
            reason = f"Agent verdict: {current_agent_verdict.upper()} ({current_confidence:.2f}); Portfolio action: HOLD, {days_held}d held"
    
    elif entry_recommendation.lower() == "sell":
        if current_agent_verdict.lower() == "buy":
            exit_signal = "REVERSE TO BUY"
            reason = f"Recommendation reversed: was SELL, now BUY"
        else:
            exit_signal = "MAINTAIN SELL"
            reason = f"Agent verdict: {current_agent_verdict.upper()}; Portfolio action: MAINTAIN SELL"
    
    else:
        if current_agent_verdict.lower() == "buy":
            exit_signal = "UPGRADE TO BUY"
            reason = f"Agent verdict: BUY ({current_confidence:.2f}); Portfolio action: UPGRADE TO BUY"
        else:
            exit_signal = "HOLD"
            reason = f"Agent verdict: {current_agent_verdict.upper()} ({current_confidence:.2f}); Portfolio action: HOLD"
    
    return {
        "Days_Held": days_held,
        "Price_Change_%": price_change_pct,
        "Exit_Signal": exit_signal,
        "Reason": reason,
        "Current_Verdict": current_agent_verdict,
        "Current_Confidence": current_confidence,
    }


def _monitor_20day_positions(df_preds: pd.DataFrame, date_str: str, agent_check_df: pd.DataFrame = None) -> tuple:
    """Analyze all recommendations from past 20 days and generate hold/exit signals."""
    recent_recs = _load_recent_recommendations(days_back=TRACKING_DAYS)
    # If a Top-3 tracker exists, only monitor tickers listed there (user request)
    tracker_path = os.path.join(OUTPUT_DIR, "Sentinel_v1_Top_3_Tracker.csv")
    if os.path.exists(tracker_path):
        try:
            tracker_df = pd.read_csv(tracker_path)
            if not tracker_df.empty and "Ticker" in tracker_df.columns:
                tickers_to_monitor = set(tracker_df["Ticker"].astype(str).str.strip().str.upper().tolist())
                recent_recs = recent_recs[recent_recs["Ticker"].astype(str).str.upper().isin(tickers_to_monitor)]
        except Exception:
            # On any error reading tracker, fall back to full recent_recs
            pass
    if recent_recs.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    monitor_rows = []
    detailed_rows = []
    
    for _, entry_row in recent_recs.iterrows():
        ticker = entry_row["Ticker"]
        entry_date = str(entry_row["Date_Recommended"])
        entry_rec = entry_row["Entry_Recommendation"]
        entry_conf = float(entry_row.get("Entry_Confidence", 0.5))
        entry_price = float(entry_row.get("Entry_Price", np.nan))
        entry_thesis = entry_row.get("Entry_Thesis", "")
        
        current_matches = df_preds[df_preds['Ticker'] == ticker]
        if current_matches.empty:
            continue
        
        current_row = current_matches.iloc[0]
        current_price = float(current_row.get("Current_Price", np.nan))
        
        current_verdict = "hold"
        current_conf = 0.5
        if agent_check_df is not None and not agent_check_df.empty:
            agent_match = agent_check_df[agent_check_df['Ticker'] == ticker]
            if not agent_match.empty:
                current_verdict = str(agent_match.iloc[0].get('Agent_Recommendation', 'hold')).lower()
                current_conf = float(agent_match.iloc[0].get('Agent_Confidence', 0.5))
        
        exit_analysis = _analyze_position_exit(
            ticker, entry_date, entry_rec, entry_price, entry_conf,
            current_row, current_verdict, current_conf, current_price
        )
        
        monitor_rows.append({
            "Ticker": ticker,
            "Date_Recommended": entry_date,
            "Entry_Recommendation": entry_rec,
            "Entry_Confidence": entry_conf,
            "Entry_Price": entry_price,
            "Current_Price": current_price,
            "Days_Held": exit_analysis["Days_Held"],
            "Price_Change_%": exit_analysis["Price_Change_%"],
            "Exit_Signal": exit_analysis["Exit_Signal"],
            "Current_Verdict": exit_analysis["Current_Verdict"],
            "Current_Confidence": exit_analysis["Current_Confidence"],
            "Reason": exit_analysis["Reason"],
        })
        
        detailed_rows.append({
            "Date_Recommended": entry_date,
            "Today": date_str,
            "Ticker": ticker,
            "Entry_Recommendation": entry_rec,
            "Entry_Confidence": entry_conf,
            "Entry_Price": entry_price,
            "Entry_Thesis": entry_thesis,
            "Current_Price": current_price,
            "Current_Pred_Return_%": current_row.get("Pred_Return_%", np.nan),
            "Current_XGB_Score": current_row.get("XGB_Rank_Score", np.nan),
            "Current_20D_Return_%": current_row.get("Recent_20D_Return_%", np.nan),
            "Days_Held": exit_analysis["Days_Held"],
            "Price_Change_%": exit_analysis["Price_Change_%"],
            "Exit_Signal": exit_analysis["Exit_Signal"],
            "Reason": exit_analysis["Reason"],
        })
    
    monitor_df = pd.DataFrame(monitor_rows)
    detailed_df = pd.DataFrame(detailed_rows)
    
    if not monitor_df.empty:
        monitor_path = os.path.join(OUTPUT_DIR, POSITIONS_MONITOR_FILENAME)
        monitor_df.to_csv(monitor_path, index=False)
        print(f"📊 20-day position monitor saved: {monitor_path}")
        
        detailed_path = os.path.join(OUTPUT_DIR, DOCUMENTED_20DAY_FILENAME)
        detailed_df.to_csv(detailed_path, index=False)
        print(f"📋 20-day detailed history saved: {detailed_path}")
    
    return monitor_df, detailed_df


def _run_csv_only_agent_analysis(input_csv_path: str, output_csv_path: str, date_str: str, top_n: int = 3):
    """Run only the TradingAgents cross-check from a CSV input and write CSV output.

    Expected minimum column:
    - Ticker

    Recommended columns (for best analysis quality):
    - Current_Price, Current_Volume or Dollar_Turnover, Recent_20D_Return_%,
      Pred_Return_%, XGB_Rank_Score, FinBERT_Pos, FinBERT_Neg, News_Corpus
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")
    if "Ticker" not in df.columns:
        raise ValueError("Input CSV must contain a 'Ticker' column.")

    # Ensure required columns exist for prompt generation and cross-check metadata.
    required_defaults = {
        "Current_Price": np.nan,
        "Current_Volume": np.nan,
        "Dollar_Turnover": np.nan,
        "Recent_20D_Return_%": np.nan,
        "Pred_Return_%": np.nan,
        "XGB_Rank_Score": np.nan,
        "FinBERT_Pos": np.nan,
        "FinBERT_Neg": np.nan,
        "News_Corpus": "",
    }
    for col, default_val in required_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # Backfill turnover from price * volume if possible.
    if "Dollar_Turnover" in df.columns and "Current_Price" in df.columns and "Current_Volume" in df.columns:
        missing_turnover = df["Dollar_Turnover"].isna()
        df.loc[missing_turnover, "Dollar_Turnover"] = (
            pd.to_numeric(df.loc[missing_turnover, "Current_Price"], errors="coerce")
            * pd.to_numeric(df.loc[missing_turnover, "Current_Volume"], errors="coerce")
        )

    # Keep liquid names only where turnover is known; otherwise keep row (do not hard fail).
    liquid_df = df.copy()
    if "Dollar_Turnover" in liquid_df.columns:
        known_turnover = liquid_df["Dollar_Turnover"].notna()
        liquid_filtered = liquid_df[known_turnover & (liquid_df["Dollar_Turnover"] >= MIN_TURNOVER_DOLLARS)]
        if not liquid_filtered.empty:
            liquid_df = liquid_filtered

    # Rank by model columns when present; else preserve file order.
    sort_cols = [c for c in ["XGB_Rank_Score", "Pred_Return_%"] if c in liquid_df.columns]
    if sort_cols:
        liquid_df = liquid_df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last")

    candidates = liquid_df.head(max(1, int(top_n))).copy()
    if candidates.empty:
        raise ValueError("No rows available for agent analysis after preprocessing.")

    print(f"📥 CSV mode input rows: {len(df)} | candidates for agent analysis: {len(candidates)}")
    agent_df = run_tradingagents_crosscheck(candidates, date_str)

    # Include key source columns for easy traceability in output.
    keep_cols = [
        "Ticker",
        "Current_Price",
        "Current_Volume",
        "Dollar_Turnover",
        "Recent_20D_Return_%",
        "Pred_Return_%",
        "XGB_Rank_Score",
        "FinBERT_Pos",
        "FinBERT_Neg",
        "News_Corpus",
    ]
    base_cols = [c for c in keep_cols if c in candidates.columns]
    out_df = candidates[base_cols].merge(agent_df, on="Ticker", how="left")

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    print(f"✅ CSV-only agent analysis saved: {output_csv_path}")
    print(out_df[["Ticker", "Agent_Status", "Agent_Recommendation", "Agent_Confidence"]].to_string(index=False))

# ==========================================
# MASTER REPORT GENERATOR
# ==========================================
def generate_daily_report():
    print(f"\n{'='*50}\n🚀 SENTINEL AI: DAILY ADVISORY REPORT (NO-DEEPSEEK)\n{'='*50}")
    
    try:
        alpaca = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
    except Exception as e:
        exit(f"Alpaca Error: {e}. Check API keys.")
        
    print("🌍 Checking external API endpoints...")
    try:
        # Check if the World Monitor local server is alive
        requests.get(WORLDMONITOR_URL.replace("/api", "/"), timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"⚠️  WARNING: World Monitor is OFFLINE at {WORLDMONITOR_URL}.")
        print(f"⚠️  News inference will silently skip World Monitor and rely only on Yahoo Finance / Alpaca.")
    
    print("🧠 Loading Neural Networks & Scalers...")
    finbert_tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    finbert_dtype = torch.float16 if DEVICE.type == 'cuda' else torch.float32
    finbert_mod = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', torch_dtype=finbert_dtype).to(DEVICE).eval()
    
    pca = joblib.load(PCA_PATH)
    pca_scaler = joblib.load(PCA_SCALER_PATH)
    global_scaler = joblib.load(SCALER_PATH)
    schema_cols = joblib.load(SCHEMA_PATH)
    if os.path.exists(PT_TARGET_STATS_PATH):
        with open(PT_TARGET_STATS_PATH, "r", encoding="utf-8") as f:
            pt_target_stats = json.load(f)
        pt_target_mean = float(pt_target_stats.get("mean", 0.0))
        pt_target_std = float(pt_target_stats.get("std", 1.0)) or 1.0
        print(f"📐 PatchTST target stats loaded: mean={pt_target_mean:.6f}, std={pt_target_std:.6f}")
    else:
        pt_target_mean = 0.0
        pt_target_std = 1.0
        print("📐 PatchTST target stats not found; using raw model outputs.")
    
    pt_model = PatchTST(num_features=len(schema_cols), seq_len=60).to(DEVICE)
    pt_state = torch.load(PT_MODEL_PATH, map_location=DEVICE, weights_only=True)
    if list(pt_state.keys())[0].startswith('module.'): pt_state = {k[7:]: v for k, v in pt_state.items()}
    pt_model.load_state_dict(pt_state)
    pt_model.eval()
    
    xgb_model = xgb.Booster()
    xgb_model.load_model(XGB_MODEL_PATH)
    
    df_macro = get_live_macro_data()
    
    # --- UPDATE MACRO DATA FILE ---
    if not df_macro.empty:
        latest_macro = df_macro.iloc[-1:]
        latest_macro.index.name = "Date"
        if os.path.exists(MACRO_FILE):
             df_old_macro = pd.read_csv(MACRO_FILE, index_col='Date')
             if latest_macro.index[0].strftime("%Y-%m-%d") not in [d.split(' ')[0] for d in df_old_macro.index]:
                 pd.concat([df_old_macro, latest_macro]).to_csv(MACRO_FILE)
        else:
            latest_macro.to_csv(MACRO_FILE)

    # --- BATCH DOWNLOAD PRICES (6 months = ~125 trading days) ---
    print(f"📥 Batch Downloading Prices for {len(SP500_TICKERS)} tickers...")
    all_prices = yf.download(SP500_TICKERS, period="6mo", threads=True, progress=False, group_by='ticker')

    if OPENBB_AVAILABLE:
        print("🧩 OpenBB is available and will be used to enrich ticker context.")
    else:
        print("🧩 OpenBB is not installed; continuing without OpenBB enrichment.")

    # Ensure output folder exists for this run
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use US Eastern Time (UTC-4 during DST) for all date logging
    us_now = datetime.now(timezone(timedelta(hours=-4)))
    today_str = us_now.strftime("%Y-%m-%d")
    date_str = today_str  # Keep consistent for the report and log
    
    # --- PHASE 1: PARALLEL NEWS FETCHING ---
    scrape_targets = SP500_TICKERS + ["SPY"] if "SPY" not in SP500_TICKERS else SP500_TICKERS
    print(f"🌐 Fetching News ({len(scrape_targets)} items, High-Concurrency)...")
    ticker_news_list = []
    if ALPACA_NEWS_ENABLED:
        if ALPACA_MAX_NEWS_TICKERS and ALPACA_MAX_NEWS_TICKERS > 0:
            alpaca_tickers = set(scrape_targets[:ALPACA_MAX_NEWS_TICKERS])
        else:
            alpaca_tickers = set(scrape_targets)
    else:
        alpaca_tickers = set()
    alpaca_semaphore = threading.Semaphore(ALPACA_MAX_CONCURRENT_REQUESTS)
    news_saved = 0
    news_nonempty = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = {
            executor.submit(
                fetch_ticker_news,
                t,
                alpaca,
                t in alpaca_tickers,
                alpaca_semaphore,
            ): t
            for t in scrape_targets
        }
        for future in concurrent.futures.as_completed(futures):
            ticker, corpus = future.result()
            corpus = enhance_corpus_with_openbb(ticker, corpus)
            ticker_news_list.append((ticker, corpus))

            # Persist fetched corpus immediately so users can verify downloads while job is still running.
            upsert_ticker_news_row(ticker, date_str, corpus)
            news_saved += 1
            if corpus and len(corpus.strip()) > 20:
                news_nonempty += 1

            print(
                f"📡 News Collected: {len(ticker_news_list)}/{len(scrape_targets)} | Saved: {news_saved} | Non-empty: {news_nonempty}",
                end='\r'
            )

    # --- PHASE 2: BATCHED FINBERT ---
    finbert_vectors = batch_finbert_vectors(ticker_news_list, finbert_mod, finbert_tok, pca, pca_scaler)
    ticker_news_dict = dict(ticker_news_list)
    
    # Store Macro (SPY proxy) to ^GSPC
    macro_data = finbert_vectors.get("SPY", {'pca': np.zeros(VECTOR_DIM), 'sents': np.zeros(3)})
    macro_vec = macro_data['pca']
    macro_sents = macro_data['sents']
    
    # Write ^GSPC macro vectors to ensure consistency with backtesting
    macro_vec_path = os.path.join(VECTOR_DIR, "^GSPC_vec.csv")
    macro_csv_dict = {"date": today_str}
    for i, val in enumerate(macro_vec): macro_csv_dict[f"dim_{i+1}"] = val
    macro_csv_dict['sent_pos'] = macro_sents[0]
    macro_csv_dict['sent_neg'] = macro_sents[1]
    macro_csv_dict['sent_neu'] = macro_sents[2]
    df_new_macro = pd.DataFrame([macro_csv_dict])
    if os.path.exists(macro_vec_path):
        df_old_macro = pd.read_csv(macro_vec_path)
        if today_str not in df_old_macro['date'].values: pd.concat([df_old_macro, df_new_macro]).to_csv(macro_vec_path, index=False)
    else: 
        os.makedirs(VECTOR_DIR, exist_ok=True)
        df_new_macro.to_csv(macro_vec_path, index=False)

    def process_ticker_report(ticker, df_macro, global_scaler, schema_cols, pt_model, xgb_model, macro_vec, macro_sents):
        try:
            if ticker not in finbert_vectors: return None
            finbert_data = finbert_vectors[ticker]
            finbert_vec = finbert_data['pca']
            sents = finbert_data['sents'] # [sent_pos, sent_neg, sent_neu]
            corpus = ticker_news_dict.get(ticker, "")

            try:
                df_price = all_prices[ticker].dropna(how='all')
            except (KeyError, ValueError, AttributeError): return None
            latest_quote_source = "daily_close"
            # Overlay with freshest available quote (pre/post preferred) for latest-row price.
            try:
                latest_price, latest_volume, latest_quote_source = get_latest_price_quote(ticker)
                if pd.notna(latest_price) and float(latest_price) > 0 and not df_price.empty:
                    if isinstance(df_price.columns, pd.MultiIndex):
                        df_price.columns = [c[0] if isinstance(c, tuple) and len(c) > 0 else c for c in df_price.columns]
                    df_cols = {c.lower(): c for c in df_price.columns}
                    df_close_col = df_cols.get('close')
                    df_vol_col = df_cols.get('volume')
                    if df_close_col:
                        df_price.at[df_price.index[-1], df_close_col] = float(latest_price)
                    if df_vol_col and pd.notna(latest_volume):
                        df_price.at[df_price.index[-1], df_vol_col] = float(latest_volume)
            except Exception:
                latest_quote_source = "daily_close"
            if df_price.empty or len(df_price) < 62: return None
            
            if isinstance(df_price.columns, pd.MultiIndex): df_price.columns = [c[0] for c in df_price.columns]
            df_price.columns = [c.lower() for c in df_price.columns]
            if 'open' in df_price.columns and 'close' in df_price.columns and 'adj close' in df_price.columns:
                df_price['adj open'] = (df_price['open'] / df_price['close']) * df_price['adj close']
            df_price.index = pd.to_datetime(df_price.index, utc=True).tz_convert(None)
            df_price = add_technical_features(df_price)
            
            df = df_price.join(df_macro, how='left').ffill()
            for i in range(VECTOR_DIM): df[f'dim_{i+1}'] = finbert_vec[i]
            df['sent_pos'] = sents[0]
            df['sent_neg'] = sents[1]
            df['sent_neu'] = sents[2]
            
            for i in range(VECTOR_DIM): df[f'market_dim_{i+1}'] = macro_vec[i]
            df['market_sent_pos'] = macro_sents[0]
            df['market_sent_neg'] = macro_sents[1]
            df['market_sent_neu'] = macro_sents[2]
            
            num_cols = df.select_dtypes(include=[np.number]).columns
            cols_to_pct = [c for c in num_cols if any(x in c.lower() for x in ["close", "open", "high", "low", "cpi", "m2", "volume"])]
            cols_to_diff = [c for c in num_cols if any(x in c.lower() for x in ["fed", "vix", "rsi", "macd"])]
            df[cols_to_pct] = df[cols_to_pct].pct_change().fillna(0.0)
            df[cols_to_diff] = df[cols_to_diff].diff().fillna(0.0)
            
            df_aligned = df.reindex(columns=schema_cols, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            recent_60 = df_aligned.values[-60:]
            if len(recent_60) != 60: return None
            
            data_scaled = global_scaler.transform(recent_60)
            x_3d = torch.tensor(np.array([data_scaled]), dtype=torch.float32).to(DEVICE)
            feats = np.concatenate([data_scaled[-1], data_scaled.mean(axis=0), data_scaled.std(axis=0)])
            x_1d = xgb.DMatrix(np.array([feats], dtype=np.float32))

            if DEBUG_FORWARD and ticker.upper() in DEBUG_FORWARD_TICKERS:
                raw_std = np.std(recent_60, axis=0)
                scaled_std = np.std(data_scaled, axis=0)
                print(
                    f"\n[DEBUG] {ticker}"
                    f" | raw_last60_mean={float(np.mean(recent_60)):.6f}"
                    f" raw_last60_std={float(np.std(recent_60)):.6f}"
                    f" raw_nonzero_var={int((raw_std > 1e-12).sum())}"
                    f" | scaled_mean={float(np.mean(data_scaled)):.6f}"
                    f" scaled_std={float(np.std(data_scaled)):.6f}"
                    f" scaled_nonzero_var={int((scaled_std > 1e-12).sum())}"
                    f" | feats_mean={float(np.mean(feats)):.6f}"
                    f" feats_std={float(np.std(feats)):.6f}"
                )
            
            with torch.no_grad():
                pred_ret_raw = pt_model(x_3d).cpu().numpy().flatten()[0]
                pred_ret_model = pred_ret_raw * pt_target_std + pt_target_mean
                pred_rank = xgb_model.predict(x_1d)[0]

            recent_20d_return = estimate_recent_return_pct(df_price, horizon=20)
            pred_ret = pred_ret_model
            used_fallback = False
            if recent_20d_return is not None:
                # If the model collapses near its center, blend in recent momentum so the
                # live report still shows a ticker-specific return estimate.
                if not np.isfinite(pred_ret_model) or abs(pred_ret_model) < 0.25:
                    pred_ret = 0.5 * pred_ret_model + 0.5 * recent_20d_return
                    used_fallback = True

            if DEBUG_FORWARD and ticker.upper() in DEBUG_FORWARD_TICKERS:
                debug_parts = [
                    f"[DEBUG] {ticker} | pred_ret_raw={float(pred_ret_raw):.8f}",
                    f"pred_ret_model={float(pred_ret_model):.8f}",
                ]
                if recent_20d_return is not None:
                    debug_parts.append(f"recent_20d_return={float(recent_20d_return):.8f}")
                debug_parts.extend([
                    f"pred_ret={float(pred_ret):.8f}",
                    f"pred_rank={float(pred_rank):.8f}",
                    f"fallback={used_fallback}",
                ])
                print(" | ".join(debug_parts))
                
            current_price = df_price['close'].iloc[-1]
            current_volume = float(df_price['volume'].iloc[-1]) if 'volume' in df_price.columns and pd.notna(df_price['volume'].iloc[-1]) else np.nan
            dollar_turnover = current_price * current_volume if pd.notna(current_volume) else np.nan
            result = {
                "Ticker": ticker,
                "Current_Price": current_price,
                "Current_Volume": current_volume,
                "Current_Price_Source": latest_quote_source,
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
                "News_Corpus": corpus,
                "News_Corpus_Len": len(corpus),
            }
            result["AI_Input_Summary"] = _format_ai_input_summary(pd.Series(result), today_str)
            result["AI_Model_Output_Summary"] = _format_ai_output_summary(pd.Series(result))
            
            # --- MERGE INTO HISTORICAL TRAINING DATA ---
            # 1. Update Vectors
            vec_path = os.path.join(VECTOR_DIR, f"{ticker}_vec.csv")
            vec_row = {"date": today_str}
            for i, val in enumerate(finbert_vec): vec_row[f"dim_{i+1}"] = val
            vec_row['sent_pos'] = sents[0]
            vec_row['sent_neg'] = sents[1]
            vec_row['sent_neu'] = sents[2]
            df_new_vec = pd.DataFrame([vec_row])
            if os.path.exists(vec_path):
                df_old_vec = pd.read_csv(vec_path)
                if today_str not in df_old_vec['date'].values: 
                    pd.concat([df_old_vec, df_new_vec]).to_csv(vec_path, index=False)
            else: 
                os.makedirs(VECTOR_DIR, exist_ok=True)
                df_new_vec.to_csv(vec_path, index=False)

            # 3. Update News Text
            news_path = os.path.join(NEWS_DIR, f"{ticker}_news.csv")
            news_row = {"source_dataset": "Live_v1", "date": today_str, "title": f"v1 Scan {ticker}", "content": corpus, "original_url": "live"}
            df_new_news = pd.DataFrame([news_row])
            if os.path.exists(news_path):
                df_old_news = pd.read_csv(news_path)
                # Upsert today's row so reruns can repair stale/empty same-day content.
                if 'date' in df_old_news.columns:
                    df_old_news['date'] = df_old_news['date'].astype(str)
                    df_old_news = df_old_news[df_old_news['date'] != today_str]
                pd.concat([df_old_news, df_new_news], ignore_index=True).to_csv(news_path, index=False)
            else:
                os.makedirs(NEWS_DIR, exist_ok=True)
                df_new_news.to_csv(news_path, index=False)

            # 4. Update Daily Price
            price_path = os.path.join(PRICE_DIR, f"{ticker}_daily.csv")
            base_cols = [
                'close', 'high', 'low', 'open', 'volume',
                'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'
            ]
            for col in base_cols:
                if col not in df_price.columns:
                    df_price[col] = np.nan
            latest_price = df_price[base_cols].iloc[-1:].join(df_macro.iloc[-1:], how='left')
            latest_price.index.name = "Date"
            if os.path.exists(price_path):
                df_old_price = pd.read_csv(price_path, index_col='Date')
                if latest_price.index[0].strftime("%Y-%m-%d") not in [d.split(' ')[0] for d in df_old_price.index]:
                    pd.concat([df_old_price, latest_price]).to_csv(price_path)
            else:
                os.makedirs(PRICE_DIR, exist_ok=True)
                latest_price.to_csv(price_path)
            
            return result
        except Exception: return None

    print(f"\n🔍 Finalizing Network Inference for {len(SP500_TICKERS)} tickers...")
    predictions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        worker = partial(process_ticker_report, df_macro=df_macro, global_scaler=global_scaler, 
                         schema_cols=schema_cols, pt_model=pt_model, xgb_model=xgb_model, macro_vec=macro_vec, macro_sents=macro_sents)
        futures = {executor.submit(worker, ticker): ticker for ticker in SP500_TICKERS}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                predictions.append(res)
                print(f"📊 Analyzing: {len(predictions)}/{len(SP500_TICKERS)}", end='\r')

    # ==========================================
    # COMPILE THE REPORT
    # ==========================================
    df_preds = pd.DataFrame(predictions)
    if df_preds.empty: 
        exit("\n❌ No valid predictions generated today.")
    
    df_preds_filtered = df_preds[(df_preds['Pred_Return_%'] >= MIN_PRED_RETURN) & (df_preds['XGB_Rank_Score'] >= MIN_XGB_SCORE)]
    df_preds_filtered = df_preds_filtered[df_preds_filtered['Dollar_Turnover'] >= MIN_TURNOVER_DOLLARS]
    df_preds_filtered = df_preds_filtered.sort_values(by=['XGB_Rank_Score', 'Pred_Return_%'], ascending=[False, False])
    if df_preds_filtered.empty:
        print("⚠️  No rows met the return threshold; falling back to the top XGB-ranked names.")
        df_preds_filtered = df_preds[df_preds['XGB_Rank_Score'] >= MIN_XGB_SCORE].sort_values(
            by=['XGB_Rank_Score', 'Pred_Return_%'],
            ascending=[False, False],
        )
        df_preds_filtered = df_preds_filtered[df_preds_filtered['Dollar_Turnover'] >= MIN_TURNOVER_DOLLARS]
        if df_preds_filtered.empty:
            df_preds_filtered = df_preds[df_preds['Dollar_Turnover'] >= MIN_TURNOVER_DOLLARS].sort_values(by=['XGB_Rank_Score', 'Pred_Return_%'], ascending=[False, False])

    agent_check_df = run_tradingagents_crosscheck(df_preds_filtered, date_str)
    if not agent_check_df.empty:
        df_preds_filtered = df_preds_filtered.merge(agent_check_df, on="Ticker", how="left")

    top_targets = df_preds_filtered.head(TOP_N_RECOMMENDATIONS)
    
    # Save today's recommendations to 20-day history and get hold/exit signals
    for _, rec_row in top_targets.iterrows():
        ticker = rec_row["Ticker"]
        agent_rec = str(rec_row.get("Agent_Recommendation", "hold")).lower()
        agent_conf = float(rec_row.get("Agent_Confidence", 0.5)) if pd.notna(rec_row.get("Agent_Confidence")) else 0.5
        agent_thesis = str(rec_row.get("Agent_Investment_Thesis", ""))[:8000]
        entry_price = float(rec_row.get("Current_Price", np.nan)) if pd.notna(rec_row.get("Current_Price")) else np.nan
        _save_recommendation_to_history(ticker, date_str, agent_rec, agent_conf, entry_price, agent_thesis, rec_row.to_dict())
    
    # --- LOG TOP 3 RECOMMENDATIONS TO SIMPLE TRACKER CSV ---
    # Write the latest top-3 picks first so the 20-day monitor can reference them
    tracker_filename = os.path.join(OUTPUT_DIR, "Sentinel_v1_Top_3_Tracker.csv")
    generated_at_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if not top_targets.empty:
        tracker_df = top_targets.loc[:, ["Ticker"]].copy()
        tracker_df.insert(0, "Date", date_str)
        tracker_df["Generated_At_UTC"] = generated_at_now

        # Append to existing tracker or create new one (avoid duplicate date)
        if os.path.exists(tracker_filename):
            try:
                existing_tracker = pd.read_csv(tracker_filename)
                existing_tracker = existing_tracker[existing_tracker["Date"] != date_str]
                pd.concat([existing_tracker, tracker_df], ignore_index=True).to_csv(tracker_filename, index=False)
            except Exception:
                tracker_df.to_csv(tracker_filename, index=False)
        else:
            tracker_df.to_csv(tracker_filename, index=False)
        print(f"📊 Top 3 recommendations tracked in: {tracker_filename}")
    else:
        print(f"📉 No recommendations to track today.")

    # Analyze positions from past 20 days
    monitor_df, detailed_df = _monitor_20day_positions(df_preds, date_str, agent_check_df if not agent_check_df.empty else None)
    
    report_filename = f"Sentinel_v1_Daily_Report_{date_str}.md"
    report_path = os.path.join(OUTPUT_DIR, report_filename)
    
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report = f"# 🛡️ Sentinel AI - Daily Strategy Report\n"
    report += f"**Date:** {date_str}\n"
    report += f"**Generated At (UTC):** {generated_at}\n"
    generated_at_us = datetime.now(timezone(timedelta(hours=-4))).strftime("%Y-%m-%d %H:%M:%S EDT")
    report += f"**Generated At (US Eastern):** {generated_at_us}\n"
    report += f"**Mode:** Fast Inference (High Concurrency, No DeepSeek Mode)\n\n"
    report += f"---\n\n"

    # Include recent Top-3 picks history from the tracker if available
    tracker_path = os.path.join(OUTPUT_DIR, "Sentinel_v1_Top_3_Tracker.csv")
    if os.path.exists(tracker_path):
        try:
            tracker_df = pd.read_csv(tracker_path, parse_dates=["Date"]) if os.path.getsize(tracker_path) > 0 else pd.DataFrame()
        except Exception:
            tracker_df = pd.DataFrame()

        if not tracker_df.empty:
            report += "## Recent Top-3 Picks (by Date)\n"
            # Show most recent TRACKING_DAYS days found in tracker
            tracker_df["Date"] = pd.to_datetime(tracker_df["Date"]).dt.strftime("%Y-%m-%d")
            grouped = tracker_df.groupby("Date")["Ticker"].apply(lambda s: ", ".join(s.tolist()))
            last_dates = list(grouped.index)[-TRACKING_DAYS:]
            for d in last_dates:
                ticks = grouped.get(d, "")
                report += f"- {d}: {ticks}\n"
            report += "\n"
    
    if top_targets.empty:
        report += "### 🛑 No Trades Recommended Today\n"
    else:
        report += f"### 🎯 Top {len(top_targets)} Trade Recommendations (20-Day Horizon)\n"
        report += "**Tickers:** " + ", ".join(top_targets["Ticker"].astype(str).tolist()) + "\n\n"
        # Quick verdicts for Top targets: show simple BUY / DON'T BUY guidance (agent preferred, fallback to model)
        report += "### Quick Verdicts (Buy / Don't Buy)\n"
        for _, trow in top_targets.iterrows():
            t_ticker = trow['Ticker']
            agent_rec = str(trow.get('Agent_Recommendation', '')).lower() if 'Agent_Recommendation' in trow else ''
            if agent_rec in {'buy', 'long'}:
                quick = 'BUY'
            elif agent_rec:
                quick = "DON'T BUY"
            else:
                # fallback to model signal based on predicted return
                try:
                    quick = 'BUY' if float(trow.get('Pred_Return_%', 0.0)) >= float(MIN_PRED_RETURN) else "DON'T BUY"
                except Exception:
                    quick = "DON'T BUY"
            report += f"- **{t_ticker}**: {quick}\n"
        report += "\n"
        for index, row in top_targets.iterrows():
            report += f"## {row['Ticker']} - Buy at Daily Average Price (MTM Simulation)\n"
            report += f"- **Current Price:** ${row['Current_Price']:.2f}\n"
            if pd.notna(row.get('Current_Price_Source', np.nan)):
                report += f"- **Current Price Source:** {row.get('Current_Price_Source')}\n"
            if pd.notna(row.get('Dollar_Turnover', np.nan)):
                report += f"- **Daily Dollar Turnover:** ${float(row.get('Dollar_Turnover')):,.0f}\n"
            if pd.notna(row.get("Recent_20D_Return_%", np.nan)):
                report += f"- **Recent 20D Return (Momentum Context):** {float(row.get('Recent_20D_Return_%')):+.2f}%\n"
            if pd.notna(row.get("Pred_Return_Raw", np.nan)):
                report += f"- **PatchTST Raw Output:** {float(row.get('Pred_Return_Raw')):.6f}\n"
            if pd.notna(row.get("Pred_Return_Model", np.nan)):
                report += f"- **PatchTST De-normalized Output:** {float(row.get('Pred_Return_Model')):+.2f}%\n"
            report += f"- **Predicted 20D Alpha (PatchTST):** +{row['Pred_Return_%']:.2f}%\n"
            report += f"- **XGBoost Relative Rank (Strategy Fit):** {row['XGB_Rank_Score']:.3f}\n"
            report += f"- **Fallback Applied:** {bool(row.get('Used_Fallback', False))}\n"
            # Display TradingAgents detailed analysis (new format) or simple action (legacy format)
            has_detailed = "Agent_Recommendation" in row and pd.notna(row.get("Agent_Recommendation", np.nan))
            has_legacy = "Agent_Action" in row and pd.notna(row.get("Agent_Action", np.nan))
            
            if has_detailed:
                report += f"### TradingAgents Detailed Analysis\n"
                report += f"- **Cross-check Verdict:** {_format_agent_crosscheck_verdict(row)}\n"
                report += f"- **Raw Recommendation:** {str(row.get('Agent_Recommendation', 'n/a')).upper()}\n"
                if pd.notna(row.get("Agent_Confidence", np.nan)):
                    report += f"- **Confidence:** {float(row.get('Agent_Confidence')):.3f}\n"
                if pd.notna(row.get("Agent_Technical_Analysis", np.nan)) and row.get("Agent_Technical_Analysis"):
                    report += f"- **Technical Analysis:** {_shorten_text(row.get('Agent_Technical_Analysis'), 8000)}\n"
                if pd.notna(row.get("Agent_Fundamental_Assessment", np.nan)) and row.get("Agent_Fundamental_Assessment"):
                    report += f"- **Fundamental Assessment:** {_shorten_text(row.get('Agent_Fundamental_Assessment'), 8000)}\n"
                if pd.notna(row.get("Agent_Risk_Assessment", np.nan)) and row.get("Agent_Risk_Assessment"):
                    report += f"- **Risk Assessment:** {_shorten_text(row.get('Agent_Risk_Assessment'), 8000)}\n"
                if pd.notna(row.get("Agent_Price_Target", np.nan)) and row.get("Agent_Price_Target"):
                    report += f"- **Price Target:** {_shorten_text(row.get('Agent_Price_Target'), 8000)}\n"
                if pd.notna(row.get("Agent_Catalysts", np.nan)) and row.get("Agent_Catalysts"):
                    report += f"- **Catalysts:** {_shorten_text(row.get('Agent_Catalysts'), 8000)}\n"
                if pd.notna(row.get("Agent_Investment_Thesis", np.nan)) and row.get("Agent_Investment_Thesis"):
                    report += f"- **Investment Thesis:** {_shorten_text(row.get('Agent_Investment_Thesis'), 8000)}\n"
                if pd.notna(row.get("Agent_Timeline", np.nan)) and row.get("Agent_Timeline"):
                    report += f"- **Timeline:** {_shorten_text(row.get('Agent_Timeline'), 8000)}\n"
                report += f"- **Status:** {row.get('Agent_Status', 'n/a')}\n"
            elif has_legacy:
                report += f"- **TradingAgents Cross-Check Action:** {row.get('Agent_Action', 'n/a')}\n"
                if pd.notna(row.get("Agent_Confidence", np.nan)):
                    report += f"- **TradingAgents Confidence:** {float(row.get('Agent_Confidence')):.3f}\n"
                report += f"- **TradingAgents Reason:** {_format_agent_reason(row)}\n"
                report += f"- **TradingAgents Status:** {row.get('Agent_Status', 'n/a')}\n"
                report += f"- **TradingAgents Pass (buy/long):** {row.get('Agent_Pass', False)}\n"
                if pd.notna(row.get("Agent_Output_Raw", np.nan)):
                    report += f"- **TradingAgents Reasoning Detail:** {_shorten_text(row.get('Agent_Output_Raw'), 400)}\n"
            report += f"- **Execution Model:** AVG (Backtested Slidance & Fees: 1%)\n\n"
    
    # Add 20-day position review section
    if not monitor_df.empty:
        report += f"\n---\n\n"  
        report += f"### 📊 20-Day Recommendation Review: Hold / Exit Guidance\n"
        report += f"Analysis of your recommendations from the past {TRACKING_DAYS} days:\n\n"
        
        # Simplify monitor into three clear actions for readability: BUY MORE, HOLD, EXIT
        mapping = {
            'EXIT NOW': 'EXIT',
            'EXIT / STOP LOSS': 'EXIT',
            'TAKE PROFITS': 'EXIT',
            'EXIT (thesis complete)': 'EXIT',
            'TRIM / REDUCE': 'HOLD',
            'REVERSE TO BUY': 'BUY MORE',
            'UPGRADE TO BUY': 'BUY MORE',
            'HOLD': 'HOLD',
            'MAINTAIN SELL': 'EXIT',
        }
        monitor_df['Simple_Action'] = monitor_df['Exit_Signal'].map(mapping).fillna('HOLD')
        simple_order = ['BUY MORE', 'HOLD', 'EXIT']
        for action in simple_order:
            rows = monitor_df[monitor_df['Simple_Action'] == action]
            if rows.empty:
                continue
            count = len(rows)
            report += f"**{action}** ({count} position{'s' if count > 1 else ''}):\n"
            for _, pos_row in rows.iterrows():
                ticker = pos_row['Ticker']
                days = int(pos_row['Days_Held'])
                price_chg = pos_row['Price_Change_%']
                reason = pos_row['Reason']
                report += f"- **{ticker}**: {days}d held, {price_chg:+.1f}% price change | {reason}\n"
            report += "\n"
        
        report += f"\n---\n"
    
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n\n✅ Report successfully generated: {report_path}")
    
    

    # --- LOG ALL PREDICTIONS TO TOTAL CSV (Mirroring Folder E Format) ---
    total_log_filename = os.path.join(OUTPUT_DIR, "final_ensemble_results_live_v1.csv")
    ai_io_log_filename = os.path.join(OUTPUT_DIR, AI_IO_LOG_FILENAME)
    all_preds_df = pd.DataFrame(predictions)
    if not all_preds_df.empty:
        if not agent_check_df.empty:
            all_preds_df = all_preds_df.merge(agent_check_df, on="Ticker", how="left")

        all_preds_df.insert(0, 'Date', date_str)
        all_preds_df["Generated_At_UTC"] = generated_at
        
        all_preds_df["Actual_20D_Return_%"] = np.nan
        all_preds_df["Actual_Alpha_%"] = np.nan
        all_preds_df["AI_Output_Summary"] = all_preds_df.apply(_format_ai_output_summary, axis=1)

        if os.path.exists(total_log_filename):
            existing_total = pd.read_csv(total_log_filename)
            existing_total = existing_total[~((existing_total['Date'] == date_str))]
            pd.concat([existing_total, all_preds_df]).to_csv(total_log_filename, index=False)
        else:
            all_preds_df.to_csv(total_log_filename, index=False)
        print(f"🗄️ All {len(all_preds_df)} predictions logged to total CSV: {total_log_filename}")

        ai_io_df = all_preds_df.copy()
        ai_io_columns = [
            "Date",
            "Generated_At_UTC",
            "Ticker",
            "Current_Price",
            "Current_Volume",
            "Dollar_Turnover",
            "Recent_20D_Return_%",
            "Pred_Return_Raw",
            "Pred_Return_Model",
            "Pred_Return_%",
            "XGB_Rank_Score",
            "Used_Fallback",
            "FinBERT_Pos",
            "FinBERT_Neg",
            "FinBERT_Neu",
            "Market_Sent_Pos",
            "Market_Sent_Neg",
            "Market_Sent_Neu",
            "News_Corpus_Len",
            "News_Corpus",
            "AI_Input_Summary",
            "AI_Model_Output_Summary",
            "AI_Output_Summary",
            "Agent_Input_Summary",
            "Agent_Output_Raw",
            "Agent_Action",
            "Agent_Confidence",
            "Agent_Summary",
            "Agent_Status",
            "Agent_Pass",
            "Actual_20D_Return_%",
            "Actual_Alpha_%",
        ]
        for col in ai_io_columns:
            if col not in ai_io_df.columns:
                ai_io_df[col] = np.nan
        ai_io_df = ai_io_df[ai_io_columns]
        if os.path.exists(ai_io_log_filename):
            existing_ai_io = pd.read_csv(ai_io_log_filename)
            if 'Date' in existing_ai_io.columns:
                existing_ai_io = existing_ai_io[existing_ai_io['Date'] != date_str]
            pd.concat([existing_ai_io, ai_io_df]).to_csv(ai_io_log_filename, index=False)
        else:
            ai_io_df.to_csv(ai_io_log_filename, index=False)
        print(f"🧾 AI input/output log saved to: {ai_io_log_filename}")
    
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel forward pipeline with optional CSV-only agent mode.")
    parser.add_argument("--agent-csv-input", type=str, default="", help="CSV input path for CSV-only TradingAgents analysis")
    parser.add_argument("--agent-csv-output", type=str, default="", help="CSV output path for CSV-only TradingAgents analysis")
    parser.add_argument("--agent-date", type=str, default="", help="Date label (YYYY-MM-DD) used in prompts for CSV-only mode")
    parser.add_argument("--agent-top-n", type=int, default=3, help="Number of tickers to analyze in CSV-only mode")
    args = parser.parse_args()

    if args.agent_csv_input:
        us_now = datetime.now(timezone(timedelta(hours=-4)))
        date_str = args.agent_date.strip() if args.agent_date.strip() else us_now.strftime("%Y-%m-%d")
        output_path = args.agent_csv_output.strip() if args.agent_csv_output.strip() else os.path.join(OUTPUT_DIR, f"Sentinel_v1_Agent_Top{max(1, int(args.agent_top_n))}_{date_str}.csv")
        _run_csv_only_agent_analysis(
            input_csv_path=args.agent_csv_input.strip(),
            output_csv_path=output_path,
            date_str=date_str,
            top_n=max(1, int(args.agent_top_n)),
        )
    else:
        generate_daily_report()
