import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import joblib
import xgboost as xgb
from tqdm import tqdm

PRICE_DIR = "001_final_db_daily"
VECTOR_DIR_TRAIN = "002_final_vectors_train"
VECTOR_DIR_TEST = "002_final_vectors_backtest" 
SCALER_DIR = "003_scalers_20" 
MODEL_SAVE_DIR = "004_saved_models_20"
OUTPUT_FILE = "final_ensemble_results_20.csv"

LOOKBACK_WINDOW = 60
PREDICTION_HORIZON = 20
VECTOR_DIM = 28
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_combined_data(ticker, train_dir, test_dir):
    df_list = []
    train_path = os.path.join(train_dir, f"{ticker}_vec.csv")
    test_path = os.path.join(test_dir, f"{ticker}_vec.csv")
    
    for path in [train_path, test_path]:
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['date']).set_index('date')
            df_list.append(df)
            
    if not df_list: return None
    combined = pd.concat(df_list).sort_index()
    return combined[~combined.index.duplicated(keep='last')]

def validate_required_paths():
    required_dirs = [PRICE_DIR, SCALER_DIR, MODEL_SAVE_DIR]
    missing_dirs = [d for d in required_dirs if not os.path.isdir(d)]
    if missing_dirs:
        print(f"❌ Missing required directories: {missing_dirs}")
        return False

    required_files = [
        os.path.join(SCALER_DIR, "global_scaler.pkl"),
        os.path.join(SCALER_DIR, "schema_cols.pkl"),
        os.path.join(MODEL_SAVE_DIR, "patchtst_20.pth"),
        os.path.join(MODEL_SAVE_DIR, "xgboost_20.json"),
    ]
    missing_files = [p for p in required_files if not os.path.isfile(p)]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    return True

def run_ensemble_inference():
    print("🧠 SENTINEL AI: 20-DAY NO-DS INFERENCE (Day 1 Open Execution)")
    if not validate_required_paths():
        return

    scaler = joblib.load(os.path.join(SCALER_DIR, "global_scaler.pkl"))
    schema_cols = joblib.load(os.path.join(SCALER_DIR, "schema_cols.pkl"))

    model_pt = PatchTST(num_features=len(schema_cols), seq_len=LOOKBACK_WINDOW).to(DEVICE)
    model_path = os.path.join(MODEL_SAVE_DIR, "patchtst_20.pth")
    try:
        pt_state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        pt_state = torch.load(model_path, map_location=DEVICE)
    if list(pt_state.keys())[0].startswith('module.'): pt_state = {k[7:]: v for k, v in pt_state.items()}
    model_pt.load_state_dict(pt_state)
    model_pt.eval()

    model_xgb = xgb.Booster()
    model_xgb.load_model(os.path.join(MODEL_SAVE_DIR, "xgboost_20.json"))

    files = [f for f in os.listdir(PRICE_DIR) if f.endswith("_daily.csv")]
    results = []

    with torch.no_grad():
        for f in tqdm(files, desc="Running NO-DS Inference"):
            ticker = f.replace("_daily.csv", "")
            try:
                df_price = pd.read_csv(os.path.join(PRICE_DIR, f), parse_dates=['Date'], index_col='Date').sort_index()
                df_price = df_price[df_price.index >= "2023-10-01"] 
                if len(df_price) < LOOKBACK_WINDOW + PREDICTION_HORIZON: continue

                close_col = next((c for c in ['adj close', 'Close', 'close'] if c in df_price.columns), None)
                open_col = next((c for c in ['adj open', 'Open', 'open'] if c in df_price.columns), close_col)
                if close_col is None or 'SPY_Close' not in df_price.columns: continue

                spy_open_col = 'SPY_Open' if 'SPY_Open' in df_price.columns else 'SPY_Close'
                volume_col = next((c for c in ['volume', 'Volume'] if c in df_price.columns), None)

                stock_ret = df_price[close_col].pct_change()
                spy_ret = df_price['SPY_Close'].pct_change()
                rolling_cov = stock_ret.rolling(LOOKBACK_WINDOW).cov(spy_ret)
                rolling_var = spy_ret.rolling(LOOKBACK_WINDOW).var()
                rolling_beta = (rolling_cov / rolling_var).clip(-2.0, 3.0).fillna(1.0)

                full_cal = pd.date_range(start=df_price.index.min(), end=df_price.index.max(), freq='D')

                # FinBERT Vectors
                df_vec_combined = get_combined_data(ticker, VECTOR_DIR_TRAIN, VECTOR_DIR_TEST)
                if df_vec_combined is not None:
                    df_vec = df_vec_combined.reindex(full_cal).fillna(0.0).ewm(alpha=0.3, adjust=False).mean().shift(1)
                    df = df_price.ffill().join(df_vec, how='left').fillna(0.0)
                else:
                    df = pd.concat([df_price, pd.DataFrame(0.0, index=df_price.index, columns=[f'dim_{i+1}' for i in range(VECTOR_DIM)])], axis=1)

                # No DeepSeek Join

                df_feat = df.copy()
                num_cols = df_feat.select_dtypes(include=[np.number]).columns
                cols_to_pct = [c for c in num_cols if any(x in c.lower() for x in ["close", "open", "high", "low", "cpi", "m2", "volume"])]
                cols_to_diff = [c for c in num_cols if any(x in c.lower() for x in ["fed", "vix", "rsi", "macd"])]
                
                df_feat[cols_to_pct] = df_feat[cols_to_pct].pct_change().fillna(0.0)
                df_feat[cols_to_diff] = df_feat[cols_to_diff].diff().fillna(0.0)
                df_feat = df_feat.reindex(columns=schema_cols, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

                data_scaled = scaler.transform(df_feat.values)
                xs_3d, xs_1d, metadata = [], [], []
                
                # Keep only fully realized 20-day windows to avoid "Live_NA" rows.
                for i in range(len(data_scaled) - LOOKBACK_WINDOW - PREDICTION_HORIZON + 1):
                    target_date_idx = i + LOOKBACK_WINDOW
                    if target_date_idx >= len(df.index):
                        continue

                    target_date = df.index[target_date_idx]
                    if target_date < pd.Timestamp("2024-01-01"): continue
                    
                    entry_idx = i + LOOKBACK_WINDOW
                    exit_idx = i + LOOKBACK_WINDOW + PREDICTION_HORIZON - 1
                    if exit_idx >= len(df_price):
                        continue

                    exec_curr = float(df_price[open_col].iloc[entry_idx])
                    spy_exec_curr = float(df_price[spy_open_col].iloc[entry_idx])

                    if exec_curr <= 0 or spy_exec_curr <= 0:
                        continue

                    if volume_col is not None and entry_idx - 20 >= 0:
                        recent_vol = df_price[volume_col].iloc[entry_idx - 20:entry_idx].mean()
                        recent_price = df_price[close_col].iloc[entry_idx - 20:entry_idx].mean()
                        if recent_vol < 750000 or recent_price < 5.0:
                            continue

                    p_fut = float(df_price[close_col].iloc[exit_idx])
                    spy_fut = float(df_price['SPY_Close'].iloc[exit_idx])
                    act_ret = ((p_fut - exec_curr) / exec_curr) * 100
                    beta = float(rolling_beta.iloc[target_date_idx]) if target_date_idx < len(rolling_beta) else 1.0
                    spy_ret_pct = ((spy_fut - spy_exec_curr) / spy_exec_curr) * 100
                    act_alpha = act_ret - (beta * spy_ret_pct)
                    
                    window = data_scaled[i : i + LOOKBACK_WINDOW]
                    xs_3d.append(window)
                    xs_1d.append(np.concatenate([window[-1], window.mean(axis=0), window.std(axis=0)]))
                    
                    metadata.append({
                        "Date": target_date.strftime("%Y-%m-%d"), 
                        "Ticker": ticker, 
                        "Entry_Price": exec_curr, 
                        "Actual_20D_Return_%": act_ret, 
                        "Actual_Alpha_%": act_alpha
                    })

                if not xs_3d: continue

                pred_returns = model_pt(torch.tensor(np.array(xs_3d), dtype=torch.float32).to(DEVICE)).cpu().numpy().flatten()
                pred_ranks = model_xgb.predict(xgb.DMatrix(np.array(xs_1d, dtype=np.float32)))

                for meta, p_ret, p_rank in zip(metadata, pred_returns, pred_ranks):
                    results.append({
                        "Date": meta["Date"], "Ticker": meta["Ticker"], 
                        "Current_Price": round(meta["Entry_Price"], 2),
                        "Pred_Return_%": round(p_ret, 2),
                        "XGB_Rank_Score": round(p_rank, 4),      
                        "Actual_20D_Return_%": round(meta["Actual_20D_Return_%"], 2),
                        "Actual_Alpha_%": round(meta["Actual_Alpha_%"], 2),
                    })
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

    pd.DataFrame(results).sort_values(by=["Date", "Ticker"]).to_csv(OUTPUT_FILE, index=False)
    print(f"NO-DS Results Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_ensemble_inference()
