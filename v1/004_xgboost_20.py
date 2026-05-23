import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from tqdm import tqdm
import gc

PRICE_DIR = "001_final_db_daily"
VECTOR_DIR_TRAIN = "002_final_vectors_train" 
SCALER_DIR = "003_scalers_20"       
MODEL_SAVE_DIR = "004_saved_models_20"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "xgboost_20.json")

LOOKBACK_WINDOW = 60 
PREDICTION_HORIZON = 20         

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def extract_all_features(files, scaler, schema_cols):
    X_all, y_all, qid_all = [], [], []
    
    for f in tqdm(files, desc="Extracting XGBoost NO-DS Data"):
        ticker = f.replace("_daily.csv", "")
        price_path = os.path.join(PRICE_DIR, f)
        if not os.path.exists(price_path): continue
        
        try:
            df_price = pd.read_csv(price_path, parse_dates=['Date'], index_col='Date').sort_index()
            df_price = df_price[(df_price.index >= "2016-01-01") & (df_price.index < "2024-01-01")] 
            if len(df_price) < (LOOKBACK_WINDOW + PREDICTION_HORIZON + 5): continue

            close_col = next((c for c in ['adj close', 'Close', 'close'] if c in df_price.columns), None)
            open_col = next((c for c in ['adj open', 'Open', 'open'] if c in df_price.columns), close_col)
            if close_col is None or 'SPY_Close' not in df_price.columns: continue

            full_cal = pd.date_range(start=df_price.index.min(), end=df_price.index.max(), freq='D')

            # FinBERT
            vec_path = os.path.join(VECTOR_DIR_TRAIN, f"{ticker}_vec.csv")
            if os.path.exists(vec_path):
                df_vec = pd.read_csv(vec_path, parse_dates=['date']).set_index('date').sort_index()
                df_vec = df_vec[~df_vec.index.duplicated(keep='first')]
                df_vec = df_vec.reindex(full_cal).fillna(0.0).ewm(alpha=0.3, adjust=False).mean().shift(1)
                df = df_price.ffill().join(df_vec, how='left').fillna(0.0)
            else:
                df = pd.concat([df_price, pd.DataFrame(0.0, index=df_price.index, columns=[f'dim_{i+1}' for i in range(28)])], axis=1)

            # DeepSeek Removed

            df_feat = df.copy()
            num_cols = df_feat.select_dtypes(include=[np.number]).columns
            cols_to_pct = [c for c in num_cols if any(x in c.lower() for x in ["close", "open", "high", "low", "cpi", "m2", "volume"])]
            cols_to_diff = [c for c in num_cols if any(x in c.lower() for x in ["fed", "vix", "rsi", "macd"])]
            
            df_feat[cols_to_pct] = df_feat[cols_to_pct].pct_change().fillna(0.0)
            df_feat[cols_to_diff] = df_feat[cols_to_diff].diff().fillna(0.0)

            df_scaled_values = df_feat.reindex(columns=schema_cols, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            data_scaled = scaler.transform(df_scaled_values.values)
            
            df_copy = df.copy()
            df_copy['stock_ret'] = df_copy[close_col].pct_change()
            df_copy['spy_ret'] = df_copy['SPY_Close'].pct_change()
            rolling_cov = df_copy['stock_ret'].rolling(LOOKBACK_WINDOW).cov(df_copy['spy_ret'])
            rolling_var = df_copy['spy_ret'].rolling(LOOKBACK_WINDOW).var()
            df_copy['rolling_beta'] = (rolling_cov / rolling_var).clip(-2.0, 3.0).fillna(1.0)
            beta_col_idx = df_copy.columns.get_loc('rolling_beta')

            for i in range(len(data_scaled) - LOOKBACK_WINDOW - PREDICTION_HORIZON):
                target_date = df.index[i + LOOKBACK_WINDOW]
                p_start = df[open_col].iloc[i + LOOKBACK_WINDOW] 
                p_end = df[close_col].iloc[i + LOOKBACK_WINDOW + PREDICTION_HORIZON - 1]
                spy_start = df['SPY_Open'].iloc[i + LOOKBACK_WINDOW]  
                spy_end   = df['SPY_Close'].iloc[i + LOOKBACK_WINDOW + PREDICTION_HORIZON - 1]
                
                if p_start <= 0 or spy_start <= 0: continue

                window = data_scaled[i : i + LOOKBACK_WINDOW]
                feats = np.concatenate([window[-1], window.mean(axis=0), window.std(axis=0)])
                X_all.append(feats)
                beta = df_copy.iloc[i + LOOKBACK_WINDOW - 1, beta_col_idx]
                alpha = ((p_end - p_start) / p_start) - (beta * ((spy_end - spy_start) / spy_start))
                y_all.append(alpha * 100.0)
                qid_all.append(int(target_date.strftime("%Y%m%d")))
        except Exception: continue

    if not qid_all: return np.array([]), np.array([]), np.array([])
    
    sort_idx = np.argsort(qid_all)
    return np.array(X_all, dtype=np.float32)[sort_idx], np.array(y_all, dtype=np.float32)[sort_idx], np.array(qid_all, dtype=np.int32)[sort_idx]

if __name__ == "__main__":
    print("--- NO-DEEPSEEK XGBOOST TRAINING ---")
    files = [f for f in os.listdir(PRICE_DIR) if f.endswith("_daily.csv")]
    
    scaler = joblib.load(os.path.join(SCALER_DIR, "global_scaler.pkl"))
    schema_cols = joblib.load(os.path.join(SCALER_DIR, "schema_cols.pkl"))
    
    params = {'objective': 'rank:pairwise', 'eval_metric': 'ndcg', 'tree_method': 'hist', 'max_depth': 8, 'learning_rate': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42}

    X, y, qid = extract_all_features(files, scaler, schema_cols)
    
    if len(X) > 0:
        dtrain = xgb.DMatrix(X, label=y, qid=qid)
        try:
            cuda_params = dict(params)
            cuda_params['device'] = 'cuda'
            xgb_model = xgb.train(cuda_params, dtrain, num_boost_round=100)
        except Exception as e:
            print(f"CUDA training unavailable ({e}); falling back to CPU.")
            cpu_params = dict(params)
            cpu_params['device'] = 'cpu'
            xgb_model = xgb.train(cpu_params, dtrain, num_boost_round=100)
        xgb_model.save_model(MODEL_PATH)
        print(f"Model saved: {MODEL_PATH}")
    else:
        print("Error: No valid data extracted.")
