import gc
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRICE_DIR = os.path.join(BASE_DIR, "001_final_db_daily")
VECTOR_DIR_ALL_NEWS = os.path.join(BASE_DIR, "0002_final_vectors_all_news")
OUTPUT_DIR = os.path.join(BASE_DIR, "0003_ready_20")
SCALER_DIR = os.path.join(BASE_DIR, "0003_scalers_20")

LOOKBACK_WINDOW = 60
PREDICTION_HORIZON = 20
VECTOR_DIM = 28
BATCH_SIZE = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)


class FusionEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.schema_cols = None

    def align_to_schema(self, df, ticker):
        if self.schema_cols is None:
            if "SPY_Close" in df.columns and len(df.columns) >= (5 + 28):
                self.schema_cols = df.columns.tolist()
                joblib.dump(self.schema_cols, os.path.join(SCALER_DIR, "schema_cols.pkl"))
            else:
                return None
        return df.reindex(columns=self.schema_cols, fill_value=0.0)

    def load_ticker_data(self, ticker, start_date="2016-01-01", end_date="2024-01-01"):
        price_path = os.path.join(PRICE_DIR, f"{ticker}_daily.csv")
        if not os.path.exists(price_path):
            return None

        try:
            df_price = pd.read_csv(price_path, parse_dates=["Date"], index_col="Date").sort_index()
            df_price = df_price[(df_price.index >= start_date) & (df_price.index < end_date)]
            if len(df_price) < 100:
                return None

            full_cal = pd.date_range(start=df_price.index.min(), end=df_price.index.max(), freq="D")

            vec_path = os.path.join(VECTOR_DIR_ALL_NEWS, f"{ticker}_vec.csv")
            if os.path.exists(vec_path):
                df_vec = pd.read_csv(vec_path, parse_dates=["date"]).set_index("date").sort_index()
                df_vec = df_vec[~df_vec.index.duplicated(keep="first")]
                df_vec = df_vec.reindex(full_cal).fillna(0.0).ewm(alpha=0.3, adjust=False).mean().shift(1)
                df = df_price.ffill().join(df_vec, how="left").fillna(0.0)
            else:
                df = pd.concat(
                    [
                        df_price,
                        pd.DataFrame(
                            0.0,
                            index=df_price.index,
                            columns=[f"dim_{i + 1}" for i in range(VECTOR_DIM)],
                        ),
                    ],
                    axis=1,
                )

            return df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        except Exception:
            return None

    def fit_scalers(self, files):
        print("--- Calibrating NO-DEEPSEEK Scaler ---")
        batch_values = []
        import random

        sampled_files = random.sample(files, min(300, len(files)))
        for f in tqdm(sampled_files):
            ticker = f.replace("_daily.csv", "")
            df = self.load_ticker_data(ticker, start_date="2015-01-01", end_date="2016-01-01")
            if df is not None:
                df_train_only = df[df.index < "2016-01-01"]
                if len(df_train_only) < 50:
                    continue

                df_feat = df_train_only.copy()
                num_cols = df_feat.select_dtypes(include=[np.number]).columns
                cols_to_pct = [
                    c
                    for c in num_cols
                    if any(x in c.lower() for x in ["close", "open", "high", "low", "cpi", "m2", "volume"])
                ]
                cols_to_diff = [
                    c
                    for c in num_cols
                    if any(x in c.lower() for x in ["fed", "vix", "rsi", "macd"])
                ]

                df_feat[cols_to_pct] = (
                    df_feat[cols_to_pct]
                    .pct_change()
                    .fillna(0.0)
                    .replace([np.inf, -np.inf], 0.0)
                    .clip(-5.0, 5.0)
                )
                df_feat[cols_to_diff] = (
                    df_feat[cols_to_diff]
                    .diff()
                    .fillna(0.0)
                    .replace([np.inf, -np.inf], 0.0)
                    .clip(-5.0, 5.0)
                )

                aligned = self.align_to_schema(df_feat, ticker)
                if aligned is not None:
                    batch_values.append(aligned.values)

            if len(batch_values) > 50:
                X_batch = np.concatenate(batch_values, axis=0)
                if np.all(np.isfinite(X_batch)):
                    self.scaler.partial_fit(X_batch)
                else:
                    print(f"    [SKIP BATCH] Non-finite values detected in batch ({ticker})")
                batch_values = []

        if batch_values:
            X_batch = np.concatenate(batch_values, axis=0)
            if np.all(np.isfinite(X_batch)):
                self.scaler.partial_fit(X_batch)
            else:
                print("    [SKIP BATCH] Non-finite values detected in final batch.")

        joblib.dump(self.scaler, os.path.join(SCALER_DIR, "global_scaler.pkl"))

    def create_sequences(self, df, ticker):
        if df is None:
            return None, None

        df_feat = df.copy()
        num_cols = df_feat.select_dtypes(include=[np.number]).columns
        cols_to_pct = [
            c
            for c in num_cols
            if any(x in c.lower() for x in ["close", "open", "high", "low", "cpi", "m2", "volume"])
        ]
        cols_to_diff = [
            c
            for c in num_cols
            if any(x in c.lower() for x in ["fed", "vix", "rsi", "macd"])
        ]

        df_feat[cols_to_pct] = (
            df_feat[cols_to_pct].pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0).clip(-5.0, 5.0)
        )
        df_feat[cols_to_diff] = (
            df_feat[cols_to_diff].diff().fillna(0.0).replace([np.inf, -np.inf], 0.0).clip(-5.0, 5.0)
        )

        df_feat = self.align_to_schema(df_feat, ticker)
        if df_feat is None:
            return None, None

        try:
            data_scaled = self.scaler.transform(df_feat.values)
        except Exception:
            return None, None

        close_col = next((c for c in ["adj close", "Close", "close"] if c in df.columns), None)
        open_col = next((c for c in ["adj open", "Open", "open"] if c in df.columns), close_col)

        target_idx_close = df.columns.get_loc(close_col)
        target_idx_open = df.columns.get_loc(open_col)
        spy_close_col = df.columns.get_loc("SPY_Close")
        spy_open_col = df.columns.get_loc("SPY_Open")

        df_copy = df.copy()
        df_copy["stock_ret"] = df_copy[close_col].pct_change()
        df_copy["spy_ret"] = df_copy["SPY_Close"].pct_change()
        rolling_cov = df_copy["stock_ret"].rolling(LOOKBACK_WINDOW).cov(df_copy["spy_ret"])
        rolling_var = df_copy["spy_ret"].rolling(LOOKBACK_WINDOW).var()
        df_copy["rolling_beta"] = (rolling_cov / rolling_var).clip(-2.0, 3.0).fillna(1.0)
        beta_col_idx = df_copy.columns.get_loc("rolling_beta")

        xs, ys = [], []
        for i in range(len(data_scaled) - LOOKBACK_WINDOW - PREDICTION_HORIZON):
            xs.append(data_scaled[i : i + LOOKBACK_WINDOW])
            p_curr = df.iloc[i + LOOKBACK_WINDOW, target_idx_open]
            p_fut = df.iloc[i + LOOKBACK_WINDOW + PREDICTION_HORIZON - 1, target_idx_close]
            spy_curr = df.iloc[i + LOOKBACK_WINDOW, spy_open_col]
            spy_fut = df.iloc[i + LOOKBACK_WINDOW + PREDICTION_HORIZON - 1, spy_close_col]
            beta = df_copy.iloc[i + LOOKBACK_WINDOW - 1, beta_col_idx]

            if p_curr > 0 and spy_curr > 0:
                alpha = ((p_fut - p_curr) / p_curr) - (beta * ((spy_fut - spy_curr) / spy_curr))
                ys.append([alpha * 100.0])
            else:
                ys.append([0.0])

        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


if __name__ == "__main__":
    files = [f for f in os.listdir(PRICE_DIR) if f.endswith("_daily.csv")]
    engine = FusionEngine()
    engine.fit_scalers(files)

    print("--- Generating NO-DEEPSEEK Batches ---")
    batch_X, batch_y, batch_id = [], [], 0
    for i, f in enumerate(tqdm(files)):
        X, y = engine.create_sequences(
            engine.load_ticker_data(f.replace("_daily.csv", ""), start_date="2016-01-01", end_date="2024-01-01"),
            f,
        )
        if X is None or len(X) == 0:
            continue
        batch_X.append(X)
        batch_y.append(y)

        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == len(files):
            if not batch_X:
                continue
            np.savez_compressed(
                os.path.join(OUTPUT_DIR, f"train_batch_nodis_{batch_id}.npz"),
                X=np.concatenate(batch_X, axis=0),
                y=np.concatenate(batch_y, axis=0),
            )
            batch_id += 1
            batch_X, batch_y = [], []
            gc.collect()
