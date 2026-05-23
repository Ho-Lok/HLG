import gc
import glob
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TRAIN_INPUT_DIR = "001_ticker_news_train"
BACKTEST_INPUT_DIR = "001_ticker_news_backtest"
ALL_NEWS_OUTPUT_DIR = "0002_final_vectors_all_news"

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 64
MAX_TOKENS = 256
MAX_FILE_SIZE_MB = 800

PCA_COMPONENTS = 28
PCA_SAMPLE_SIZE = 500000
PCA_MODEL_PATH = "pca_model_live.pkl"
SCALER_MODEL_PATH = "scaler_model_live.pkl"

COLUMN_ALIASES = {
    "date": ["date", "published_at", "timestamp", "time", "datetime"],
    "title": ["title", "headline", "header", "subject", "article_title"],
    "body": ["content", "body", "article", "text", "summary"],
}


class FinancialVectorEngine:
    def __init__(self, model_name=MODEL_NAME, batch_size=BATCH_SIZE):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"--- Hardware: Forced {self.device} ---")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Loading FinBERT in FP16 Turbo Mode...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device).eval()
        self.batch_size = batch_size

    def get_embeddings(self, texts):
        all_embeddings, all_sentiments = [], []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1].to(torch.float32)
                cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
                probs = torch.nn.functional.softmax(outputs.logits.to(torch.float32), dim=-1).cpu().numpy()
            all_embeddings.append(cls_embeddings)
            all_sentiments.append(probs)
        return np.vstack(all_embeddings), np.vstack(all_sentiments)


def smart_load_df(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        cols = {c.lower(): c for c in df.columns}
        title_col = next((cols[a] for a in COLUMN_ALIASES["title"] if a in cols), None)
        body_col = next((cols[a] for a in COLUMN_ALIASES["body"] if a in cols), None)
        date_col = next((cols[a] for a in COLUMN_ALIASES["date"] if a in cols), None)

        if not title_col and not body_col:
            return None

        if title_col and body_col:
            df["processed_text"] = df[title_col].fillna("") + ". " + df[body_col].fillna("")
        elif title_col:
            df["processed_text"] = df[title_col].fillna("")
        else:
            df["processed_text"] = df[body_col].fillna("")

        if date_col:
            df["processed_date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.date
            df = df.dropna(subset=["processed_date"])
        else:
            return None

        return df[["processed_date", "processed_text"]]
    except Exception as e:
        print(f"\n[Read Error] {filepath}: {e}")
        return None


def get_or_train_pca(engine, reference_input_dir):
    if os.path.exists(PCA_MODEL_PATH) and os.path.exists(SCALER_MODEL_PATH):
        with open(PCA_MODEL_PATH, "rb") as f:
            pca_model = pickle.load(f)
        with open(SCALER_MODEL_PATH, "rb") as f:
            scaler_model = pickle.load(f)
        return scaler_model, pca_model

    print("\n=== Fitting PCA on Burn-In Data (2015) ===")
    all_files = glob.glob(os.path.join(reference_input_dir, "*.csv"))
    sampled_embeddings = []

    for f in tqdm(all_files, desc="Sampling early data for PCA"):
        df = smart_load_df(f)
        if df is None or df.empty:
            continue

        early_df = df[df["processed_date"] < pd.Timestamp("2016-01-01").date()]
        if early_df.empty:
            continue

        sample = early_df.sample(min(len(early_df), 500))["processed_text"].tolist()
        if sample:
            embeddings, _ = engine.get_embeddings(sample)
            sampled_embeddings.append(embeddings)
            if sum(len(x) for x in sampled_embeddings) >= PCA_SAMPLE_SIZE:
                break

    if not sampled_embeddings:
        raise RuntimeError("No burn-in samples found for PCA. Ensure news files contain pre-2016 data.")

    X = np.vstack(sampled_embeddings)
    scaler = StandardScaler()
    pca = PCA(n_components=PCA_COMPONENTS)
    pca.fit(scaler.fit_transform(X))

    with open(PCA_MODEL_PATH, "wb") as f:
        pickle.dump(pca, f)
    with open(SCALER_MODEL_PATH, "wb") as f:
        pickle.dump(scaler, f)
    return scaler, pca


def collect_all_news_files():
    ticker_to_files = {}
    for input_dir in [TRAIN_INPUT_DIR, BACKTEST_INPUT_DIR]:
        if not os.path.exists(input_dir):
            continue
        for f in glob.glob(os.path.join(input_dir, "*_news.csv")):
            ticker = os.path.basename(f).replace("_news.csv", "").upper()
            ticker_to_files.setdefault(ticker, []).append(f)
    return ticker_to_files


def transform_all_news_and_save(engine, scaler, pca):
    print("\n=== Processing ALL NEWS (TRAIN + BACKTEST) ===")
    os.makedirs(ALL_NEWS_OUTPUT_DIR, exist_ok=True)

    ticker_to_files = collect_all_news_files()
    tickers = sorted(ticker_to_files.keys())

    for ticker in tqdm(tickers, desc="Vectorizing ALL NEWS"):
        out_path = os.path.join(ALL_NEWS_OUTPUT_DIR, f"{ticker}_vec.csv")
        if os.path.exists(out_path):
            continue

        source_files = ticker_to_files[ticker]
        total_size_mb = sum(os.path.getsize(p) for p in source_files) / (1024 * 1024)
        if total_size_mb > MAX_FILE_SIZE_MB:
            print(f"\n[WHALE SHIELD] Skipping {ticker} ({total_size_mb:.1f} MB merged)")
            continue

        parts = []
        for src in source_files:
            df = smart_load_df(src)
            if df is not None and not df.empty:
                parts.append(df)

        if not parts:
            continue

        merged = pd.concat(parts, ignore_index=True)
        merged = merged.drop_duplicates(subset=["processed_date", "processed_text"])

        if merged.empty:
            continue

        try:
            raw_emb, sentiments = engine.get_embeddings(merged["processed_text"].tolist())
            reduced_emb = pca.transform(scaler.transform(raw_emb))
            res_df = pd.DataFrame(reduced_emb, columns=[f"dim_{i+1}" for i in range(reduced_emb.shape[1])])
            res_df["sent_pos"] = sentiments[:, 0]
            res_df["sent_neg"] = sentiments[:, 1]
            res_df["sent_neu"] = sentiments[:, 2]
            res_df["date"] = merged["processed_date"].values
            res_df.groupby("date").mean().reset_index().to_csv(out_path, index=False)
        except Exception as e:
            print(f"\n[GPU Error] {ticker}: {e}")

        del parts, merged, res_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    has_train = os.path.exists(TRAIN_INPUT_DIR) and bool(os.listdir(TRAIN_INPUT_DIR))
    has_backtest = os.path.exists(BACKTEST_INPUT_DIR) and bool(os.listdir(BACKTEST_INPUT_DIR))

    if not has_train and not has_backtest:
        print(f"CRITICAL ERROR: Both '{TRAIN_INPUT_DIR}' and '{BACKTEST_INPUT_DIR}' are empty.")
        exit()

    pca_reference_dir = TRAIN_INPUT_DIR if has_train else BACKTEST_INPUT_DIR

    engine = FinancialVectorEngine()
    scaler_model, pca_model = get_or_train_pca(engine, pca_reference_dir)
    transform_all_news_and_save(engine, scaler_model, pca_model)
