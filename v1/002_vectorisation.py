import os
import torch
import pandas as pd
import numpy as np
import pickle
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import glob

TRAIN_INPUT_DIR = "001_ticker_news_train"
BACKTEST_INPUT_DIR = "001_ticker_news_backtest"
TRAIN_OUTPUT_DIR = "002_final_vectors_train"
BACKTEST_OUTPUT_DIR = "002_final_vectors_backtest"

MODEL_NAME = 'ProsusAI/finbert'
BATCH_SIZE = 64            
MAX_TOKENS = 256           
MAX_FILE_SIZE_MB = 800

PCA_COMPONENTS = 28        
PCA_SAMPLE_SIZE = 500000    
PCA_MODEL_PATH = 'pca_model_train.pkl'
SCALER_MODEL_PATH = 'scaler_model_train.pkl'

COLUMN_ALIASES = {
    'date': ['date', 'published_at', 'timestamp', 'time', 'datetime'],
    'title': ['title', 'headline', 'header', 'subject', 'article_title'],
    'body': ['content', 'body', 'article', 'text', 'summary']
}

class FinancialVectorEngine:
    def __init__(self, model_name=MODEL_NAME, batch_size=BATCH_SIZE):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"--- Hardware: Forced {self.device} ---")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Loading FinBERT in FP16 Turbo Mode...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device).eval()
        self.batch_size = batch_size

    def get_embeddings(self, texts):
        all_embeddings, all_sentiments = [], []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(self.device)
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
        title_col = next((cols[a] for a in COLUMN_ALIASES['title'] if a in cols), None)
        body_col = next((cols[a] for a in COLUMN_ALIASES['body'] if a in cols), None)
        date_col = next((cols[a] for a in COLUMN_ALIASES['date'] if a in cols), None)

        if not title_col and not body_col: return None, None

        if title_col and body_col: df['processed_text'] = df[title_col].fillna('') + ". " + df[body_col].fillna('')
        elif title_col: df['processed_text'] = df[title_col].fillna('')
        else: df['processed_text'] = df[body_col].fillna('')
            
        if date_col:
            df['processed_date'] = pd.to_datetime(df[date_col], errors='coerce', utc=True).dt.date
            df = df.dropna(subset=['processed_date'])
        else: return None, None
        
        return df[['processed_date', 'processed_text']], date_col
    except Exception as e: 
        print(f"\n[Read Error] {filepath}: {e}")
        return None, None

def get_or_train_pca(engine, input_dir):
    if os.path.exists(PCA_MODEL_PATH) and os.path.exists(SCALER_MODEL_PATH):
        with open(PCA_MODEL_PATH, 'rb') as f: pca_model = pickle.load(f)
        with open(SCALER_MODEL_PATH, 'rb') as f: scaler_model = pickle.load(f)
        return scaler_model, pca_model

    print("\n=== Fitting PCA on Burn-In Data (2015) ===")
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    sampled_embeddings = []
    samples_per_file = max(10, PCA_SAMPLE_SIZE // len(all_files)) if all_files else 10
    
    # FIX: Loop over all files instead of just [:100] to reach PCA threshold
    for f in tqdm(all_files, desc="Sampling early data for PCA"): 
        df, _ = smart_load_df(f)
        if df is None or df.empty: continue
        
        # Filter for burn-in year only to avoid leakage
        early_df = df[df['processed_date'] < pd.Timestamp("2016-01-01").date()]
        if early_df.empty: continue
            
        sample = early_df.sample(min(len(early_df), 500))['processed_text'].tolist()
        if sample:
            embeddings, _ = engine.get_embeddings(sample)
            sampled_embeddings.append(embeddings)
            if sum(len(x) for x in sampled_embeddings) >= PCA_SAMPLE_SIZE: break
            
    X = np.vstack(sampled_embeddings)
    scaler = StandardScaler()
    pca = PCA(n_components=PCA_COMPONENTS)
    pca.fit(scaler.fit_transform(X))
    
    with open(PCA_MODEL_PATH, 'wb') as f: pickle.dump(pca, f)
    with open(SCALER_MODEL_PATH, 'wb') as f: pickle.dump(scaler, f)
    return scaler, pca

def transform_and_save(engine, scaler, pca, input_dir, output_dir, split_name):
    print(f"\n=== Processing {split_name} ===")
    os.makedirs(output_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    all_files.sort(key=os.path.getsize)
    
    for f in tqdm(all_files, desc=f"Vectorizing {split_name}"):
        ticker = os.path.basename(f).replace("_news.csv", "").upper()
        out_path = os.path.join(output_dir, f"{ticker}_vec.csv")
        
        if os.path.exists(out_path): continue 
            
        file_mb = os.path.getsize(f) / (1024 * 1024)
        if file_mb > MAX_FILE_SIZE_MB:
            print(f"\n[WHALE SHIELD] Skipping {ticker} ({file_mb:.1f} MB)")
            continue
            
        df, _ = smart_load_df(f)
        if df is None or df.empty: continue
        
        try:
            raw_emb, sentiments = engine.get_embeddings(df['processed_text'].tolist())
            reduced_emb = pca.transform(scaler.transform(raw_emb))
            res_df = pd.DataFrame(reduced_emb, columns=[f"dim_{i+1}" for i in range(reduced_emb.shape[1])])
            res_df['sent_pos'], res_df['sent_neg'], res_df['sent_neu'] = sentiments[:, 0], sentiments[:, 1], sentiments[:, 2]
            res_df['date'] = df['processed_date'].values
            res_df.groupby('date').mean().reset_index().to_csv(out_path, index=False)
        except Exception as e: print(f"\n[GPU Error] {ticker}: {e}")
        
        del df, res_df
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    if not os.path.exists(TRAIN_INPUT_DIR) or not os.listdir(TRAIN_INPUT_DIR):
        print(f"CRITICAL ERROR: '{TRAIN_INPUT_DIR}' empty.")
        exit()

    engine = FinancialVectorEngine()
    scaler_model, pca_model = get_or_train_pca(engine, TRAIN_INPUT_DIR)
    transform_and_save(engine, scaler_model, pca_model, TRAIN_INPUT_DIR, TRAIN_OUTPUT_DIR, "TRAIN")
    if os.path.exists(BACKTEST_INPUT_DIR) and os.listdir(BACKTEST_INPUT_DIR):
        transform_and_save(engine, scaler_model, pca_model, BACKTEST_INPUT_DIR, BACKTEST_OUTPUT_DIR, "BACKTEST")
