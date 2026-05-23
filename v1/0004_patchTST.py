import glob
import json
import os
import copy

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "0003_ready_20")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "0004_saved_models_20")
PATCHTST_MODEL_FILE = "patchtst_live_20.pth"
PATCHTST_TARGET_STATS_FILE = "patchtst_target_stats.json"
EPOCHS = 30
EARLY_STOP_PATIENCE = 10
MIN_DELTA = 1e-4
BATCH_SIZE = 512
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


class PatchTST(nn.Module):
    def __init__(self, num_features, seq_len, patch_len=10, stride=10, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super(PatchTST, self).__init__()
        self.seq_len, self.patch_len, self.stride = seq_len, patch_len, stride
        self.num_patches = int((seq_len - patch_len) / stride) + 1
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers=num_layers,
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.head = nn.Sequential(
            nn.Linear(num_features * self.num_patches * d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        b, s, f = x.shape
        x = x.permute(0, 2, 1).reshape(b * f, s)
        enc_in = self.patch_embedding(x.unfold(1, self.patch_len, self.stride)) + self.pos_embedding
        return self.head(self.flatten(self.encoder(enc_in).reshape(b, f, self.num_patches, -1)))


class BatchedDataset(Dataset):
    def __init__(self, npz_file, target_mean=0.0, target_std=1.0):
        data = np.load(npz_file)
        self.X = torch.from_numpy(data["X"]).float()
        raw_y = torch.clamp(torch.from_numpy(data["y"]).float(), min=-50.0, max=50.0)
        self.y = (raw_y - target_mean) / target_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    print("--- NO-DEEPSEEK PatchTST TRAINING (LIVE) ---")
    print(f"Hyperparams: batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, min_delta={MIN_DELTA}")
    batch_files = sorted(glob.glob(os.path.join(DATA_DIR, "train_batch_nodis_*.npz")))
    if not batch_files:
        exit(f"ERROR: No batches found in {DATA_DIR}")

    if len(batch_files) < 2:
        train_files = batch_files
        val_files = []
        print("WARNING: Not enough batches for validation split; early stopping is disabled.")
    else:
        split_idx = max(1, int(len(batch_files) * 0.8))
        train_files = batch_files[:split_idx]
        val_files = batch_files[split_idx:]
        if not val_files:
            val_files = train_files[-1:]
            train_files = train_files[:-1] or train_files
        print(f"Train batches: {len(train_files)} | Val batches: {len(val_files)}")

    def collect_target_stats(files):
        total = 0
        target_sum = 0.0
        target_sumsq = 0.0
        for file in files:
            try:
                data = np.load(file)
                raw_y = np.clip(data["y"].astype(np.float32), -50.0, 50.0)
                total += raw_y.size
                target_sum += float(raw_y.sum())
                target_sumsq += float(np.square(raw_y).sum())
            except Exception:
                continue
        if total == 0:
            return 0.0, 1.0
        mean = target_sum / total
        variance = max(target_sumsq / total - mean * mean, 1e-8)
        return mean, float(np.sqrt(variance))

    target_mean, target_std = collect_target_stats(train_files)
    with open(os.path.join(MODEL_SAVE_DIR, PATCHTST_TARGET_STATS_FILE), "w", encoding="utf-8") as f:
        json.dump({"mean": target_mean, "std": target_std}, f, indent=2)
    print(f"Target normalization: mean={target_mean:.6f} std={target_std:.6f}")
    
    first_batch = np.load(batch_files[0])
    model = PatchTST(num_features=first_batch["X"].shape[2], seq_len=first_batch["X"].shape[1]).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=1, min_lr=1e-6)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    def evaluate(files):
            model.eval()
            total_loss, batch_count = 0.0, 0
            sample_pairs = []
            with torch.no_grad():
                for file in files:
                    try:
                        loader = DataLoader(
                            BatchedDataset(file, target_mean=target_mean, target_std=target_std),
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            drop_last=False,
                        )
                        for X, y in loader:
                            preds = model(X.to(DEVICE))
                            loss = criterion(preds, y.to(DEVICE).view(-1, 1))
                            if torch.isnan(loss):
                                continue
                            total_loss += loss.item()
                            batch_count += 1
                            # capture a few sample (de-normalized) pred/true pairs for quick inspection
                            if len(sample_pairs) < 5:
                                p = preds.cpu().numpy().ravel()
                                t = y.numpy().ravel()
                                for pi, ti in zip(p, t):
                                    sample_pairs.append((float(pi * target_std + target_mean), float(ti * target_std + target_mean)))
                                    if len(sample_pairs) >= 5:
                                        break
                    except Exception:
                        continue
            val_loss = total_loss / max(1, batch_count)
            val_de_rmse = math.sqrt(val_loss) * target_std
            return val_loss, val_de_rmse, sample_pairs

    for epoch in range(EPOCHS):
        model.train()
        total_loss, batch_count = 0, 0
        np.random.shuffle(train_files)

        pbar = tqdm(train_files, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for file in pbar:
            try:
                loader = DataLoader(
                    BatchedDataset(file, target_mean=target_mean, target_std=target_std),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=True,
                )
                for X, y in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(X.to(DEVICE)), y.to(DEVICE).view(-1, 1))
                    if torch.isnan(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    batch_count += 1
                pbar.set_postfix({"loss": f"{loss.item():.5f}"})
            except Exception:
                continue

        avg_loss = total_loss / max(1, batch_count)
        train_de_rmse = math.sqrt(avg_loss) * target_std
        if val_files:
            val_loss_norm, val_de_rmse, val_samples = evaluate(val_files)
        else:
            val_loss_norm, val_de_rmse, val_samples = avg_loss, math.sqrt(avg_loss) * target_std, []

        scheduler.step(val_loss_norm)

        if val_files:
            print(f"Epoch {epoch+1}: train_loss_norm={avg_loss:.5f} train_RMSE={train_de_rmse:.4f} | val_loss_norm={val_loss_norm:.5f} val_RMSE={val_de_rmse:.4f}")
        else:
            print(f"Epoch {epoch+1}: train_loss_norm={avg_loss:.5f} train_RMSE={train_de_rmse:.4f}")

        # show a few sample de-normalized predictions vs true values for quick inspection
        if val_samples:
            print("Sample val preds (pred, true):")
            for p, t in val_samples[:5]:
                print(f"  {p:.4f}, {t:.4f}")

        if val_loss_norm < best_loss - MIN_DELTA:
            best_loss = val_loss_norm
            best_state = copy.deepcopy(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())
            torch.save(best_state, os.path.join(MODEL_SAVE_DIR, PATCHTST_MODEL_FILE))
            epochs_without_improvement = 0
            print(f"Saved best model with loss {val_loss_norm:.5f} (RMSE {val_de_rmse:.4f})")
        else:
            epochs_without_improvement += 1
            if val_files and epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs. Best val loss: {best_loss:.5f}")
                break

    if best_state is not None:
        torch.save(best_state, os.path.join(MODEL_SAVE_DIR, PATCHTST_MODEL_FILE))
