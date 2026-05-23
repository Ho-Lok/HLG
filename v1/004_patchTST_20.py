import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import copy
from tqdm import tqdm

DATA_DIR = "003_ready_20" 
MODEL_SAVE_DIR = "004_saved_models_20"
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
MIN_DELTA = 1e-4
BATCH_SIZE = 512
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

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

class BatchedDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.X = torch.from_numpy(data['X']).float()
        self.y = torch.clamp(torch.from_numpy(data['y']).float(), min=-50.0, max=50.0) 
        
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

if __name__ == "__main__":
    print(f"--- NO-DEEPSEEK PatchTST TRAINING ---")
    batch_files = sorted(glob.glob(os.path.join(DATA_DIR, "train_batch_nodis_*.npz")))
    if not batch_files: exit(f"ERROR: No batches found in {DATA_DIR}")

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
    
    first_batch = np.load(batch_files[0])
    model = PatchTST(num_features=first_batch['X'].shape[2], seq_len=first_batch['X'].shape[1]).to(DEVICE)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    criterion = nn.HuberLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0

    def evaluate(files):
        model.eval()
        total_loss, batch_count = 0.0, 0
        with torch.no_grad():
            for file in files:
                try:
                    loader = DataLoader(BatchedDataset(file), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
                    for X, y in loader:
                        loss = criterion(model(X.to(DEVICE)), y.to(DEVICE).view(-1, 1))
                        if torch.isnan(loss):
                            continue
                        total_loss += loss.item()
                        batch_count += 1
                except Exception:
                    continue
        return total_loss / max(1, batch_count)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, batch_count = 0, 0
        np.random.shuffle(train_files)
        
        pbar = tqdm(train_files, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for file in pbar:
            try:
                loader = DataLoader(BatchedDataset(file), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
                for X, y in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(X.to(DEVICE)), y.to(DEVICE).view(-1, 1))
                    if torch.isnan(loss): continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    batch_count += 1
                pbar.set_postfix({'loss': f'{loss.item():.5f}'})
            except Exception: continue
                
        avg_loss = total_loss / max(1, batch_count)
        val_loss = evaluate(val_files) if val_files else avg_loss
        scheduler.step(val_loss)

        if val_files:
            print(f"Epoch {epoch+1}: train_loss={avg_loss:.5f} val_loss={val_loss:.5f}")
        else:
            print(f"Epoch {epoch+1}: train_loss={avg_loss:.5f}")

        if val_loss < best_loss - MIN_DELTA:
            best_loss = val_loss
            best_state = copy.deepcopy(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())
            torch.save(best_state, os.path.join(MODEL_SAVE_DIR, "patchtst_20.pth"))
            epochs_without_improvement = 0
            print(f"Saved best model with loss {val_loss:.5f}")
        else:
            epochs_without_improvement += 1
            if val_files and epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs. Best val loss: {best_loss:.5f}")
                break

    if best_state is not None:
        torch.save(best_state, os.path.join(MODEL_SAVE_DIR, "patchtst_20.pth"))
