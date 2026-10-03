"""
V4 training.

Run after preprocess_dataset_v4.py:
    python src/train_v4.py

Saves models/best_v4.pth containing the weights AND the label list,
so realtime_predict_v4.py can't get out of sync with training.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from features_v4 import (
    PROJECT_ROOT, SEQUENCE_LENGTH, FEATURE_DIM,
    prepare_sequence, resample, mirror, to_model_features,
)
from model_v4 import SignGRU


PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed_v4")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_v4.pth")

MIN_SAMPLES = 5
EPOCHS = 120
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# LOAD
# ============================================================

labels = []
for d in sorted(os.listdir(PROCESSED_DIR)):
    folder = os.path.join(PROCESSED_DIR, d)
    if os.path.isdir(folder) and not d.startswith("."):
        n = len([f for f in os.listdir(folder) if f.endswith(".npy")])
        if n >= MIN_SAMPLES:
            labels.append(d)
        else:
            print(f"Skipping '{d}' (only {n} videos)")

label_to_id = {label: i for i, label in enumerate(labels)}

X, y = [], []
for label in labels:
    folder = os.path.join(PROCESSED_DIR, label)
    for f in sorted(os.listdir(folder)):
        if f.endswith(".npy"):
            raw = np.load(os.path.join(folder, f))
            if raw.ndim == 2 and raw.shape[1] == 225 and np.isfinite(raw).all():
                X.append(prepare_sequence(raw))
                y.append(label_to_id[label])

X = np.stack(X)
y = np.array(y)
print(f"\n{len(labels)} classes, {len(X)} samples")

X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.15, stratify=y_trval, random_state=SEED)
print("train", len(X_train), "| val", len(X_val), "| test", len(X_test))


# ============================================================
# AUGMENTATION (on raw landmarks, before feature conversion)
# ============================================================

def augment(seq):
    # 1. mirror left/right (left-handed signers, webcam vs dataset)
    if np.random.rand() < 0.5:
        seq = mirror(seq)

    # 2. speed change: random crop at both ends, stretch back
    n = len(seq)
    a = np.random.randint(0, 4)
    b = n - np.random.randint(0, 4)
    seq = resample(seq[a:b])

    # 3. small rotation, scale and shift (camera angle / distance)
    s = seq.reshape(SEQUENCE_LENGTH, 75, 3).copy()
    mask = np.abs(s).sum(axis=2, keepdims=True) > 0
    th = np.random.uniform(-0.2, 0.2)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    s[:, :, :2] = (s[:, :, :2] @ rot.T) * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.15, 0.15, 2)
    s = s * mask  # keep missing hands as zeros
    return s.reshape(SEQUENCE_LENGTH, 225).astype(np.float32)


def batch_features(seqs, train):
    return torch.tensor(np.stack([
        to_model_features(augment(s) if train else s) for s in seqs
    ])).to(device)


def evaluate(model, Xs, ys):
    model.eval()
    with torch.no_grad():
        pred = model(batch_features(Xs, False)).argmax(1).cpu().numpy()
    return (pred == ys).mean() * 100, pred


# ============================================================
# TRAIN
# ============================================================

model = SignGRU(FEATURE_DIM, len(labels)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
steps = EPOCHS * ((len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, LEARNING_RATE, total_steps=steps)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_val = -1
for epoch in range(EPOCHS):
    model.train()
    perm = np.random.permutation(len(X_train))
    total_loss = 0
    for i in range(0, len(perm), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb = batch_features(X_train[idx], True)
        yb = torch.tensor(y_train[idx]).to(device)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(idx)

    val_acc, _ = evaluate(model, X_val, y_val)
    print(f"Epoch {epoch + 1:3d}/{EPOCHS} | loss {total_loss / len(X_train):.3f} | val {val_acc:.1f}%")

    # >= so later (better-trained) epochs win ties on a small val set
    if val_acc >= best_val:
        best_val = val_acc
        torch.save({"state_dict": model.state_dict(), "labels": labels}, MODEL_PATH)


# ============================================================
# TEST
# ============================================================

ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["state_dict"])
test_acc, pred = evaluate(model, X_test, y_test)

print(f"\nBest val accuracy: {best_val:.1f}%")
print(f"TEST accuracy:     {test_acc:.1f}%")

print("\nPer-class test accuracy:")
for i, label in enumerate(labels):
    m = y_test == i
    if m.any():
        print(f"  {label:12s} {(pred[m] == i).mean() * 100:5.0f}%  ({m.sum()} samples)")

with open(os.path.join(PROJECT_ROOT, "models", "labels_v4.json"), "w") as f:
    json.dump(label_to_id, f, indent=4)
print("\nSaved:", MODEL_PATH)
