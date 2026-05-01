# -*- coding: utf-8 -*-
"""
Train LSTM Gesture Classifier
==============================
Load data da thu thap tu data_collector.py va train LSTM.

Chay:
  python train_lstm.py

Output:
  models/gesture_lstm.keras   <- LSTM model
  models/gesture_classes.npy  <- class list
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data", "sequences")
MODEL_DIR  = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSES = ["idle", "forward", "forward_left", "handbrake", "reverse"]
SEQ_LEN  = 30
FEATURES = 63

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n=== LOADING DATA ===")
X, y = [], []

for label_idx, cls_name in enumerate(CLASSES):
    cls_dir = os.path.join(DATA_DIR, cls_name)
    if not os.path.exists(cls_dir):
        print(f"  [SKIP] {cls_name}: thu muc khong ton tai")
        continue
    files = [f for f in os.listdir(cls_dir) if f.endswith(".npy")]
    print(f"  {cls_name:20s}: {len(files)} sequences")
    for f in files:
        seq = np.load(os.path.join(cls_dir, f))
        if seq.shape == (SEQ_LEN, FEATURES):
            X.append(seq)
            y.append(label_idx)

if len(X) == 0:
    print("\n[ERROR] Khong co du lieu! Chay data_collector.py truoc.")
    sys.exit(1)

X = np.array(X)
y = np.array(y)
print(f"\n  Tong cong: {len(X)} sequences, {len(CLASSES)} classes")

# ── Split ─────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}")

# ── Build LSTM model ──────────────────────────────────────────────────────────
print("\n=== BUILDING MODEL ===")
import tensorflow as tf
from tensorflow.keras.models import Sequential         # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint         # type: ignore

model = Sequential([
    LSTM(64, return_sequences=True, activation='tanh',
         input_shape=(SEQ_LEN, FEATURES)),
    Dropout(0.3),
    LSTM(32, activation='tanh'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(CLASSES), activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n=== TRAINING ===")
BEST_MODEL = os.path.join(MODEL_DIR, "gesture_lstm_best.keras")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_MODEL, monitor='val_accuracy',
                    save_best_only=True, verbose=0),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=16,
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n=== EVALUATION ===")
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"  Val Accuracy : {acc*100:.2f}%")
print(f"  Val Loss     : {loss:.4f}")

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
print("\n  Classification Report:")
print(classification_report(y_val, y_pred, target_names=CLASSES))

# ── Save ──────────────────────────────────────────────────────────────────────
FINAL_MODEL  = os.path.join(MODEL_DIR, "gesture_lstm.keras")
CLASSES_FILE = os.path.join(MODEL_DIR, "gesture_classes.npy")

model.save(FINAL_MODEL)
np.save(CLASSES_FILE, np.array(CLASSES))

print(f"\n[SAVED] Model    : {FINAL_MODEL}")
print(f"[SAVED] Classes  : {CLASSES_FILE}")
print("\nChay lai hand_tracking.py de su dung model moi!")
