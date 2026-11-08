"""Train a small Keras 3-class classifier on data/features.csv
(rows logged by `python driver_monitor.py --log --label 0|1|2`).
Classes: 0 = normal, 1 = distracted, 2 = fatigue. Saves fatigue_model.keras."""
import csv
import os
import sys

import numpy as np
import tensorflow as tf

from features import FEATURES

NAMES = ["normal", "distracted", "fatigue"]
N_CLASSES = len(NAMES)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "data", "features.csv")
OUT_PATH = os.path.join(HERE, "fatigue_model.keras")
tf.keras.utils.set_random_seed(0)

if not os.path.exists(CSV_PATH):
    sys.exit("No data/features.csv - record first: python driver_monitor.py --log --label 0|1|2")

sessions = {}
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        label = int(r["label"])
        if label not in range(N_CLASSES):
            sys.exit(f"Found label {label} in the CSV. Labels must be 0, 1 or 2 - "
                     "delete the old data/features.csv (old label 1 meant drowsy) and re-record.")
        sessions.setdefault(r["session"], []).append(([float(r[k]) for k in FEATURES], label))

# Time-ordered split per session: first 80% trains, last 20% validates
train, val = [], []
for rows in sessions.values():
    cut = int(len(rows) * 0.8)
    train += rows[:cut]
    val += rows[cut:]


def to_xy(rows):
    return (np.array([r[0] for r in rows], dtype=np.float32),
            np.array([r[1] for r in rows], dtype=np.int32))


Xtr, ytr = to_xy(train)
Xva, yva = to_xy(val)
counts = [int((ytr == c).sum()) for c in range(N_CLASSES)]
print(f"{len(sessions)} sessions | train {len(ytr)} ("
      + ", ".join(f"{n} {c}" for n, c in zip(NAMES, counts)) + f") | val {len(yva)}")
if min(counts) == 0 or len(yva) < 10:
    sys.exit("Need rows for all three labels (0, 1, 2) and more data - record more sessions.")

norm = tf.keras.layers.Normalization()      # scaling is stored inside the model
norm.adapt(Xtr)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(len(FEATURES),)),
    norm,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(N_CLASSES, activation="softmax"),   # P(normal), P(distracted), P(fatigue)
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cw = {c: len(ytr) / (N_CLASSES * counts[c]) for c in range(N_CLASSES)}   # balance classes
model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=100, batch_size=64, class_weight=cw,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
          verbose=2)

pred = model.predict(Xva, verbose=0).argmax(axis=1)
cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for t, p in zip(yva, pred):
    cm[t, p] += 1
print(f"Val accuracy {np.trace(cm) / len(yva):.3f}")
for c, n in enumerate(NAMES):
    print(f"{n:>10}: precision {cm[c, c] / max(cm[:, c].sum(), 1):.3f} "
          f"| recall {cm[c, c] / max(cm[c].sum(), 1):.3f}")
print("Confusion (rows = true, cols = predicted; order normal/distracted/fatigue):")
print(cm)
model.save(OUT_PATH)
print("Saved", OUT_PATH, "(val rows share sessions with training, so treat accuracy as optimistic)")
