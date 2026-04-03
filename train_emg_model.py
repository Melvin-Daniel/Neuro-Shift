"""
Train RandomForest on windowed EMG CSVs (must match live_emg_demo + window_size=100).

Data layout (single AD8232, legacy):
  timestamp,signal

Data layout (two AD8232 — recommended):
  timestamp,signal_jaw,signal_brow

  data/<user_id>/rest.csv, jaw_clench.csv, eyebrow_raise.csv — all files in a folder must use the same format.

Training-only: WINDOW_STRIDE < WINDOW_SIZE uses overlapping windows (more examples from each CSV).
Live demo still classifies any 100-sample window the same way.

Accuracy tips (data > tuning):
  - jaw_clench: record more minutes; match live clench strength.
  - rest: include talking, chewing, near-clench motion (reduces false jaw).
  - eyebrow_raise: clear held raises; keep electrode placement fixed.

Run: python train_emg_model.py
"""
from __future__ import annotations

import pickle
import os
from collections import Counter
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

WINDOW_SIZE = 100
# Stride between training windows. 50 = overlapping (more samples; adjacent windows correlate).
# 100 = no overlap (fewer samples; holdout scores are less optimistic). Tune if needed.
WINDOW_STRIDE = 50

GESTURE_FILES: list[tuple[str, str]] = [
    ("rest.csv", "rest"),
    ("jaw_clench.csv", "jaw_clench"),
    ("eyebrow_raise.csv", "eyebrow_raise"),
]

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=24,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


def extract_features(signal: np.ndarray) -> list[float]:
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    peak = float(np.max(signal) - np.min(signal))
    abs_mean = float(np.mean(np.abs(signal)))
    energy = float(np.sum(signal**2))
    zero_cross = float(np.sum(np.diff(np.sign(signal)) != 0))
    return [mean, std, peak, abs_mean, energy, zero_cross]


def extract_features_dual(jaw: np.ndarray, brow: np.ndarray) -> list[float]:
    return extract_features(jaw) + extract_features(brow)


def read_gesture_csv(filename: Path) -> pd.DataFrame:
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read().replace("\r", "\n")
    df = pd.read_csv(StringIO(raw), on_bad_lines="skip")
    if "signal_jaw" in df.columns and "signal_brow" in df.columns:
        df["signal_jaw"] = pd.to_numeric(df["signal_jaw"], errors="coerce")
        df["signal_brow"] = pd.to_numeric(df["signal_brow"], errors="coerce")
        return df.dropna(subset=["signal_jaw", "signal_brow"])
    if "signal" not in df.columns:
        if len(df.columns) == 1:
            df = df.rename(columns={df.columns[0]: "signal"})
        elif "value" in df.columns:
            df = df.rename(columns={"value": "signal"})
    df["signal"] = pd.to_numeric(df.get("signal"), errors="coerce")
    df = df.dropna(subset=["signal"])
    return df


def csv_is_dual(path: Path) -> bool:
    df = read_gesture_csv(path)
    return "signal_jaw" in df.columns and "signal_brow" in df.columns


def windows_from_csv(path: Path, label: str, user_id: str) -> tuple[list, list, list]:
    df = read_gesture_csv(path)
    Xw, yw, gw = [], [], []
    if "signal_jaw" in df.columns and "signal_brow" in df.columns:
        va = df["signal_jaw"].values.astype(float)
        vb = df["signal_brow"].values.astype(float)
        n = min(len(va), len(vb))
        for start in range(0, n - WINDOW_SIZE + 1, WINDOW_STRIDE):
            wa = va[start : start + WINDOW_SIZE]
            wb = vb[start : start + WINDOW_SIZE]
            Xw.append(extract_features_dual(wa, wb))
            yw.append(label)
            gw.append(user_id)
    else:
        values = df["signal"].values
        for start in range(0, len(values) - WINDOW_SIZE + 1, WINDOW_STRIDE):
            window = values[start : start + WINDOW_SIZE]
            Xw.append(extract_features(window))
            yw.append(label)
            gw.append(user_id)
    return Xw, yw, gw


def discover_user_sources() -> list[tuple[str, Path]]:
    from_data: list[tuple[str, Path]] = []

    if DATA_DIR.is_dir():
        only_user = (os.environ.get("EMG_ONLY_USER", "") or "").strip()
        only_user = only_user.lower() if only_user else ""
        for sub in sorted(DATA_DIR.iterdir()):
            if not sub.is_dir():
                continue
            if sub.name.startswith("."):
                continue
            if only_user and sub.name.lower() != only_user:
                continue
            ok = all((sub / name).is_file() for name, _ in GESTURE_FILES)
            if ok:
                from_data.append((sub.name, sub))

    if from_data:
        return from_data

    root_ok = all((ROOT / name).is_file() for name, _ in GESTURE_FILES)
    if root_ok:
        return [("root", ROOT)]

    return []


def main() -> int:
    sources = discover_user_sources()
    if not sources:
        print("No training data found.")
        print("  Add data/<name>/ with rest.csv, jaw_clench.csv, eyebrow_raise.csv")
        print("  Tip: set EMG_ONLY_USER=<name> to train a specific user folder.")
        return 1

    X: list = []
    y: list = []
    groups: list = []
    user_feature_dims: dict[str, int] = {}
    expected_dim: int | None = None

    for user_id, folder in sources:
        dual_modes: list[bool] = []
        for fname, label in GESTURE_FILES:
            path = folder / fname
            dual_modes.append(csv_is_dual(path))
        if any(dual_modes) and not all(dual_modes):
            print(
                f"Error: {folder} mixes single-column and dual-column CSVs. "
                "Re-record all gestures with collect_data.py (two channels) or all legacy."
            )
            return 1
        is_dual = all(dual_modes)
        X_user: list = []
        y_user: list = []
        g_user: list = []
        for fname, label in GESTURE_FILES:
            path = folder / fname
            xw, yw, gw = windows_from_csv(path, label, user_id)
            X_user.extend(xw)
            y_user.extend(yw)
            g_user.extend(gw)

        if not X_user:
            print(f"Warning: {user_id!r} produced 0 windows; skipping.")
            continue

        dim = len(X_user[0])
        user_feature_dims[user_id] = dim
        if expected_dim is None:
            expected_dim = dim
        if dim != expected_dim:
            print("\nError: mixed feature dimensions across users/folders.")
            for uid, d in sorted(user_feature_dims.items()):
                fmt = "dual (jaw+brow)" if d >= 12 else "single-channel"
                print(f"  - {uid}: {fmt} ({d} features)")
            print("\nFix options:")
            print("  1) Re-record the older user(s) so everyone is dual-channel (timestamp,jaw,brow).")
            print("  2) Train only one user folder by setting: EMG_ONLY_USER=<user>")
            print("     Example (PowerShell): $env:EMG_ONLY_USER='vijay'; python train_emg_model.py")
            return 1

        X.extend(X_user)
        y.extend(y_user)
        groups.extend(g_user)
        fmt = "dual (jaw+brow)" if is_dual else "single-channel"
        print(f"Loaded user {user_id!r} from {folder} ({fmt})")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)

    print(
        f"\nWindowing: size={WINDOW_SIZE}, stride={WINDOW_STRIDE} (training only; matches live window size)"
    )
    print("Class distribution:", Counter(y))
    print("Users:", sorted(set(groups)), "total windows:", len(y))

    user_ids = np.unique(groups)
    if len(user_ids) >= 2:
        print("\n--- Leave-one-user-out (generalization check) ---")
        gkf = GroupKFold(n_splits=len(user_ids))
        for _, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            holdout = groups[test_idx][0]
            clf_cv = make_rf()
            clf_cv.fit(X[train_idx], y[train_idx])
            pred = clf_cv.predict(X[test_idx])
            print(f"\nHeld-out user: {holdout}")
            print(classification_report(y[test_idx], pred, zero_division=0))
    else:
        print("\n(Only one user — skipping leave-one-user-out.)")

    print("\n--- 5-fold CV (macro F1, full data) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(make_rf(), X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    print(f"macro F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f}) per fold: {cv_scores.round(3)}")

    print("\n--- Random 70/30 split (pooled) ---")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
    clf = make_rf()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred, zero_division=0))

    print("\n--- Final model: trained on ALL windows ---")
    clf_final = make_rf()
    clf_final.fit(X, y)
    out_path = ROOT / "emg_rf_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(clf_final, f)
    print(f"Saved {out_path} ({len(y)} windows from {len(user_ids)} user(s)).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
