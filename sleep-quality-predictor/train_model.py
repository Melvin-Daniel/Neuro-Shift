"""
Train and compare LR, RF, DT, SVM on Sleep Health and Lifestyle data.
Saves best model pipeline + metadata for the Streamlit app.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Sleep_health_and_lifestyle_dataset.csv")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
DEMO_PATH = os.path.join(os.path.dirname(__file__), "data", "demo_sleep_data.csv")

REQUIRED_COLS = [
    "Quality of Sleep",
    "Sleep Duration",
    "Physical Activity Level",
    "Stress Level",
]


def parse_blood_pressure(series: pd.Series) -> pd.DataFrame:
    systolic: list[float] = []
    diastolic: list[float] = []

    for v in series:
        if pd.isna(v) or not isinstance(v, str):
            systolic.append(np.nan)
            diastolic.append(np.nan)
            continue
        parts = v.replace(" ", "").split("/")
        if len(parts) != 2:
            systolic.append(np.nan)
            diastolic.append(np.nan)
            continue
        try:
            systolic.append(float(int(parts[0])))
            diastolic.append(float(int(parts[1])))
        except ValueError:
            systolic.append(np.nan)
            diastolic.append(np.nan)

    return pd.DataFrame({"BP_Systolic": systolic, "BP_Diastolic": diastolic})


def load_and_prepare(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"CSV missing columns: {missing}. Expected Kaggle 'Sleep Health and Lifestyle Dataset'."
        )

    y_raw = df["Quality of Sleep"].astype(float)
    # Three-way split: Poor <=5, Average 6-7, Good >=8
    def bin_quality(v: float) -> str:
        if v <= 5:
            return "Poor"
        if v <= 7:
            return "Average"
        return "Good"

    y = y_raw.map(bin_quality)
    if y.nunique() < 2:
        raise SystemExit("Target has fewer than 2 classes after binning; check data.")

    X = df.drop(columns=["Quality of Sleep"], errors="ignore")
    if "Person ID" in X.columns:
        X = X.drop(columns=["Person ID"])

    bp = parse_blood_pressure(X["Blood Pressure"]) if "Blood Pressure" in X.columns else None
    if bp is not None:
        X = X.drop(columns=["Blood Pressure"])
        X = pd.concat([X.reset_index(drop=True), bp.reset_index(drop=True)], axis=1)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    for c in cat_cols:
        X[c] = X[c].astype(str).replace("nan", "Unknown").fillna("Unknown")

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = X[c].median()
        X[c] = X[c].fillna(med)

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers = [
        ("num", StandardScaler(), num_cols),
        ("cat", _one_hot_encoder(), cat_cols),
    ]
    return ColumnTransformer(transformers)


def make_models() -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
    }


def generate_demo_csv(path: str, n: int = 500) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    occupations = ["Software Engineer", "Doctor", "Artist", "Teacher", "Nurse"]
    genders = ["Male", "Female"]
    bmi = ["Normal", "Overweight", "Obese"]
    disorder = ["None", "Insomnia", "Sleep Apnea"]

    rows = []
    for _ in range(n):
        sleep_dur = rng.uniform(4.0, 9.5)
        activity = rng.integers(20, 100)
        stress = rng.integers(3, 10)
        # Higher quality correlates with more sleep, more activity, lower stress
        score = (
            (sleep_dur - 4) / 5.5 * 3
            + (activity - 20) / 80 * 2
            + (10 - stress) / 7 * 3
            + rng.normal(0, 0.8)
        )
        qos = int(np.clip(round(4 + score), 4, 9))
        sys_bp = int(rng.integers(110, 145))
        dia = int(rng.integers(70, 95))
        rows.append(
            {
                "Person ID": len(rows) + 1,
                "Gender": rng.choice(genders),
                "Age": int(rng.integers(25, 65)),
                "Occupation": rng.choice(occupations),
                "Sleep Duration": round(sleep_dur, 1),
                "Quality of Sleep": qos,
                "Physical Activity Level": int(activity),
                "Stress Level": int(stress),
                "BMI Category": rng.choice(bmi),
                "Blood Pressure": f"{sys_bp}/{dia}",
                "Heart Rate": int(rng.integers(65, 95)),
                "Daily Steps": int(rng.integers(3000, 12000)),
                "Sleep Disorder": rng.choice(disorder),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote demo dataset: {path} ({n} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help="Path to Sleep_health_and_lifestyle_dataset.csv",
    )
    parser.add_argument(
        "--generate-demo",
        action="store_true",
        help=f"Write synthetic CSV to {DEMO_PATH} and exit",
    )
    args = parser.parse_args()

    if args.generate_demo:
        os.makedirs(os.path.dirname(DEMO_PATH), exist_ok=True)
        generate_demo_csv(DEMO_PATH)
        return

    csv_path = args.data
    if not os.path.isfile(csv_path):
        print(
            f"Data file not found: {csv_path}\n"
            "Download 'Sleep Health and Lifestyle Dataset' from Kaggle into data/, or run:\n"
            "  python train_model.py --generate-demo\n"
            "  python train_model.py --data data/demo_sleep_data.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    X, y = load_and_prepare(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    results = {}
    best_name = None
    best_f1 = -1.0
    best_pipe = None

    for name, estimator in make_models().items():
        pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        results[name] = {"accuracy": float(acc), "macro_f1": float(macro_f1)}
        print(f"{name}: accuracy={acc:.4f} macro_f1={macro_f1:.4f}")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = name
            best_pipe = pipe

    assert best_pipe is not None
    pred_best = best_pipe.predict(X_test)
    print("\nBest model:", best_name)
    print(classification_report(y_test, pred_best, zero_division=0))

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    joblib.dump(best_pipe, model_path)

    # Defaults for app inference (features not on main form)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    default_numeric = {c: float(X[c].median()) for c in num_cols}
    default_categorical = {c: str(X[c].mode().iloc[0]) if len(X[c].mode()) else "Unknown" for c in cat_cols}

    meta = {
        "best_model_name": best_name,
        "metrics": results,
        "test_classification_report": classification_report(
            y_test, pred_best, output_dict=True, zero_division=0
        ),
        "target_binning": "Poor: Quality of Sleep <= 5; Average: 6-7; Good: >= 8",
        "feature_columns": list(X.columns),
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "default_numeric": default_numeric,
        "default_categorical": default_categorical,
        "class_order": ["Poor", "Average", "Good"],
    }
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved pipeline -> {model_path}")
    print(f"Saved metadata -> {os.path.join(ARTIFACTS_DIR, 'metadata.json')}")


if __name__ == "__main__":
    main()
