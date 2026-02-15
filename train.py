import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


RANDOM_STATE = 42
DATA_PATH = "Digital_Payment_Fraud_Detection_Dataset.csv"
TARGET_COL = "fraud_label"
DROP_COLS = ["transaction_id", "user_id"]


def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)


def compute_binary_auc(y_true_enc, proba_2d):
    # y_true_enc must be 0/1, proba_2d shape (n,2)
    return roc_auc_score(y_true_enc, proba_2d[:, 1])


def main():
    ensure_dirs()

    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {DATA_PATH}")

    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    y_raw = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Infer types (after dropping ids)
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Encode target to 0..K-1 (here should be binary)
    y_le = LabelEncoder()
    y = y_le.fit_transform(y_raw)
    class_names = y_le.classes_.tolist()

    if len(class_names) != 2:
        raise ValueError(f"Expected binary target, got classes={class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Create a validation split from training data for threshold selection (imbalanced dataset)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    # Save a holdout test file for Streamlit demo (includes true labels)
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_le.inverse_transform(y_test)
    test_df.to_csv("reports/test_data.csv", index=False)

    # Baseline (majority class)
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    print("\nDummy (most_frequent) baseline:")
    print(" accuracy =", float(accuracy_score(y_test, dummy_pred)))
    print(
        " f1 =",
        float(f1_score(y_test, dummy_pred, pos_label=1, zero_division=0)),
    )
    print(
        " mcc =",
        float(matthews_corrcoef(y_test, dummy_pred)),
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    # For XGBoost imbalance handling
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / max(1, n_pos))

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "k-Nearest Neighbor": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        ),
    }

    results = []
    fitted = {}
    thresholds = {}

    def pick_threshold(y_true, pos_proba):
        # Grid-search a threshold that maximizes F1; tie-break by MCC.
        best = {"thr": 0.5, "f1": -1.0, "mcc": -2.0}
        for thr in np.linspace(0.05, 0.95, 19):
            pred = (pos_proba >= thr).astype(int)
            f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)
            mcc = matthews_corrcoef(y_true, pred)
            if (f1 > best["f1"]) or (f1 == best["f1"] and mcc > best["mcc"]):
                best = {"thr": float(thr), "f1": float(f1), "mcc": float(mcc)}
        return best["thr"]

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
        pipe.fit(X_fit, y_fit)

        # Choose a threshold on validation set
        val_proba = pipe.predict_proba(X_val)
        thr = pick_threshold(y_val, val_proba[:, 1])
        thresholds[name] = thr

        proba = pipe.predict_proba(X_test)
        auc = compute_binary_auc(y_test, proba)
        y_pred = (proba[:, 1] >= thr).astype(int)

        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "auc": float(auc),
            "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
        }

        results.append(row)
        fitted[name] = pipe

    metrics_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
    metrics_df.to_csv("reports/metrics.csv", index=False)
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    artifact = {
        "data_path": DATA_PATH,
        "target_col": TARGET_COL,
        "drop_cols": DROP_COLS,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "class_names": class_names,
        "label_encoder": y_le,
        "models": fitted,
        "thresholds": thresholds,
    }
    joblib.dump(artifact, "models/models.joblib")

    print("\nMetrics on holdout test split:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()


