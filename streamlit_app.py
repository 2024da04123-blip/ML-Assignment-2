import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


st.set_page_config(page_title="Fraud Detection — Streamlit App", layout="wide")


def compute_binary_auc(y_true_enc, proba_2d):
    return roc_auc_score(y_true_enc, proba_2d[:, 1])


@st.cache_resource
def load_artifact():
    return joblib.load("models/models.joblib")


def main():
    st.title("Digital Payment Fraud Detection — Model Comparator")

    artifact = load_artifact()
    models = artifact["models"]
    target_col = artifact["target_col"]
    drop_cols = artifact.get("drop_cols", [])
    class_names = artifact["class_names"]
    y_le = artifact["label_encoder"]
    thresholds = artifact.get("thresholds", {})

    with st.sidebar:
        st.header("Inputs")
        model_name = st.selectbox("Choose a model", list(models.keys()))
        uploaded = st.file_uploader("Upload CSV (test data)", type=["csv"])
        run = st.button("Run prediction")

    st.caption(
        f"Target: `{target_col}` | Dropped columns if present: {drop_cols} | Classes: {class_names}"
    )

    if not uploaded:
        st.info("Upload a CSV to begin. Tip: upload `reports/test_data.csv` after training.")
        return

    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    if not run:
        return

    df2 = df.copy()
    for c in drop_cols:
        if c in df2.columns:
            df2 = df2.drop(columns=[c])

    y_true_str = None
    if target_col in df2.columns:
        y_true_str = df2[target_col].astype(str)
        df2 = df2.drop(columns=[target_col])

    model = models[model_name]
    thr = thresholds.get(model_name, 0.5)

    # Predict
    proba = model.predict_proba(df2)
    y_pred_enc = (proba[:, 1] >= thr).astype(int)
    y_pred_str = y_le.inverse_transform(y_pred_enc)

    out = df.copy()
    out["prediction"] = y_pred_str

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Predictions")
        st.dataframe(out.head(25), use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

    with col2:
        st.subheader("Evaluation (only if true labels are included)")

        if y_true_str is None:
            st.warning(f"No `{target_col}` column found in upload → cannot compute metrics.")
            return

        # Validate labels
        unknown = sorted(set(y_true_str.unique()) - set(y_le.classes_))
        if unknown:
            st.error(f"Upload contains unknown labels not seen in training: {unknown}")
            return

        y_true_enc = y_le.transform(y_true_str)

        auc = compute_binary_auc(y_true_enc, proba)

        metrics = {
            "accuracy": float(accuracy_score(y_true_enc, y_pred_enc)),
            "auc": float(auc),
            "precision": float(precision_score(y_true_enc, y_pred_enc, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true_enc, y_pred_enc, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_true_enc, y_pred_enc, pos_label=1, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true_enc, y_pred_enc)),
        }

        st.table(pd.DataFrame(metrics, index=[model_name]).T)
        st.caption(f"Decision threshold used for positive class: {thr:.2f}")

        cm = confusion_matrix(y_true_enc, y_pred_enc)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"Confusion Matrix — {model_name}")
        st.pyplot(fig)


if __name__ == "__main__":
    main()


