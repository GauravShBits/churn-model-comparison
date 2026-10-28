"""Streamlit dashboard: Telco customer churn - five classifiers compared.

Loads pre-trained pipelines from model/ (no retraining) and evaluates them on
an uploaded test CSV that contains all input features and the 'Churn' column.
"""
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from churn_utils import (FEATURES, MODEL_FILES, TARGET_COL, clean_frame,
                         encode_target)

st.set_page_config(page_title="Churn Model Lab", page_icon="📉", layout="wide")


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def score(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    out = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba) if y.nunique() > 1 else float("nan"),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return out, pred, proba


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📉 Churn Model Lab")
st.sidebar.caption("BITS Pilani WILP · M.Tech AIML · ML Assignment 2")

uploaded = st.sidebar.file_uploader(
    "1. Upload test data (CSV)", type="csv",
    help="Must contain all 19 input features plus the 'Churn' column.")
use_sample = False
if uploaded is None and os.path.exists("test_data.csv"):
    use_sample = st.sidebar.checkbox("Use bundled test_data.csv instead", value=True)

model_name = st.sidebar.selectbox("2. Choose a model", list(MODEL_FILES.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Task:** predict whether a telecom customer will churn "
    "(`Yes` = churn, treated as the positive class).")

# ------------------------------------------------------------------- main
st.title("Customer Churn Prediction – Model Comparison")
st.write("Five classifiers were trained offline on the IBM Telco Customer Churn "
         "dataset. Upload a held-out test CSV to see how each one performs.")

if uploaded is not None:
    raw = pd.read_csv(uploaded)
elif use_sample:
    raw = pd.read_csv("test_data.csv")
else:
    st.info("Upload a CSV in the sidebar (or tick the bundled sample) to begin.")
    st.stop()

data = clean_frame(raw)
missing = [c for c in FEATURES + [TARGET_COL] if c not in data.columns]
if missing:
    st.error(f"The uploaded file is missing required columns: {missing}")
    st.stop()

X = data[FEATURES]
y = encode_target(data[TARGET_COL])

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(data):,}")
c2.metric("Input features", len(FEATURES))
c3.metric("Churn rate in file", f"{y.mean():.1%}")
with st.expander("Preview uploaded data"):
    st.dataframe(raw.head(20), width="stretch")

tab_single, tab_all = st.tabs([f"Selected model: {model_name}", "All models side by side"])

with tab_single:
    model = load_model(MODEL_FILES[model_name])
    metrics, pred, proba = score(model, X, y)

    st.subheader("Evaluation metrics")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No churn", "Churn"],
                    yticklabels=["No churn", "Churn"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        rep = classification_report(y, pred, target_names=["No churn", "Churn"],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

        if y.nunique() > 1:
            fpr, tpr, _ = roc_curve(y, proba)
            fig2, ax2 = plt.subplots(figsize=(4.5, 3.2))
            ax2.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
            ax2.plot([0, 1], [0, 1], "--", color="grey")
            ax2.set_xlabel("False positive rate")
            ax2.set_ylabel("True positive rate")
            ax2.legend(loc="lower right")
            st.pyplot(fig2)

with tab_all:
    st.subheader("Metrics for every model on the uploaded data")
    rows = []
    for name, path in MODEL_FILES.items():
        m, _, _ = score(load_model(path), X, y)
        rows.append({"ML Model Name": name, **m})
    table = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"),
                 width="stretch")
    best = table["MCC"].idxmax()
    st.success(f"Highest MCC on this file: **{best}** ({table.loc[best, 'MCC']:.4f})")
    st.bar_chart(table[["Accuracy", "AUC", "F1", "MCC"]])
