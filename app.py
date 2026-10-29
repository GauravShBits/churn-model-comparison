"""Streamlit dashboard: Telco customer churn - five classifiers compared.

Pre-trained pipelines are loaded from model/ (nothing is retrained here).
Upload a test CSV (all 19 input features + the 'Churn' column) to evaluate models,
tune the decision threshold, compare all five models, or score a single customer.
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from churn_utils import (FEATURES, MODEL_FILES, TARGET_COL, clean_frame,
                         encode_target)

st.set_page_config(page_title="Churn Model Lab", page_icon="📉", layout="wide")

ACCENT = "#4f46e5"
PALETTE = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444"]
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
SAMPLE_FILE = "test_data.csv"

YES_NO = ["No", "Yes"]
INTERNET_ADDON = ["No", "Yes", "No internet service"]
CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

st.markdown(
    f"""
    <style>
    .block-container {{padding-top: 1.6rem;}}
    .hero {{background: linear-gradient(120deg, {ACCENT} 0%, #0ea5e9 100%);
            padding: 1.4rem 1.8rem; border-radius: 14px; color: white; margin-bottom: 1rem;}}
    .hero h1 {{margin: 0; font-size: 1.9rem; color: white;}}
    .hero p {{margin: .3rem 0 0 0; opacity: .92;}}
    div[data-testid="stMetric"] {{background: rgba(128,128,128,.10); border-radius: 12px;
            padding: .7rem .9rem; border: 1px solid rgba(128,128,128,.20);}}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------- helpers
@st.cache_resource
def load_model(model_name: str):
    """Load a saved scikit-learn pipeline (cached across reruns)."""
    return joblib.load(MODEL_FILES[model_name])


@st.cache_data(show_spinner=False)
def predict_churn_probability(model_name: str, features: pd.DataFrame) -> np.ndarray:
    """Probability of churn for every row, from the saved pipeline."""
    return load_model(model_name).predict_proba(features)[:, 1]


@st.cache_data(show_spinner=False)
def predict_labels(model_name: str, features: pd.DataFrame, threshold: float) -> np.ndarray:
    """0/1 churn labels. At the default 0.50 the model's own predict() is used (this matches the
    README metrics exactly); at any other threshold, churn is predicted when probability >= threshold."""
    if abs(threshold - 0.5) < 1e-9:
        return load_model(model_name).predict(features).astype(int)
    return (predict_churn_probability(model_name, features) >= threshold).astype(int)


def evaluate_predictions(y_true: pd.Series, predicted: np.ndarray, probability: np.ndarray) -> dict:
    """Six evaluation metrics for a set of 0/1 predictions and churn probabilities."""
    has_both_classes = y_true.nunique() > 1
    return {
        "Accuracy": accuracy_score(y_true, predicted),
        "AUC": roc_auc_score(y_true, probability) if has_both_classes else float("nan"),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predicted),
    }


def draw_confusion_matrix(y_true, predicted):
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(4.4, 3.7))
    axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            shade = "white" if matrix[row, col] > matrix.max() / 2 else "black"
            axis.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center",
                      fontsize=14, fontweight="bold", color=shade)
    axis.set_xticks([0, 1], ["No churn", "Churn"])
    axis.set_yticks([0, 1], ["No churn", "Churn"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    return figure


def draw_roc_curve(y_true, probability, threshold, label):
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    figure, axis = plt.subplots(figsize=(4.4, 3.7))
    axis.plot(fpr, tpr, color=ACCENT, lw=2, label=label)
    axis.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    nearest = int(np.argmin(np.abs(thresholds - threshold)))
    axis.scatter([fpr[nearest]], [tpr[nearest]], color="#ef4444", zorder=5,
                 label=f"threshold = {threshold:.2f}")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    return figure


def draw_probability_histogram(y_true, probability, threshold):
    figure, axis = plt.subplots(figsize=(4.4, 3.7))
    bins = np.linspace(0, 1, 26)
    axis.hist(probability[y_true.values == 0], bins=bins, alpha=.65, color="#0ea5e9", label="Stayed")
    axis.hist(probability[y_true.values == 1], bins=bins, alpha=.65, color="#ef4444", label="Churned")
    axis.axvline(threshold, color="black", ls="--", lw=1)
    axis.set_xlabel("Predicted churn probability")
    axis.set_ylabel("Customers")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def draw_grouped_metric_bars(table: pd.DataFrame, metrics: list):
    """Side-by-side bars: one group per model, one bar per chosen metric."""
    figure, axis = plt.subplots(figsize=(8, 3.8))
    width = 0.8 / len(metrics)
    positions = np.arange(len(table))
    for i, metric in enumerate(metrics):
        bars = axis.bar(positions + i * width, table[metric], width, label=metric, color=PALETTE[i % len(PALETTE)])
        axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    axis.set_xticks(positions + width * (len(metrics) - 1) / 2, table.index, fontsize=8)
    axis.set_ylim(0, 1.05)
    axis.legend(ncol=len(metrics), fontsize=8, loc="upper right")
    figure.tight_layout()
    return figure


def draw_feature_drivers(table: pd.DataFrame, column: str):
    """Horizontal bars; red = pushes toward churn, blue = pushes toward staying."""
    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    colours = ["#ef4444" if v > 0 else "#0ea5e9" for v in table[column]]
    axis.barh(table["Feature"], table[column], color=colours)
    axis.axvline(0, color="black", lw=.8)
    axis.set_xlabel(column, fontsize=8)
    axis.tick_params(axis="y", labelsize=8)
    figure.tight_layout()
    return figure


def feature_drivers(model_name: str):
    """Top drivers for models that expose them (Logistic Regression, Decision Tree)."""
    pipeline = load_model(model_name)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        values, kind = classifier.coef_[0], "Coefficient (+ raises churn risk)"
    elif hasattr(classifier, "feature_importances_"):
        values, kind = classifier.feature_importances_, "Importance"
    else:
        return None, None
    table = pd.DataFrame({"Feature": names, kind: values})
    table = table.reindex(table[kind].abs().sort_values(ascending=False).index).head(10)
    return table.iloc[::-1], kind


# --------------------------------------------------------------- sidebar
st.sidebar.title("📉 Churn Model Lab")
st.sidebar.caption("BITS Pilani WILP · M.Tech AIML · ML Assignment 2")

uploaded_file = st.sidebar.file_uploader(
    "1. Upload test data (CSV)", type="csv",
    help="Needs all 19 input features plus the 'Churn' column. Use test_data.csv from the repo.")
use_sample = False
if uploaded_file is None and os.path.exists(SAMPLE_FILE):
    use_sample = st.sidebar.checkbox("Use bundled test_data.csv", value=True)

selected_model = st.sidebar.selectbox("2. Choose a model", list(MODEL_FILES.keys()))
threshold = st.sidebar.slider(
    "3. Decision threshold", 0.05, 0.95, 0.50, 0.01,
    help="Customers whose churn probability is at or above this value are flagged as churners. "
         "Lower = catch more churners (higher recall); higher = fewer false alarms (higher precision).")
st.sidebar.markdown("---")
st.sidebar.caption("Positive class: **Churn = Yes**. Models and preprocessing are loaded from `model/*.joblib`.")

# ------------------------------------------------------------------ hero
st.markdown(
    "<div class='hero'><h1>Customer Churn Prediction</h1>"
    "<p>Five classifiers trained on the IBM Telco dataset. Evaluate them on your test file, "
    "tune the threshold, compare them side by side, or score a single customer.</p></div>",
    unsafe_allow_html=True)

if uploaded_file is not None:
    raw_frame = pd.read_csv(uploaded_file)
elif use_sample:
    raw_frame = pd.read_csv(SAMPLE_FILE)
else:
    st.info("👈 Upload a CSV in the sidebar (or tick the bundled sample) to evaluate the models. "
            "You can already score a single customer in the **Predict a customer** tab.")
    raw_frame = None

tab_eval, tab_compare, tab_predict, tab_rows = st.tabs(
    ["📊 Model evaluation", "⚖️ Compare models", "🧪 Predict a customer", "📥 Predictions"])

# ---------------------------------------------- tabs that need test data
features = target = None
if raw_frame is not None:
    cleaned = clean_frame(raw_frame)
    missing_columns = [c for c in FEATURES + [TARGET_COL] if c not in cleaned.columns]
    if missing_columns:
        st.error(f"The uploaded file is missing required columns: {missing_columns}")
    else:
        features, target = cleaned[FEATURES], encode_target(cleaned[TARGET_COL])

NEEDS_DATA = "Upload a valid test CSV in the sidebar (or tick the bundled sample) to use this tab."

with tab_eval:
    if features is None:
        st.info(NEEDS_DATA)
    else:
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(features):,}")
        b.metric("Input features", len(FEATURES))
        c.metric("Churn rate in file", f"{target.mean():.1%}")
        with st.expander("Preview uploaded data"):
            st.dataframe(raw_frame.head(20), width="stretch")

        probability = predict_churn_probability(selected_model, features)
        predicted = predict_labels(selected_model, features, threshold)
        scores = evaluate_predictions(target, predicted, probability)
        baseline = evaluate_predictions(target, predict_labels(selected_model, features, 0.5), probability)

        st.subheader(f"{selected_model} · threshold {threshold:.2f}")
        for column, name in zip(st.columns(6), METRIC_NAMES):
            change = scores[name] - baseline[name]
            column.metric(name, f"{scores[name]:.4f}",
                          delta=None if abs(threshold - 0.5) < 1e-9 or name == "AUC" else f"{change:+.4f}",
                          help="Change versus the default 0.50 threshold." if name != "AUC"
                          else "AUC does not depend on the threshold.")

        left, middle, right = st.columns(3)
        with left:
            st.markdown("**Confusion matrix**")
            figure = draw_confusion_matrix(target, predicted)
            st.pyplot(figure)
            plt.close(figure)
        with middle:
            st.markdown("**ROC curve**")
            if target.nunique() > 1:
                figure = draw_roc_curve(target, probability, threshold, f"AUC = {scores['AUC']:.3f}")
                st.pyplot(figure)
                plt.close(figure)
            else:
                st.info("ROC needs both classes in the file.")
        with right:
            st.markdown("**Score distribution**")
            figure = draw_probability_histogram(target, probability, threshold)
            st.pyplot(figure)
            plt.close(figure)

        report_col, driver_col = st.columns(2)
        with report_col:
            st.markdown("**Classification report**")
            report = classification_report(target, predicted, target_names=["No churn", "Churn"],
                                           output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")
        with driver_col:
            st.markdown("**What drives this model?**")
            drivers, kind = feature_drivers(selected_model)
            if drivers is None:
                st.info(f"{selected_model} has no built-in feature importances. "
                        "Pick Logistic Regression or Decision Tree to see the top drivers.")
            else:
                figure = draw_feature_drivers(drivers, kind)
                st.pyplot(figure)
                plt.close(figure)

with tab_compare:
    if features is None:
        st.info(NEEDS_DATA)
    else:
        st.subheader(f"All five models at threshold {threshold:.2f}")
        probabilities = {name: predict_churn_probability(name, features) for name in MODEL_FILES}
        comparison = pd.DataFrame(
            {name: evaluate_predictions(target, predict_labels(name, features, threshold), prob)
             for name, prob in probabilities.items()}
        ).T
        comparison.index.name = "ML Model Name"
        st.dataframe(comparison.style.format("{:.4f}").highlight_max(axis=0, color="#bbf7d0"),
                     width="stretch")

        rank_by = st.radio("Rank models by", METRIC_NAMES, index=5, horizontal=True)
        best_name = comparison[rank_by].idxmax()
        st.success(f"Best model by {rank_by}: **{best_name}** ({comparison.loc[best_name, rank_by]:.4f})")

        chosen = st.multiselect("Metrics to plot", METRIC_NAMES, default=["Accuracy", "F1", "MCC"])
        if chosen:
            figure = draw_grouped_metric_bars(comparison, chosen)
            st.pyplot(figure)
            plt.close(figure)

        if target.nunique() > 1:
            st.markdown("**ROC curves**")
            roc_column, _ = st.columns([3, 2])
            figure, axis = plt.subplots(figsize=(6.2, 4.2))
            for colour, (name, prob) in zip(PALETTE, probabilities.items()):
                fpr, tpr, _ = roc_curve(target, prob)
                axis.plot(fpr, tpr, color=colour, lw=2,
                          label=f"{name} (AUC {roc_auc_score(target, prob):.3f})")
            axis.plot([0, 1], [0, 1], "--", color="grey", lw=1)
            axis.set_xlabel("False positive rate")
            axis.set_ylabel("True positive rate")
            axis.legend(loc="lower right", fontsize=8)
            figure.tight_layout()
            with roc_column:
                st.pyplot(figure)
            plt.close(figure)

with tab_predict:
    st.subheader("Score a single customer")
    st.caption(f"Uses the selected model (**{selected_model}**) and the sidebar threshold ({threshold:.2f}).")
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, 0.5)
            total = st.number_input("Total charges ($)", 0.0, 10000.0, float(round(70.0 * 12, 2)), 10.0)
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No")
            gender = st.selectbox("Gender", CATEGORY_OPTIONS["gender"])
            partner = st.selectbox("Partner", CATEGORY_OPTIONS["Partner"])
            dependents = st.selectbox("Dependents", CATEGORY_OPTIONS["Dependents"])
        with col2:
            contract = st.selectbox("Contract", CATEGORY_OPTIONS["Contract"])
            payment = st.selectbox("Payment method", CATEGORY_OPTIONS["PaymentMethod"])
            paperless = st.selectbox("Paperless billing", CATEGORY_OPTIONS["PaperlessBilling"], index=1)
            phone = st.selectbox("Phone service", CATEGORY_OPTIONS["PhoneService"], index=1)
            lines = st.selectbox("Multiple lines", CATEGORY_OPTIONS["MultipleLines"])
            internet = st.selectbox("Internet service", CATEGORY_OPTIONS["InternetService"], index=1)
        with col3:
            security = st.selectbox("Online security", CATEGORY_OPTIONS["OnlineSecurity"])
            backup = st.selectbox("Online backup", CATEGORY_OPTIONS["OnlineBackup"])
            protection = st.selectbox("Device protection", CATEGORY_OPTIONS["DeviceProtection"])
            support = st.selectbox("Tech support", CATEGORY_OPTIONS["TechSupport"])
            tv = st.selectbox("Streaming TV", CATEGORY_OPTIONS["StreamingTV"])
            movies = st.selectbox("Streaming movies", CATEGORY_OPTIONS["StreamingMovies"])
        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if submitted:
        customer = pd.DataFrame([{
            "SeniorCitizen": senior, "tenure": tenure, "MonthlyCharges": monthly,
            "TotalCharges": total, "gender": gender, "Partner": partner,
            "Dependents": dependents, "PhoneService": phone, "MultipleLines": lines,
            "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
            "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment}])[FEATURES]
        risk = float(predict_churn_probability(selected_model, customer)[0])
        flagged = int(predict_labels(selected_model, customer, threshold)[0]) == 1
        verdict = "likely to churn" if flagged else "likely to stay"
        (st.error if flagged else st.success)(
            f"**{risk:.1%}** churn probability: this customer is **{verdict}** "
            f"(threshold {threshold:.0%}).")
        st.progress(min(max(risk, 0.0), 1.0))

        by_model = pd.DataFrame({
            "Churn probability": {name: float(predict_churn_probability(name, customer)[0])
                                  for name in MODEL_FILES}})
        st.markdown("**What each model says about this customer**")
        st.bar_chart(by_model)

with tab_rows:
    if features is None:
        st.info(NEEDS_DATA)
    else:
        probability = predict_churn_probability(selected_model, features)
        predicted = predict_labels(selected_model, features, threshold)
        results = raw_frame.copy()
        results["Churn probability"] = probability.round(4)
        results["Predicted"] = np.where(predicted == 1, "Yes", "No")
        results["Correct"] = predicted == target.values

        view = st.radio("Show", ["All rows", "Predicted to churn", "Misclassified only"], horizontal=True)
        if view == "Predicted to churn":
            results = results[results["Predicted"] == "Yes"]
        elif view == "Misclassified only":
            results = results[~results["Correct"]]
        results = results.sort_values("Churn probability", ascending=False)

        st.caption(f"{len(results):,} rows · model: {selected_model} · threshold {threshold:.2f}")
        st.dataframe(results, width="stretch", height=420)
        st.download_button("Download these predictions (CSV)",
                           results.to_csv(index=False).encode("utf-8"),
                           file_name="churn_predictions.csv", mime="text/csv")
