"""Train the five classifiers on the Telco churn data, evaluate on a held-out
test split, and save models, test_data.csv and the metrics table."""
import json
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from churn_utils import (CATEGORICAL_FEATURES, FEATURES, MODEL_FILES,
                         NUMERIC_FEATURES, TARGET_COL, clean_frame,
                         encode_target)

warnings.filterwarnings("ignore")
SEED = 42

raw = pd.read_csv("data/telco_raw.csv")
df = clean_frame(raw)
y = encode_target(df[TARGET_COL])
X = df[FEATURES]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

# Test file for the Streamlit app: all input features + the target column.
test_out = X_test.copy()
test_out[TARGET_COL] = y_test.map({1: "Yes", 0: "No"})
test_out.to_csv("test_data.csv", index=False)


def make_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])


classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                            random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True,
                                  random_state=SEED),
}

rows = []
for name, clf in classifiers.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    joblib.dump(pipe, MODEL_FILES[name])
    print(f"saved {MODEL_FILES[name]}")

metrics = pd.DataFrame(rows).round(4)
metrics.to_csv("model/metrics_summary.csv", index=False)
print(metrics.to_string(index=False))
print("\nMarkdown table:\n")
print(metrics.to_markdown(index=False))
json.dump({"n_rows": len(df), "n_features": len(FEATURES),
           "n_train": len(X_train), "n_test": len(X_test),
           "churn_rate": float(y.mean())},
          open("model/dataset_info.json", "w"), indent=2)
