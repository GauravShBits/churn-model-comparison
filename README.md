# Customer Churn Prediction – Five-Classifier Comparison

**Course:** Machine Learning · M.Tech (AIML) · BITS Pilani WILP · Assignment 2
**Student:** Gaurav Sharma · **BITS ID:** 2026AJ05141

## a. Problem statement

A telecom operator loses revenue whenever a subscriber cancels service (churn).
The goal is to build a **binary classifier** that predicts, from a customer's
account and service details, whether the customer will churn (`Churn = Yes`) or
stay (`Churn = No`). Five classical classification algorithms are trained on the
same data, compared on six evaluation metrics, and exposed through an
interactive Streamlit app deployed on Streamlit Community Cloud.

## b. Dataset description  [1 mark]

| Item | Details |
|---|---|
| Name | IBM Telco Customer Churn |
| Source | Public dataset (IBM sample data, also hosted on Kaggle: *blastchar/telco-customer-churn*) |
| Task | Binary classification – target column `Churn` (Yes / No) |
| Instances | 7,043 (requirement: ≥ 500) |
| Input features | 19 (requirement: ≥ 12) after dropping the `customerID` identifier |
| Numeric features (4) | `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges` |
| Categorical features (15) | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Class balance | 26.5 % churn / 73.5 % no churn (imbalanced) |

**Pre-processing** (stored *inside* every saved pipeline, so the app never
retrains or re-encodes by hand):

* `TotalCharges` contains 11 blank strings (customers with tenure 0) → coerced to numeric and median-imputed.
* Numeric columns → median imputation + `StandardScaler` (important for KNN, SVM and Logistic Regression).
* Categorical columns → `OneHotEncoder(handle_unknown="ignore")`.
* Stratified 80 / 20 train–test split (`random_state=42`): 5,634 training rows, 1,409 test rows. The test rows are saved as `test_data.csv`.

## c. GitHub repository link  [1 mark]

`<PASTE YOUR GITHUB REPO URL HERE>`

Live Streamlit app: `<PASTE YOUR STREAMLIT URL HERE>`

```
project-folder/
│-- app.py                 Streamlit application
│-- churn_utils.py         shared constants + cleaning helpers
│-- ML_Assignment2_Lab.ipynb  notebook executed on BITS Virtual Lab (training + evaluation)
│-- train_models.py        script version of the notebook (same models, same metrics)
│-- requirements.txt
│-- README.md
│-- test_data.csv          held-out test set (1,409 rows, 19 features + Churn)
│-- data/telco_raw.csv     original dataset
│-- model/                 logistic_regression.joblib, decision_tree.joblib,
│                          knn.joblib, naive_bayes.joblib, svm.joblib,
│                          metrics_summary.csv, dataset_info.json
```

Run locally: `pip install -r requirements.txt && python train_models.py && streamlit run app.py`
(or run `ML_Assignment2_Lab.ipynb` top to bottom instead of `train_models.py`).

## d. Models used  [5 marks]

All five models use the same train/test split. Precision, Recall and F1 are
computed for the positive class (`Churn = Yes`); AUC uses predicted probabilities.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.8055 | 0.8419 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| Decision Tree | 0.7970 | 0.8383 | 0.6679 | 0.4679 | 0.5503 | 0.4357 |
| KNN | 0.7771 | 0.8215 | 0.5833 | 0.5615 | 0.5722 | 0.4217 |
| Naive Bayes | 0.6948 | 0.8074 | 0.4589 | 0.8369 | 0.5928 | 0.4245 |
| Support Vector Machine | 0.7913 | 0.7905 | 0.6418 | 0.4840 | 0.5518 | 0.4264 |

Hyper-parameters: Logistic Regression (`max_iter=1000`), Decision Tree
(`max_depth=6`, `min_samples_leaf=20`), KNN (`k=15`), Gaussian Naive Bayes
(defaults), SVM (RBF kernel, `C=1.0`, `probability=True`).

### Observations on model performance  [3 marks]

| ML Model Name | Observation about model performance |
|:--|:--|
| Logistic Regression | Best overall: highest Accuracy (0.806), AUC (0.842), F1 (0.604) and MCC (0.479). Churn depends largely on contract type, tenure and charges, which a linear boundary captures well after scaling and one-hot encoding. Recall (0.56) is only moderate because of the 26.5 % class imbalance. |
| Decision Tree | Depth-limited tree gives the highest precision (0.668) and a strong AUC (0.838), so its churn flags are fairly trustworthy, but it misses over half of the churners (recall 0.468). Easy to interpret (contract type and tenure dominate the splits), but a single tree is less stable than a linear model. |
| KNN | Middling on every metric (accuracy 0.777, MCC 0.422). Distance in a 45-dimensional one-hot space is noisy, which limits it; it has the lowest precision of the "balanced" models and the slowest prediction time. |
| Naive Bayes | Lowest accuracy (0.695) and precision (0.459) but by far the highest recall (0.837): it flags most churners at the cost of many false alarms. The independence assumption is violated by correlated service columns (e.g. `InternetService` vs. streaming/security add-ons). Useful if catching every churner matters more than campaign cost. |
| Support Vector Machine | Accuracy (0.791) close to the best models but the lowest AUC (0.791) and low recall (0.484). The default RBF configuration is conservative on the minority class; class weighting or tuning `C`/`gamma` would likely help. |
| **Overall winner for this dataset** | **Logistic Regression** – it leads on Accuracy, AUC, F1 and MCC (MCC is the most reliable single number under class imbalance) and is simple and fast. Naive Bayes is the best choice only if recall is the priority. |

## Streamlit app features

* CSV upload (test data only) with a bundled-sample fallback
* Model-selection dropdown (5 models)
* Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) for the selected model
* Confusion matrix, classification report and ROC curve
* "All models side by side" tab comparing all five on the uploaded file
* Models are loaded from `model/*.joblib` – nothing is retrained in the app
