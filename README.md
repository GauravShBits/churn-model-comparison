# Customer Churn Prediction – Five-Classifier Comparison

**Course:** Machine Learning · M.Tech (AIML) · BITS Pilani WILP · Assignment 2  
**Student:** Gaurav Sharma · **BITS ID:** 2026AJ05141

## a. Problem statement

A telecom operator loses revenue whenever a subscriber cancels service (churn).  
The goal is to build a **binary classifier** that predicts, from a customer's account and service details, whether the customer will churn (`Churn = Yes`) or stay (`Churn = No`). Five classical classification algorithms are trained on the same data, compared on six evaluation metrics, and exposed through an interactive Streamlit app deployed on Streamlit Community Cloud.

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

**Pre-processing** (stored *inside* every saved pipeline, so the app never retrains or re-encodes by hand):

* `TotalCharges` contains 11 blank strings (customers with tenure 0) → coerced to numeric and median-imputed.
* Numeric columns → median imputation + `StandardScaler` (important for KNN, SVM and Logistic Regression).
* Categorical columns → `OneHotEncoder(handle_unknown="ignore")`.
* Stratified 80 / 20 train–test split (`random_state=42`): 5,634 training rows, 1,409 test rows. The test rows are saved as `test_data.csv`.

## c. GitHub repository link  [1 mark]

https://github.com/GauravShBits/churn-model-comparison

**Live Streamlit app:** https://churn-model-comparison-5dkalurrhtppl3gadw88dg.streamlit.app/

```text
project-folder/
│-- app.py                    Streamlit application
│-- churn_utils.py            shared constants + cleaning helpers
│-- ML_Assignment2_Lab.ipynb  notebook executed on BITS Virtual Lab (training + evaluation)
│-- train_models.py           script version of the notebook (same models, same metrics)
│-- requirements.txt
│-- README.md
│-- test_data.csv             held-out test set (1,409 rows, 19 features + Churn)
│-- data/telco_raw.csv        original dataset
│-- model/                    logistic_regression.joblib, decision_tree.joblib,
│                             knn.joblib, naive_bayes.joblib, svm.joblib,
│                             metrics_summary.csv, dataset_info.json
```

Run locally:

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Alternatively, run `ML_Assignment2_Lab.ipynb` top to bottom instead of `train_models.py`.

## d. Models used  [5 marks]

All five models use the same train/test split. Precision, Recall and F1 are computed for the positive class (`Churn = Yes`); AUC uses predicted probabilities.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.8055 | 0.8419 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| Decision Tree | 0.7970 | 0.8383 | 0.6679 | 0.4679 | 0.5503 | 0.4357 |
| KNN | 0.7771 | 0.8215 | 0.5833 | 0.5615 | 0.5722 | 0.4217 |
| Naive Bayes | 0.6948 | 0.8074 | 0.4589 | 0.8369 | 0.5928 | 0.4245 |
| Support Vector Machine | 0.7913 | 0.7905 | 0.6418 | 0.4840 | 0.5518 | 0.4264 |

**Hyper-parameters:** Logistic Regression (`max_iter=1000`), Decision Tree (`max_depth=6`, `min_samples_leaf=20`), KNN (`k=15`), Gaussian Naive Bayes (defaults), SVM (RBF kernel, `C=1.0`, `probability=True`).

### Observations on model performance  [3 marks]

| ML Model Name | Observation about model performance |
|:--|:--|
| Logistic Regression | **Best overall:** highest Accuracy (0.8055), AUC (0.8419), F1 (0.6040) and MCC (0.4790). After scaling and one-hot encoding, a linear boundary fits this data well; the largest coefficients are tenure and two-year contracts (lower churn) and month-to-month contracts and fiber-optic internet (higher churn). Recall is 0.5588, meaning the model identifies about 56% of actual churners and misses about 44% of them. This indicates that while Logistic Regression performs best overall, it may still miss a substantial number of customers who are at risk of churning. |
| Decision Tree | Highest precision (0.6679) and a strong AUC (0.8383). Its relatively high precision means that customers it flags as churners are more likely to actually churn. Recall is just 0.4679, so it misses more than half of the churners, which would leave a retention team missing many at-risk customers. Its top splits are month-to-month contract, fiber optic and tenure. I limited it to `max_depth=6` and `min_samples_leaf=20` to reduce overfitting. |
| KNN | Middle of the pack on most metrics (Accuracy 0.7771, F1 0.5722), with the lowest MCC (0.4217). Distances are computed over 45 encoded columns, mostly 0/1 dummy variables, which makes "nearest" neighbours less meaningful. `k=15` smooths the predictions, but KNN still trails Logistic Regression and is slower at prediction time because it compares against all training rows. |
| Naive Bayes | Highest recall by far (0.8369) but the lowest precision (0.4589) and accuracy (0.6948), so it identifies most churners at the cost of many false alarms. Gaussian NB assumes features are independent and normally distributed, which fits poorly here: the one-hot columns are binary, and service columns such as InternetService and the streaming and security add-ons are correlated. It is useful when catching as many churners as possible is more important than the cost of extra retention offers. |
| Support Vector Machine | Accuracy (0.7913) is close to the best models, but it has the lowest AUC (0.7905) and low recall (0.4840). With default settings (RBF kernel, C=1.0) it favours the majority class, so many churners are missed. Tuning C and gamma, or using class weights, could improve minority-class recall. |
| **Overall winner for this dataset** | **Logistic Regression.** It achieves the highest Accuracy (0.8055), AUC (0.8419), F1 Score (0.6040), and MCC (0.4790), providing the best overall balance across the evaluation metrics. Given the class imbalance, MCC provides a useful single-number summary of overall classification quality and is considered alongside the other evaluation metrics when selecting the overall winner. Naive Bayes achieves substantially higher Recall (0.8369), making it preferable when the primary objective is to identify as many churners as possible, but this comes at the cost of lower Precision and Accuracy. |

## Streamlit app features

* CSV upload (test data only) with a bundled-sample option
* Model-selection dropdown (5 models)
* Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) for the selected model
* Confusion matrix, classification report, ROC curve and score distribution
* Adjustable decision threshold, with metrics updating live
* Feature-driver chart for Logistic Regression and Decision Tree
* "Compare models" tab: all five models side by side, ranking and combined ROC curves
* "Predict a customer" tab: score a single customer with any model
* "Predictions" tab: filter rows and download predictions as CSV
* Models are loaded from `model/*.joblib` – nothing is retrained in the app
