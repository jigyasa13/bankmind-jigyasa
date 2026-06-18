# BankPredict: Term Deposit Conversion Pipeline

This project builds a machine learning pipeline to predict whether a customer will subscribe to a bank term deposit based on marketing campaign data.

---

## 📊 Quick Project Insights & EDA
Before running the pipeline, we analyzed the dataset to understand our customer base:

* **Class Imbalance:** About 89% of customers said "no" and only 11% said "yes". Because of this heavy imbalance, we optimized our models for **F1-Score** instead of raw accuracy so the model doesn't just cheat by guessing "no" every time.
* **Customer Age Demographics:** The age histogram shows that the majority of targeted clients are between 30 and 45 years old. Marketing focus is heavily placed on working-age professionals.

---

## 🛠️ Pipeline Setup & Cleaning
* **Data Leakage Prevention:** We dropped the `duration` column entirely. A call's length isn't known *before* making a call, so keeping it would make the model unrealistically optimistic.
* **Preprocessing:** Categorical data was transformed using `OneHotEncoder` and `OrdinalEncoder`, and numerical features were normalized using `StandardScaler` inside a unified `ColumnTransformer`.
* **Hyperparameter Tuning:** We used `RandomizedSearchCV` with 3-fold cross-validation. By explicitly adding `scoring='f1'`, the pipeline automatically selected `class_weight='balanced'` for Logistic Regression, boosting our minority class F1-score from **0.28 to 0.34**.

---

## 🏆 Key Drivers (Top 3 Most Important Features)
Based on our Random Forest feature importance analysis, the top 3 core drivers determining customer conversions are:
1. **`balance`:** The customer's yearly average bank balance (financial capability).
2. **`age`:** The age demographic profile of the client.
3. **`pdays`:** The number of days that passed by after the client was last contacted from a previous campaign.

---

## 📈 Final Model Settings
* **Random Forest:** `{'n_estimators': 200, 'max_depth': 10}`
* **Logistic Regression:** `{'class_weight': 'balanced', 'C': 10.0}`
