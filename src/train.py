from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

from src.preprocess import X, y, preprocessor

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pipeline Transformations
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Setting Hyperparameters Gird
rf_params = {"max_depth": [5, 8, 15, None, 10],
             "n_estimators": [100, 200, 500, 1000]}

log_reg_params = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'class_weight': [None, 'balanced']
}

randomcv_models = [
    ["Random Forest", RandomForestClassifier(), rf_params],
    ["Logistic Regression", LogisticRegression(), log_reg_params]
]

model_param = {}

# Tuning Optimization Loop
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model, param_distributions=params, cv=3, verbose=3, scoring='f1')  
    random.fit(X_train, y_train)
    y_pred = random.predict(X_test)
    print(name)
    print(classification_report(y_test, y_pred))
    model_param[name] = random.best_params_ 

for name in model_param:
    print(name, ":")
    print(model_param[name])



# Customer Showcase Generation Block
y_pred = random.predict(X_test)
y_probs = random.predict_proba(X_test)[:, 1]  

X_test_readable = X.loc[y_test.index].copy()

X_test_readable['Predicted_Class'] = y_pred
X_test_readable['Actual_Class'] = y_test.values
X_test_readable['Confidence_Score (%)'] = np.round(y_probs * 100, 2)

predicted_yes = X_test_readable[X_test_readable['Predicted_Class'] == 1]
predicted_no = X_test_readable[X_test_readable['Predicted_Class'] == 0]

sample_yes = predicted_yes.tail(3)
sample_no = predicted_no.head(2)
showcase_df = pd.concat([sample_yes, sample_no])

for rank, (idx, customer) in enumerate(showcase_df.iterrows(), 1):
    print(f"Customer Showcase #{rank} (Dataset Index Row: {idx})")
    print("-" * 65)
    print(f" Demographics: {customer['age']} years old | Job: {customer['job']} | Status: {customer['marital']}")
    print(f" Financials  : Balance: {customer['balance']}€ | Housing Loan: {customer['housing']} | Personal Loan: {customer['loan']}")
    print(f" Prediction  : {'SUBSCRIBED (YES)' if customer['Predicted_Class'] == 1 else 'NO SUBSCRIPTION (NO)'}")
    print(f" Probability : {customer['Confidence_Score (%)']}% chance of a positive conversion")
    print(f" Actual Label: {'YES' if customer['Actual_Class'] == 1 else 'NO'}")
    print("=" * 65 + "\n")


## Most important features in the dataset

best_rf_params = model_param["Random Forest"]

rf_model = RandomForestClassifier(**best_rf_params, random_state=42)
rf_model.fit(X_train, y_train)

from src.preprocess import nominal_cols, ordinal_cols, numerical_cols
ohe_features = list(preprocessor.named_transformers_['OneHotEncoder'].get_feature_names_out(nominal_cols))
all_features = ohe_features + list(ordinal_cols) + list(numerical_cols)

importances = pd.DataFrame(rf_model.feature_importances_)
importances['feature'] = all_features
importances.columns = ['importance', 'feature']
importances.sort_values(by='importance', ascending=True, inplace=True)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 12))
ax.barh(importances.feature, importances.importance, color='skyblue', edgecolor='black')
ax.set_title("Random Forest Feature Importances (All Columns)")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
plt.tight_layout()
plt.show()