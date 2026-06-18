import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Reading the data
df = pd.read_csv("bank-full.csv", sep=";")

#Independent and dependent features
X = df.drop(["y", "duration"], axis=1)  
y = df["y"]
y = y.map({'yes': 1, 'no': 0})  

numerical_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous'] 
nominal_cols = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome']
ordinal_cols = ['education', 'month']

# Data Encoding and Standardization 
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

scaler = StandardScaler()
OHE = OneHotEncoder(drop='first')   
OE = OrdinalEncoder()

preprocessor = ColumnTransformer([
    ("OneHotEncoder", OHE, nominal_cols),        
    ("OrdinalEncoder", OE, ordinal_cols),
    ("StandardScaler", scaler, numerical_cols)
])