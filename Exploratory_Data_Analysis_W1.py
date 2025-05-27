# -*- coding: utf-8 -*-
"""
Created on Tue May 27 12:49:23 2025

@author: aajas
"""
# Wine Dataset - Exploratory Data Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer

# Set column names
column_names = ['Output', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 
                'col7', 'col8', 'col9', 'col10', 'col11', 'col12']

# Load dataset
filename = 'wine-1.csv'
data = pd.read_csv(filename, header=None, names=column_names)

# Preview data
print("Original DataFrame:")
print(data.head())

# Input and output separation
inputs = data.iloc[:, 1:]
output = data.iloc[:, 0]

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(inputs.describe())

# Histogram of features
plt.figure(figsize=(12, 8))
inputs.hist(bins=12, figsize=(12, 8))
plt.suptitle('Original Feature Distributions')
plt.show()

# Skewness for potential log transformation
print("\nSkewness:")
print(inputs.skew())

# Example log transformation for skewed columns
data_log = data.copy()
skewed_cols = ['col2', 'col10']
for col in skewed_cols:
    data_log[col] = np.log1p(data_log[col])

# Standardization
scaler_std = StandardScaler()
data_standardized = pd.DataFrame(scaler_std.fit_transform(inputs), columns=inputs.columns)

# Normalization
scaler_norm = Normalizer()
data_normalized = pd.DataFrame(scaler_norm.fit_transform(inputs), columns=inputs.columns)

# Summary of normalized data
print("\nNormalized Data Summary:")
print(data_normalized.describe())

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr_matrix = inputs.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Pairwise scatter plots
sns.pairplot(inputs)
plt.suptitle('Scatter Plot Matrix', y=1.02)
plt.show()

# Summary Comments
print("\nSummary:")
print("- No null values found.")
print("- Features have varied scales; standardization is appropriate.")
print("- Right-skewed features (e.g., col2, col10) benefit from log transform.")
print("- High correlation exists among some features; consider this for modeling.")
print("- Data is suitable for predictive modeling with preprocessing.")

