# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:13:53 2025

@author: aajas
"""

# Comprehensive EDA and Feature Selection on Boston Housing Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, Normalizer
from numpy import set_printoptions

# Load dataset
names = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat','medv']
df = pd.read_csv('boston.csv')
df = df.drop('Unnamed: 0', axis=1)

# Separate features and response
X1 = df.drop(columns='medv')
Y1 = df['medv']
X = X1.values
Y = Y1.values

print(df.head())

# Descriptive statistics
print(df.describe())
print(X1.describe())

# Histogram
plt.figure(figsize=(10, 8))
df.hist(bins=20, figsize=(10, 8))
plt.suptitle('Boston df Histogram')
plt.show()

# Pair Plot
sns.pairplot(df)
plt.suptitle('PairPlot Boston Data Set')
plt.show()

# Correlation heatmap
corMat = df.corr(method='pearson')
print(corMat)
plt.figure(figsize=(8,6))
sns.heatmap(data=corMat, cmap='coolwarm', center=0, annot=True)
plt.title('Correlation Heatmap')
plt.show()

# Scatter matrix
plt.figure()
scatter_matrix(df, figsize=(10,8))
plt.suptitle("Scatter Matrix")
plt.show()

# Recursive Feature Elimination
model = LinearRegression()
NUM_FEATURES = 12
rfe = RFE(model, n_features_to_select=NUM_FEATURES)
fit = rfe.fit(X, Y)

print("Num Features:", fit.n_features_)
print("Selected Features:", fit.support_)
print("Feature Ranking:", fit.ranking_)
print("Model Score:", fit.score(X, Y))

# Stepwise Selection

def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]

        if not new_pval.empty and (best_pval := new_pval.min()) < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f"Add {best_feature:30} with p-value {best_pval:.6f}")

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        if not pvalues.empty and (worst_pval := pvalues.max()) > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f"Drop {worst_feature:30} with p-value {worst_pval:.6f}")

        if not changed:
            break
    return included

# Convert Y1 to Series
Y1 = Y1.squeeze()
result = stepwise_selection(X1, Y1)
print('Resulting features:', result)

# Standardization
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
rescaledX1 = StandardScaler().fit_transform(X1)
set_printoptions(precision=3)
print("Rescaled X1:", rescaledX1[:5])

dataStandDf = pd.DataFrame(rescaledX, columns=names[0:13])
dataStandDf['class'] = Y

# Normalization
scaler= Normalizer().fit(X1)
normalizedX = scaler.transform(X1)
dataNormDf = pd.DataFrame(normalizedX, columns=names[0:13])
print("Description after Normalization:", dataNormDf.describe())

plt.figure(figsize=(12, 8))
dataNormDf.hist(bins=20, figsize=(12, 8))
plt.suptitle('Normalized Data (dataNormDf)')
plt.show()

# Standardization descriptive stats
X_scaled_df = pd.DataFrame(rescaledX, columns=X1.columns)
print("X_scaled_df.describe :", X_scaled_df.describe())

plt.figure(figsize=(12, 8))
X_scaled_df.hist(bins=20, figsize=(12, 8))
plt.suptitle('Histogram of X after Standardization')
plt.show()

sns.pairplot(X_scaled_df)
plt.suptitle('X Scatter Plot')
plt.show()

scatter_matrix(X_scaled_df, figsize=(12, 12), diagonal='hist', alpha=0.7)
plt.suptitle('X Scatter Matrix')
plt.show()

corMat_X = X_scaled_df.corr(method='pearson')
print(corMat_X)
plt.figure(figsize=(8,6))
sns.heatmap(data=corMat_X, cmap='coolwarm', center=0, annot=True)
plt.title('Correlation Heatmap of X')
plt.show()
