# Bias-Variance Tradeoff Analysis on Boston Housing Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp
import warnings

warnings.filterwarnings('ignore')

# Load dataset
column_names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis',
                'rad', 'tax', 'ptratio', 'black', 'lstat', 'medv']
df = pd.read_csv('boston.csv')
df = df.drop('Unnamed: 0', axis=1)  # Remove unnecessary column

# Separate features and target
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Define models
model_top6 = LinearRegression()
model_top12 = LinearRegression()

# Perform RFE to select top 6 and 12 features
rfe6 = RFE(model_top6, n_features_to_select=6)
rfe6.fit(X, Y)
selected6 = rfe6.support_

rfe12 = RFE(model_top12, n_features_to_select=12)
rfe12.fit(X, Y)
selected12 = rfe12.support_

# Display selected feature names
feature_names = column_names[:-1]
selected_names_6 = [name for name, flag in zip(feature_names, selected6) if flag]
selected_names_12 = [name for name, flag in zip(feature_names, selected12) if flag]

print(f"Top 6 features: {selected_names_6}")
print(f"Top 12 features: {selected_names_12}")

# Use the same train/test split and slice by selected features
X_train_full, X_test_full, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

X_train_6 = X_train_full[:, selected6]
X_test_6 = X_test_full[:, selected6]
X_train_12 = X_train_full[:, selected12]
X_test_12 = X_test_full[:, selected12]

# Evaluate bias-variance for both models
error_6, bias_6, var_6 = bias_variance_decomp(
    model_top6, X_train_6, y_train, X_test_6, y_test,
    loss='mse', random_seed=123
)

error_12, bias_12, var_12 = bias_variance_decomp(
    model_top12, X_train_12, y_train, X_test_12, y_test,
    loss='mse', random_seed=123
)

# Print results
print("\nModel with 6 features:")
print(f"Average expected loss: {error_6:.3f}")
print(f"Average bias: {bias_6:.3f}")
print(f"Average variance: {var_6:.3f}")

print("\nModel with 12 features:")
print(f"Average expected loss: {error_12:.3f}")
print(f"Average bias: {bias_12:.3f}")
print(f"Average variance: {var_12:.3f}")

# Compare variance
variance_change = (var_6 / var_12 - 1) * 100
print(f"\nVariance reduction from 12 to 6 features: {variance_change:.2f}%")

# Plot Bias vs Variance
labels = ['6 features', '12 features']
biases = [bias_6, bias_12]
variances = [var_6, var_12]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, biases, width, label='Bias')
ax.bar(x + width/2, variances, width, label='Variance')

ax.set_ylabel('Error')
ax.set_title('Bias vs Variance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()
