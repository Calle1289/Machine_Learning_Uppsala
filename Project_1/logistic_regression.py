import time

import numpy as np
import sklearn as sk
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Reading data
df = pd.read_csv('.\siren_data_train.csv')

## Correlation of features with P-Value, Outliers, Range of Values
# Correlation of features with P-Value
results = {}
for column in df.drop('heard', axis=1).columns:
    corr_coefficient, p_value = pearsonr(df[column], df['heard'])
    results[column] = {'Correlation Coefficient': corr_coefficient, 'P-value': p_value}
results_df = pd.DataFrame(results).T 
print(results_df)

## Outliers with boxplot
# Get the list of parameters
parameters = df.drop(columns=['heard', 'building', 'noise', 'asleep', 'in_vehicle', 'no_windows']).columns # Don't include binary

# Define the number of rows and columns for subplots
num_rows = 2
num_cols = 4

# Create a new figure with subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 12))

# Plot boxplots for each parameter
for i, parameter in enumerate(parameters):
    row = i // num_cols
    col = i % num_cols
    sns.boxplot(y=df[parameter], ax=axes[row, col])
    axes[row, col].set_title(f'Boxplot of {parameter}')
    axes[row, col].set_ylabel('Values')
    axes[row, col].set_xlabel('')

# Hide any unused subplots
for i in range(len(parameters), num_rows * num_cols):
    axes[i // num_cols, i % num_cols].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()

df.describe() # Range of values

## Extracting valuable values and leaving some out
# Calculate the distance and create a new column for it
df['Distance'] = np.sqrt((df['near_x'] - df['xcoor'])**2 + (df['near_y'] - df['ycoor'])**2)

# Define X and y for your model
X = df.drop(['heard', 'near_x', 'xcoor', 'near_y', 'ycoor', 'near_angle', 'near_fid', 'building'], axis=1) # Leaving out target value, coordinates and near_angle that has a really low P-value and low impact
y = df['heard']

results = {}
for column in X.columns:
    corr_coefficient, p_value = pearsonr(X[column], df['heard'])
    results[column] = {'Correlation Coefficient': corr_coefficient, 'P-value': p_value}
results_X = pd.DataFrame(results).T 
print(results_X)

## Find optimal parameters for model
kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

param_grid = {
    'C': np.logspace(-4, 4, 20),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

accuracies = []

start_time = time.time()

conf_matrix = np.zeros((2, 2))

for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Scaling within the loop to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=kfold, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    best_model = LogisticRegression(**grid_search.best_params_, max_iter=1000)
    best_model.fit(X_train_scaled, y_train)

    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    accuracies.append(accuracy)

    time_taken = time.time() - start_time
    print(f"Time taken for one outer fold: {time_taken:.2f} seconds")

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f"Nested CV Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(classification_report)

