import time

import numpy as np
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
from sklearn.metrics import precision_score

# Reading data
df = pd.read_csv('siren_data_train.csv')

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

# Define X and y
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
    'C': np.linspace(0.01, 3.4, 1000),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

accuracies = []
precision_0 = []
precision_1 = []

start_time = time.time()

best_params_list = []  # List to store the best parameters for each fold

for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Scaling within the loop to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=kfold, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    best_params_list.append(grid_search.best_params_)  # Store the best parameters for this fold

    best_model = LogisticRegression(**grid_search.best_params_, max_iter=1000)
    best_model.fit(X_train_scaled, y_train)

    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision_0 = precision_score(y_test, y_pred, pos_label=0)
    precision_1 = precision_score(y_test, y_pred, pos_label=1)

    accuracies.append(accuracy)

    time_taken = time.time() - start_time
    print(f"Time taken for one outer fold: {time_taken:.2f} seconds")

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_precision_0 = np.mean(precision_0)
std_precision_0 = np.std(precision_0)
mean_precision_1 = np.mean(precision_1)
std_precision_1 = np.std(precision_1)
print(f"Nested CV Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Nested CV Precision_0: {mean_precision_0:.4f} ± {mean_precision_0:.4f}")
print(f"Nested CV Precision_1: {mean_precision_1:.4f} ± {mean_precision_1:.4f}")

## Train model with optimal hyperparameters

# Load the test_data
df = pd.read_csv('test_without_labels.csv')

df['Distance'] = np.sqrt((df['near_x'] - df['xcoor'])**2 + (df['near_y'] - df['ycoor'])**2)

# Define X test
X_test = df.drop(['near_x', 'xcoor', 'near_y', 'ycoor', 'near_angle', 'near_fid', 'building'], axis=1) # Leaving out target value, coordinates and near_angle that has a really low P-value and low impact

c = []
for i in best_params_list:
    c.append(i['C'])
c_mean = np.mean(c)

optimal_regression_model = LogisticRegression(C=c_mean, solver='liblinear', penalty='l1')
optimal_regression_model.fit(X, y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

y_test_pred = optimal_regression_model.predict(X_test_scaled)

y_test_pred_df = pd.DataFrame(y_test_pred)

y_test_pred_csv = y_test_pred_df.to_csv('predictions.csv')