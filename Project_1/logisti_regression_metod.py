import time

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

# Reading data
df = pd.read_csv('siren_data_train.csv')

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
    'C': np.linspace(0.00001, 30, 20),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

accuracies = []
precision_0 = []
precision_1 = []

start_time = time.time()

best_params_list = []  # List to store the best parameters for each fold


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)

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