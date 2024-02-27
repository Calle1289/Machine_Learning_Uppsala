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

# Define X and y for your model
X = df.drop(['heard', 'near_x', 'xcoor', 'near_y', 'ycoor', 'near_angle', 'near_fid', 'building'], axis=1) # Leaving out target value, coordinates and near_angle that has a really low P-value and low impact
y = df['heard']

results = {}
for column in X.columns:
    corr_coefficient, p_value = pearsonr(X[column], df['heard'])
    results[column] = {'Correlation Coefficient': corr_coefficient, 'P-value': p_value}
results_X = pd.DataFrame(results).T 
print(results_X)