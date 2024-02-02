import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('Project_1\siren_data_train.csv')

X = df.drop('heard', axis=1)
y = df['heard']

param_grid = {
    'C': np.logspace(-4, 4, 20),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=StratifiedKFold(n_splits=10, random_state=42, shuffle=True), scoring='accuracy')

grid_search.fit(X, y)

print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
