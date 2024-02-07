from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

x_test_regression = np.load('X_test_regression1.npy')
x_train_regression = np.load('X_train_regression1.npy')
y_train_regression = np.load('y_train_regression1.npy')


# Define the number of folds (k)
k = 15

# Initialize the linear regression model
model_linear_regression = LinearRegression()

# Initialize KFold cross-validator
kf = KFold(n_splits=k)

# Initialize an array to store SSE values for each fold for Linear regression
sse_values_linear = []

# Initialize an array to store SSE values for each fold for Ridge regression
sse_values_ridge = []

# Initialize an array to store SSE values for each fold for Lasso regression
sse_values_lasso = []

# Perform k-fold cross-validation
for train_index, test_index in kf.split(x_train_regression):
    x_train, x_test = x_train_regression[train_index], x_train_regression[test_index]
    y_train, y_test = y_train_regression[train_index], y_train_regression[test_index]

    # Train the model on the training data
    model_linear_regression.fit(x_train, y_train)

    # Train the Ridge regression model on the training data
    alpha_ridge = 2.0
    model_ridge = Ridge(alpha=alpha_ridge)
    model_ridge.fit(x_train, y_train)

    # Train the Lasso regression model on the training data
    alpha_lasso = 0.1
    model_lasso = Lasso(alpha=alpha_lasso)
    model_lasso.fit(x_train, y_train)

    # Make predictions on the test data for Linear regression
    y_pred_linear = model_linear_regression.predict(x_test)

    # Make predictions on the test data for Ridge regression
    y_pred_ridge = model_ridge.predict(x_test)

    # Make predictions on the test data for Lasso regression
    y_pred_lasso = model_lasso.predict(x_test)

    # Calculate SSE for Linear regression
    sse = np.sum((y_test - y_pred_linear) ** 2)
    sse_values_linear.append(sse)

    # Calculate SSE for Ridge regression
    sse_ridge = np.sum((y_test - y_pred_ridge) ** 2)
    sse_values_ridge.append(sse_ridge)

    # Calculate SSE for Lasso regression
    sse_lasso = np.sum((y_test - y_pred_lasso) ** 2)
    sse_values_lasso.append(sse_lasso)

# Calculate the mean SSE over all folds for the three models
mean_sse_linear = np.mean(sse_values_linear)
mean_sse_ridge = np.mean(sse_values_ridge)
mean_sse_lasso = np.mean(sse_values_lasso)

# Print the mean SSE as a measure of model performance
print(f"Mean SSE for Linear regression: {mean_sse_linear}")
print(f"Mean SSE for Ridge regression: {mean_sse_ridge}")
print(f"Mean SSE for Lasso regression: {mean_sse_lasso}")

# Given by the mean SSE of the three different models, the Lasso regression model had the best model performance. 
# Therefore, the Lasso model is used to predict y using the x_test_regression values


y_pred_lasso = model_lasso.predict(x_test_regression)

np.save('y_pred_lasso', y_pred_lasso)
