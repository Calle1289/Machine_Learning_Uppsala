from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load in the dataset
x_test_regression = np.load('X_test_regression1.npy')
x_train_regression = np.load('X_train_regression1.npy')
y_train_regression = np.load('y_train_regression1.npy')

# Determent the amount of folds
k = 15

# Creat kfolds
kf = KFold(n_splits=k)

# Create models
model_Linear_regression = LinearRegression()
model_Ridge = Ridge(alpha=2)
model_Lasso = Lasso(alpha=0.1)

# Make models into interable list
list_models = [model_Linear_regression, model_Ridge, model_Lasso]

# Save model scores
model_mean_mse_scores = []

# Iterate over models
for model in list_models:

    # Save fold scores
    model_mse_scores = []

    # Split the training set into test and training sets
    for index_train, index_test in kf.split(x_train_regression):
        x_train_fold, x_test_fold = x_train_regression[index_train], x_train_regression[index_test]
        y_train_fold, y_test_fold = y_train_regression[index_train], y_train_regression[index_test]

        # Fit the model to the training data
        model.fit(x_train_fold, y_train_fold)

        # Predict the results from the model on the fold test data
        y_pred_fold = model.predict(x_test_fold)

        # Calculate the mean squared error for each fold
        mse_fold = mean_squared_error(y_test_fold, y_pred_fold)
        model_mse_scores.append(mse_fold)

    mean_mse_fold = sum(model_mse_scores) / k
    model_mean_mse_scores.append(mean_mse_fold)

print(model_mean_mse_scores)

model_Lasso.fit(x_train_regression, y_train_regression)

y_pred_lasso = model_Lasso.predict(x_test_regression)

# np.save("y_pred_lasso", y_pred_lasso)