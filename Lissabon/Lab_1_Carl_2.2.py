from sklearn.linear_model import LinearRegression
from sklearn.cluster import AffinityPropagation
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error

# Load in the dataset
x_test_regression = np.load('X_test_regression2.npy')
x_train_regression = np.load('X_train_regression2.npy')
y_train_regression = np.load('y_train_regression2.npy')

# Initialize AffinityPropagation
affinity_propagation = AffinityPropagation()

data_labels = affinity_propagation.fit_predict(x_train_regression)

# Extract the cluster points
clusters_list = []
for cluster_id in np.unique(data_labels):
    x_cluster = x_train_regression[data_labels == cluster_id]
    y_cluster = y_train_regression[data_labels == cluster_id]

    clusters_list.append((x_cluster, y_cluster))

## Determine the best model/part 1
# Determent the amount of folds
k = 15

# Create kfolds
kf = KFold(n_splits=k)

# Alpha values for Ridge and Lasso
ridge_alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 20]
lasso_alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 20]

# Create models
model_Linear_regression = LinearRegression()
model_Ridge = Ridge()
model_Lasso = Lasso()

# Make models into interable list
list_models = [(model_Linear_regression, None), (model_Ridge, ridge_alphas), (model_Lasso, lasso_alphas)]

# Save model scores
model_mean_mse_scores = []

# Save optimal alpha values
best_alpha_values = []

# Iterate over the clusters
for cluster in clusters_list:

    # Iterate over models
    for model, alphas in list_models:

        # Save fold scores
        model_mse_scores = []

        # If alphas is not None, perform alpha selection
        if alphas is not None:
            # Create a dictionary of parameters to search
            param_grid = {'alpha': alphas}

            # Perform Grid Search for the best alpha value
            grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=k)
            grid_search.fit(x_train_regression, y_train_regression)

            # Get the best alpha
            best_alpha = grid_search.best_params_['alpha']
            model.set_params(alpha=best_alpha)
            best_alpha_values.append(best_alpha)

        # Split the training set into test and training sets
        for index_train, index_test in kf.split(cluster[0]):
            x_train_fold, x_test_fold = cluster[0][index_train], cluster[0][index_test]
            y_train_fold, y_test_fold = cluster[1][index_train], cluster[1][index_test]

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
print(best_alpha_values)

# Model 1
model_1 = Ridge(alpha=best_alpha_values[0])

model_1.fit(clusters_list[0][0], clusters_list[0][1])

y_pred_model_1 = model_1.predict(x_test_regression)

# Model 2
model_2 = Ridge(alpha=best_alpha_values[0])

model_2.fit(clusters_list[1][0], clusters_list[1][1])

y_pred_model_2 = model_2.predict(x_test_regression)

# Create two columns of both models
y_pred_model_1_2_affinity = np.column_stack((y_pred_model_1, y_pred_model_2))

np.save("y_pred_model_1_2_", y_pred_model_1_2_affinity)