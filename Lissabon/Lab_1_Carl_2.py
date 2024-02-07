from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load in the dataset
x_test_regression = np.load('X_test_regression2.npy')
x_train_regression = np.load('X_train_regression2.npy')
y_train_regression = np.load('y_train_regression2.npy')

## Cluster classification with KMeans
# Determine the amount of clusters 
clusters = 2

# Initialize the KMeans class
kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=39)

# Sort data into clusters and give them labels 0 or 1
data_labels = kmeans.fit_predict(x_train_regression)

# Save the clusters in list
clusters_list = []

# Extract the cluser points
for cluster_id in range(0, clusters):
    x_cluster = x_train_regression[data_labels == cluster_id]
    y_cluster = y_train_regression[data_labels == cluster_id]

    clusters_list.append((x_cluster, y_cluster))


## Determine the best model/part 1
# Determent the amount of folds
k = 15

# Create kfolds
kf = KFold(n_splits=k)

# Alpha values for Ridge and Lasso
alpha_Ridge = 0
alpha_Lasso = 0

# Create models
model_Linear_regression = LinearRegression()
model_Ridge = Ridge(alpha=2)
model_Lasso = Lasso(alpha=0.1)

# Make models into interable list
list_models = [model_Linear_regression, model_Ridge, model_Lasso]

# Save model scores
model_mean_mse_scores = []

# Iterate over the clusters
for cluster in clusters_list:

    # Iterate over models
    for model in list_models:

        # Save fold scores
        model_mse_scores = []

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

# Model 1
model_1 = Ridge(alpha=2)

model_1.fit(clusters_list[0][0], clusters_list[0][1])

y_pred_model_1 = model_1.predict(x_test_regression)

# Model 2
model_2 = Ridge(alpha=2)

model_2.fit(clusters_list[1][0], clusters_list[1][1])

y_pred_model_2 = model_2.predict(x_test_regression)

# Create two columns of both models
y_pred_model_1_2 = np.column_stack((y_pred_model_1, y_pred_model_2))

# np.save("y_pred_lasso", y_pred_lasso)