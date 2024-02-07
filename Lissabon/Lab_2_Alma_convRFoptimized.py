import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from torchvision import transforms
from torchvision.models import resnet18
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.base import BaseEstimator, TransformerMixin

# Create a custom transformer to convert images to features using a pre-trained CNN
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = resnet18(pretrained=True)
        self.model.fc = torch.nn.Identity()  # Remove the fully connected layer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        for image in X:
            image = transform(image)
            image = image.unsqueeze(0)  # Add batch dimension
            feature = self.model(image).detach().numpy()
            features.append(feature)
        return np.vstack(features)

if __name__ == '__main__':
    # Load in the dataset
    x_test_classification = np.load('Xtest_Classification1.npy')
    x_train_classification = np.load('Xtrain_Classification1.npy')
    y_train_classification = np.load('ytrain_Classification1.npy')

    # Split the dataset into training
    X_train, X_temp, y_train, y_temp = train_test_split(x_train_classification, y_train_classification, test_size=0.3, random_state=39)

    # Split part of the training set into validation
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=39)

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor()

    # Extract features from images
    X_train_features = feature_extractor.transform(X_train)
    X_valid_features = feature_extractor.transform(X_valid)
    X_test_features = feature_extractor.transform(X_test)

    # Define hyperparameters for the Random Forest Classifier
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30]
    }

    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=39, n_jobs=-1)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, scoring='balanced_accuracy', n_jobs=-1)

    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train_features, y_train)

    # Get the best hyperparameters from the grid search
    best_params = grid_search.best_params_
    print(f'Best Hyperparameters: {best_params}')

    # Initialize a Random Forest Classifier with the best hyperparameters
    best_rf_classifier = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=39, n_jobs=-1)

    # Train the Random Forest Classifier with the best hyperparameters
    best_rf_classifier.fit(X_train_features, y_train)

    # Make predictions on the validation set
    y_valid_pred = best_rf_classifier.predict(X_valid_features)

    # Calculate balanced accuracy on the validation set
    balanced_accuracy = balanced_accuracy_score(y_valid, y_valid_pred)

    print(f'Validation Balanced Accuracy: {balanced_accuracy}')

    # Extract features from test images
    X_test_features = feature_extractor.transform(x_test_classification)

    # Make predictions on the test set
    y_test_pred = best_rf_classifier.predict(X_test_features)

    # Calculate balanced accuracy on the test set
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)

    print(f'Test Balanced Accuracy: {test_balanced_accuracy}')