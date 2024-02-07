import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

if __name__ == '__main__':
    # Load in the dataset
    x_test_classification = np.load('Xtest_Classification1.npy')
    x_train_classification = np.load('Xtrain_Classification1.npy')
    y_train_classification = np.load('ytrain_Classification1.npy')

    # Reshaping and normalizing the dataset
    x_train_class_normalized = x_train_classification / 255.0
    x_train_class_normalized_reshaped = x_train_class_normalized.reshape(-1, 3, 28, 28)
    x_test_class_normalized = x_test_classification / 255.0
    x_test_class_normalized_reshaped = x_test_class_normalized.reshape(-1, 3, 28, 28)

    # Identify the indices of "true" values
    true_indices = np.where(y_train_classification == 1)[0]

    # Extract the "true" data based on identified indices
    true_data = x_train_class_normalized_reshaped[true_indices]

    # Rotate the "true" data by 90 degrees
    rotated_90 = np.rot90(true_data, axes=(2, 3))

    # Rotate the "true" data by 180 degrees
    rotated_180 = np.rot90(rotated_90, axes=(2, 3))

    # Rotate the "true" data by 270 degrees
    rotated_270 = np.rot90(rotated_180, axes=(2, 3))

    # Concatenate the rotated data with the original data
    augmented_data = np.concatenate((x_train_class_normalized_reshaped, rotated_90, rotated_180, rotated_270), axis=0)

    augmented_labels = np.concatenate((y_train_classification, np.ones(sum([len(rotated_90), len(rotated_180), len(rotated_270)]))))

    # Split the dataset into training using stratification
    X_train, X_temp, y_train, y_temp = train_test_split(augmented_data, augmented_labels, shuffle=True, test_size=0.3, random_state=39, stratify=augmented_labels)

    # Split part of the training set into validation
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=39, stratify=y_temp)

    # Reshape the data for Random Forest
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    x_test_class_normalized_reshaped_flat = x_test_class_normalized_reshaped.reshape(x_test_class_normalized_reshaped.shape[0], -1)

    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=39, n_jobs=-1)

    # Train the Random Forest Classifier
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the validation set
    y_valid_pred = rf_classifier.predict(X_valid)

    # Calculate balanced accuracy on the validation set
    balanced_accuracy = balanced_accuracy_score(y_valid, y_valid_pred)

    print(f'Validation Balanced Accuracy: {balanced_accuracy}')

    # Make predictions on the test set
    y_test_pred = rf_classifier.predict(X_test)

    # Calculate balanced accuracy on the test set
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)

    print(f'Test Balanced Accuracy: {test_balanced_accuracy}')

    y_test_pred_hand_in = rf_classifier.predict(x_test_class_normalized_reshaped_flat)

    np.save('y_test_pred_hand_in', y_test_pred_hand_in)