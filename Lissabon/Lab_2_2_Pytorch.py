import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

# Create a CNN Dividing_data_model class that inherits nn.Module
class Dividing_data_model(nn.Module):
    def __init__(self):
        super(Dividing_data_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(32*5*5, 128)
        self.fc1_dropout = nn.Dropout(dropout_1) # Dropout neurons first layer
        self.fc2 = nn.Linear(128, 16) # Single ouputs for binary classification
        self.fc2_dropout = nn.Dropout(dropout_2) # Dropout neurons second layer
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x) # Applying dropout
        x = F.relu(self.fc2(x))
        x = self.fc2_dropout(x) # Applying dropout
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x
    
if __name__ == '__main__':

    # Load in the dataset
    x_test_classification = np.load('Xtest_Classification2.npy')
    x_train_classification = np.load('Xtrain_Classification2.npy')
    y_train_classification = np.load('ytrain_Classification2.npy')

    # Reshaping and normalizing the dataset
    x_train_class_normalized = x_train_classification / 255.0
    x_train_class_normalized_reshaped = x_train_class_normalized.reshape(-1, 3, 28, 28)
    x_test_class_normalized = x_test_classification / 255.0
    x_test_class_normalized_reshaped = x_test_class_normalized.reshape(-1, 3, 28, 28)

    # Identify the indices of "true" values
    true_indices = np.where(y_train_classification == 2)[0]

    # Extract the "true" data based on identified indices
    true_data = x_train_class_normalized_reshaped[true_indices]

    # Rotate the "true" data by 90 degrees
    rotated_90 = np.rot90(true_data, axes=(2, 3))

    # Rotate the "true" data by 180 degrees
    rotated_180 = np.rot90(rotated_90, axes=(2, 3))

    # Rotate the "true" data by 270 degrees
    rotated_270 = np.rot90(rotated_180, axes=(2, 3))

    # Flip the images
    flipped_2_90 = np.flip(rotated_90, axis=2)
    flipped_2_180 = np.flip(rotated_180, axis=2)
    flipped_2_270 = np.flip(rotated_270, axis=2)

    flipped_3_90 = np.flip(rotated_90, axis=3)
    flipped_3_180 = np.flip(rotated_180, axis=3)
    flipped_3_270 = np.flip(rotated_270, axis=3)
    
    len_twos = len(rotated_90) + len(rotated_180) + len(rotated_270)+ len(flipped_2_90) + len(flipped_2_180) + len(flipped_2_270) + len(flipped_3_90) + len(flipped_3_180) + len(flipped_3_270)
    
    twos = 2 * np.ones(len_twos)

    # Concatenate the rotated data with the original data
    augmented_data = np.concatenate((x_train_class_normalized_reshaped, rotated_90, rotated_180, rotated_270, flipped_2_90, flipped_2_180, flipped_2_270, flipped_3_90, flipped_3_180, flipped_3_270), axis=0)

    augmented_labels = np.concatenate((y_train_classification, twos))

    # Split the dataset into training using stratification
    X_train, X_temp, y_train, y_temp = train_test_split(x_train_class_normalized_reshaped, y_train_classification, shuffle=True, test_size=0.3, random_state=39, stratify=y_train_classification)

    # Split part of the training set into validation
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=39, stratify=y_temp)

    y_train_binary = np.array([])
    y_valid_binary = np.array([])
    y_test_binary = np.array([])

    for i in [[y_train, y_train_binary], [y_valid, y_valid_binary], [y_test, y_test_binary]]:
        for value in i[0]:
            if i <= 2:
                np.append(i[1])
                


    # Define hyperparameters
    epochs = 10 # Number of iterations through the CNNX
    epochs_check = 1 # How often we check how the model is doing during training
    learning_rate = 0.0008 # lr = learning rate (if error dosen't go down after a bunch of iterations (epox), lower our learning rate)
    dropout_1 = 0.3 # The amount of neurons that will be dropped each iteration during training to minimize overfittning
    dropout_2 = 0.3 # The amount of neurons that will be dropped each iteration during training to minimize overfittning
    weight_decay = 0.003 # Stops some of the neurons from being to or not to activated to avoid overfitting
    threshold = 0.5 # The threshhold for binary classification

    # Define variables to keep track of early stopping
    best_valid_loss = float('inf')  # Initialize with a very high value
    patience = 5  # Number of epochs to wait for improvement
    wait = 1  # Counter for the number of epochs without improvement

    # Initialize the Dividing_data_model
    torch.manual_seed(39)
    Dividing_data_model = Dividing_data_model()

    # Convert X features into Float tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    X_valid = torch.FloatTensor(X_valid)

    # Convert the y labels to Long tensors
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    y_valid = torch.FloatTensor(y_valid)

    # Set the criterion of Dividing_data_model to measure the error, how far off the predictions are from the data
    criterion = nn.CrossEntropyLoss()
    
    # Choose The Gradient Decent Optimizer, lr = learning rate (if error dosen't go down after a bunch of iterations (epox), lower our learning rate) 
    optimizer = torch.optim.Adam(Dividing_data_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training the Dividing_data_model section

    # One epoch is one run of all the data through our neural network
    losses = []
    valid_losses = []

    for i in range(epochs):

        # Training phase
        Dividing_data_model.train()
        optimizer.zero_grad()

        # Go forward and get a prediction
        y_pred = Dividing_data_model.forward(X_train).squeeze(1) # Get predicted results

        # Measure the loss/error, gonna be high at frist
        loss = criterion(y_pred, y_train)

        # Keep Track of our losses
        losses.append(loss.detach().numpy())

        # Do some back propagation: take the error rate forward propagation and feed it
        # through the network to fine tune the weights
        loss.backward()
        optimizer.step()

        # Validation phase
        Dividing_data_model.eval()
        with torch.no_grad():
            y_valid_pred = Dividing_data_model(X_valid).squeeze(1)
            binary_predictions = (y_valid_pred >= threshold).long()
            balanced_accuracy = balanced_accuracy_score(y_valid, binary_predictions)
            valid_loss = criterion(y_valid_pred, y_valid)
            valid_losses.append(valid_loss)
    
    Dividing_data_model.eval()
    with torch.no_grad():
        y_test_pred = Dividing_data_model(X_test).squeeze(1)
        binary_predictions = (y_test_pred >= threshold).long()