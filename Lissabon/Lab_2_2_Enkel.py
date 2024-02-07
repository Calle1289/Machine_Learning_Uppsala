import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

# Create a CNN CNNmodel class that inherits nn.Module
class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16*13*13, 16)
        self.fc2 = nn.Linear(16, 2) # Single ouputs for binary classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
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

    flipped_horizontal = np.flip(true_data, 3)
    flipped_vertical = np.flip(true_data, 2)

    # Concatenate the rotated data with the original data
    augmented_data = np.concatenate((x_train_class_normalized_reshaped, flipped_horizontal, flipped_vertical), axis=0)

    augmented_labels = np.concatenate((y_train_classification, np.ones(sum([len(flipped_horizontal), len(flipped_vertical)]))))

    # Split the dataset into training
    X_train, X_temp, y_train, y_temp = train_test_split(augmented_data, augmented_labels, shuffle=True, test_size=0.3, random_state=39, stratify=augmented_labels)

    # Split part of the training set into validation
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=39, stratify=y_temp)

    # Define hyperparameters
    epochs = 10 # Number of iterations through the CNNX
    epochs_check = 1 # How often we check how the model is doing during training
    learning_rate = 0.0001 # lr = learning rate (if error dosen't go down after a bunch of iterations (epox), lower our learning rate)
    weight_decay = 0.003 # Stops some of the neurons from being to or not to activated to avoid overfitting
    threshold = 0.5 # The threshhold for binary classification

    # Define variables to keep track of early stopping
    best_valid_loss = float('inf')  # Initialize with a very high value
    patience = 5  # Number of epochs to wait for improvement
    wait = 1  # Counter for the number of epochs without improvement

    # Initialize the CNNmodel
    torch.manual_seed(39)
    CNNmodel = CNNmodel()

    # Convert X features into Float tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    X_valid = torch.FloatTensor(X_valid)

    # Convert the y labels to Long tensors
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    y_valid = torch.LongTensor(y_valid)

    # Set the criterion of CNNmodel to measure the error, how far off the predictions are from the data
    criterion = nn.CrossEntropyLoss()
    
    # Choose The Gradient Decent Optimizer, lr = learning rate (if error dosen't go down after a bunch of iterations (epox), lower our learning rate) 
    optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=learning_rate)

    # Training the CNNmodel section

    # One epoch is one run of all the data through our neural network
    losses = []
    valid_losses = []

    for i in range(epochs):

        # Training phase
        CNNmodel.train()
        optimizer.zero_grad()

        # Go forward and get a prediction
        y_pred = CNNmodel.forward(X_train) # Get predicted results

        # Measure the loss/error, gonna be high at frist
        loss = criterion(y_pred, y_train)

        # Keep Track of our losses
        losses.append(loss.detach())

        # Do some back propagation: take the error rate forward propagation and feed it
        # through the network to fine tune the weights
        loss.backward()
        optimizer.step()

        # Validation phase
        CNNmodel.eval()
        with torch.no_grad():
            y_valid_pred = CNNmodel(X_valid)
            y_valid_pred_softmax = y_valid_pred.softmax(dim=1)
            y_valid_pred_softmax_argmax = y_valid_pred_softmax.argmax(dim=1)
            balanced_accuracy = balanced_accuracy_score(y_valid, y_valid_pred_softmax_argmax)
            valid_loss = criterion(y_valid_pred, y_valid)
            valid_losses.append(valid_loss)

         # Check for early stopping based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss  # Update the best validation loss
            wait = 0  # Reset the counter
        else:
            wait += 1

        if wait >= patience:
            print(f'Early stopping after {i+1} epochs without improvement.')
            break  # Stop training
        
        # Print CNNmodel status every 10 epoch
        if i % epochs_check == 0:
            print(f'Epoch: {i} Loss: {loss} Validation loss: {valid_loss}')
    
    CNNmodel.eval()
    with torch.no_grad():
        y_test_pred = CNNmodel(X_test)
        y_test_pred_softmax = y_test_pred.softmax(dim=1)
        y_test_pred_softmax_argmax = y_test_pred_softmax.argmax(dim=1)
        balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred_softmax_argmax)
        test_loss = criterion(y_test_pred, y_test)

    print(f'Final Test Loss: {test_loss} Balanced accuracy: {balanced_accuracy}')

    # Additional code for visualizing performance
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
    # plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')y_test_pred_softmax_argmax
    # plt.legend()
    # plt.show()

