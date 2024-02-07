import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load in the dataset
x_test_regression = np.load('Xtest_Classification1.npy')
x_train_regression = np.load('Xtrain_Classification1.npy')
y_train_regression = np.load('ytrain_Classification1.npy')

x_test_regression = x_test_regression.reshape(-1, 28, 28, 3)

x_test_regression = x_test_regression / 255

# Split the dataset into training
X_train, X_temp, y_train, y_temp = train_test_split(x_train_regression, y_train_regression, test_size=0.3, random_state=39)

# Split part of the training set into validation
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=39)



class Model(nn.Module):
    """
    Neural Network Model for training and testing

    # Input layers: 2352
        There are 2352 elements in our training dataset
    
    # Hidden layers: 3
        h1, h2, two hidden layers to begin with

    # Output: 2
        The outdata is either 0 or 1
    """
    def __init__(self, in_features=2352, h1=1500, h2=700, h3=4, out_features=2):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)

    def forward (self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x