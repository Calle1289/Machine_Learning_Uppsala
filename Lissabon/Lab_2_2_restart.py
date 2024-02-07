import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

x_test_classification = np.load('Xtest_Classification1.npy')
x_train_classification = np.load('Xtrain_Classification1.npy')
y_train_classification = np.load('ytrain_Classification1.npy')

# Reshaping and normalizing the dataset
x_train_class_normalized = x_train_classification / 255.0
x_train_class_normalized_reshaped = x_train_class_normalized.reshape(-1, 28, 28, 3)
x_test_class_normalized = x_test_classification / 255.0
x_test_class_normalized_reshaped = x_test_class_normalized.reshape(-1, 28, 28, 3)

plt.figure(1)
plt.imshow(x_train_class_normalized_reshaped[0])

rotated_90 = np.flip(x_train_class_normalized_reshaped, 2)

plt.figure(2)
plt.imshow(rotated_90[0])

plt.show()