# ... other parts of your code ...

class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc1_dropout = nn.Dropout(dropout_1) 
        self.fc2 = nn.Linear(256, 64) 
        self.fc2_dropout = nn.Dropout(dropout_2) 
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x) 
        x = F.relu(self.fc2(x))
        x = self.fc2_dropout(x) 
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# ... rest of your code ...

# For class weights
class_counts = np.bincount(y_train_classification.astype(int))
class_weights = 1./class_counts
weights = class_weights[y_train_classification.astype(int)]
sample_weights = torch.FloatTensor(weights)

# Adjust the BCELoss function to include weights
criterion = nn.BCELoss(weight=sample_weights)

# ... rest of your code ...
