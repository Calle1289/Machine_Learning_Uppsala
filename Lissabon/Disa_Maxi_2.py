#%%
import tensorflow as tf 
from keras import layers, models, callbacks, optimizers
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import seaborn as sns
import cv2

#%%
path = 'C:/Users/Disa/OneDrive - Lund University/IST/Machine learning project/'
numpy_train_x = np.load(path+'Xtrain_Classification1.npy')/255.0
numpy_train_y = np.load(path+'ytrain_Classification1.npy')
numpy_test_x = np.load(path+'Xtest_Classification1.npy')/255.0


# Check the shapes of the NumPy arrays
for var, var_name in zip([numpy_test_x, numpy_train_x, numpy_train_y], ['x_test', 'x_train', 'y_train']):
    print(f"{var_name} shape: {var.shape}")


#%%
# Split in training and validation set
x_train, x_validation, y_train, y_validation = train_test_split(numpy_train_x,numpy_train_y,stratify=numpy_train_y, test_size=0.2) #, random_state=42

#%%
x_val_shape = x_validation.reshape(-1,28,28,3)
x_train_shaped = x_train.reshape(-1,28,28,3)

def rotate_images(images, angles):
    rotated_images = []
    for image in images:
        for angle in angles:
            rotated = np.rot90(image, angle // 90)
            rotated_images.append(rotated)
    return np.array(rotated_images)

rotation_angles = [90,180,270]
class_0_indices = (y_train == 0)
class_1_indices = (y_train == 1)

class_0_data = x_train_shaped[class_0_indices]
class_1_data = x_train_shaped[class_1_indices]


rotated_images_class_1 = rotate_images(class_1_data, rotation_angles)

rotated_labels_class_1 = np.tile(1, len(rotated_images_class_1))

# Concatenate the original Class 0 data, original Class 1 data, and rotated Class 1 data
x_train_shaped = np.vstack((class_0_data, class_1_data, rotated_images_class_1))
y_train = np.hstack((y_train[class_0_indices],y_train[class_1_indices], rotated_labels_class_1))


#%%
def check_imbalance(y_classes,name_set):
    '''
    check and print the percentage of the each data class
    '''
    classes, class_counts = np.unique(y_classes, return_counts=True)
    total = class_counts[0]+class_counts[1]
    for class_label, count in zip(classes, class_counts):
        percentage = count/total*100
        print(f"{name_set}: Class{class_label}: {count} instances: {percentage:.2f}% representive percentage")


#check class inbalance
check_imbalance(numpy_train_y,"Full Training Set")
check_imbalance(y_train,"Splittet Training Set")
check_imbalance(y_validation,"Validation Set")

#%%
model = models.Sequential()
model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.1))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.1))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dropout(0.1))
model.add(layers.Dense(2,activation = 'softmax'))

model.summary()

# %%



optimizer =optimizers.Adam(
    learning_rate=0.001,
    weight_decay = 0.03
)

#optimizer = 'adam'
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

early_stopping = callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

history = model.fit(x_train_shaped, y_train, epochs=15, 
                    validation_data=(x_val_shape, y_validation),callbacks=[early_stopping])

#%%

# Evaluate Loss 
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')


#%%
# Evaluation Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(x_val_shape,  y_validation, verbose=2)


plt.tight_layout()


# Show confusion Matrix
y_pred = np.argmax(model.predict(x_val_shape), axis=1)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_validation, y_pred)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Validation Set')
plt.show()


sensitivity = conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[1,0])
specificity = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
balanced_acc = (sensitivity+specificity)/2
print("Balanced Accuracy :",balanced_acc)
balanced_acc1 = balanced_accuracy_score(y_validation, y_pred)