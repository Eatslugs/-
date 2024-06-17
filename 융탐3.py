import numpy as np
import tensorflow
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Rescaling
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.metrics import Precision, Recall 
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# To get the images and labels from file
with h5py.File('D:/Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

# To convert the labels to categorical 10 classes
labels = utils.to_categorical(labels, 10)

# To convert to desirable type
labels = labels.astype(np.float32)
images = images.astype(np.float32)

#split into train and test 
train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

base_path='E:/'
checkpoint_cb = ModelCheckpoint(base_path + 'best-model-decals.keras', save_best_only=True, save_weights_only=False)

#model
model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer=HeNormal()))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_initializer=HeNormal()))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_initializer=HeNormal()))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax', kernel_initializer=HeNormal())) 
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[ 'accuracy', Precision(), Recall() ])

history = model.fit(train_images, train_labels, epochs=30, validation_data=(test_images, test_labels), callbacks=[checkpoint_cb]) 

model.load_weights(base_path + 'best-model-decals.keras')

loss, accuracy, precision, recall = model.evaluate(test_images, test_labels)
print(f'Accuracy: {accuracy}, Loss: {loss}, Precision: {precision}, Recall: {recall}')
prediction = model.predict(test_images)

#Confusion Matrix
predicted_labels = np.argmax(prediction, axis = 1)
true_labels = np.argmax(test_labels, axis = 1)
cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize = (10, 8))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('predicted labels')
plt.ylabel('true labels')
plt.show()

# loss
plt.plot(history.history['loss'], color='b')
plt.plot(history.history['val_loss'], color='r')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# accuracy
plt.plot(history.history['accuracy'], color='b')
plt.plot(history.history['val_accuracy'], color='r')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
