import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

base_path='C:/Users/Genesis/Desktop/2024년 융탐_결과물/'
checkpoint_cb=ModelCheckpoint(base_path+'best-model.keras', save_best_only=True)

train_dir = 'C:/Users/Genesis/Desktop/2024년 융탐_결과물/Dat/Train2'
test_dir = 'C:/Users/Genesis/Desktop/2024년 융탐_결과물/Dat/Test2'

test_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(200, 200), batch_size=32, class_mode='categorical',  subset='training')
validation_generator = train_datagen.flow_from_directory(train_dir, target_size=(200, 200), batch_size=32, class_mode='categorical', subset='validation')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(200, 200), batch_size=32, class_mode='categorical')

model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(200, 200, 3)))
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

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same' ))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding = 'same' ))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer=HeNormal()))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_initializer=HeNormal()))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_initializer=HeNormal()))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax', kernel_initializer=HeNormal()))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=[ 'accuracy', Precision(), Recall() ])
history = model.fit(train_generator, validation_data=validation_generator, epochs=30, callbacks=[checkpoint_cb])

model.load_weights(base_path+'best-model.keras')

loss, accuracy, precision, recall = model.evaluate(test_generator)
print(f'Accuracy: {accuracy}, Loss: {loss}, Precision: {precision}, Recall: {recall}')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=test_generator.class_indices)
cmd.plot(cmap='Blues')
plt.show()

plt.plot(history.history['loss'], color='b')
plt.plot(history.history['val_loss'], color='r')
plt.title('Loss')
plt.legend(['Train', 'Test'])
plt.show()

plt.plot(history.history['accuracy'], color='b')
plt.plot(history.history['val_accuracy'], color='r')
plt.title('Accuracy')
plt.legend(['Train', 'Test'])
plt.show()

plt.plot(history.history['precision'], color='b')
plt.plot(history.history['val_precision'], color='r')
plt.title('Accuracy')
plt.legend(['Train', 'Test'])
plt.show()

plt.plot(history.history['recall'], color='b')
plt.plot(history.history['val_recall'], color='r')
plt.title('Accuracy')
plt.legend(['Train', 'Test'])
plt.show()
