import numpy as np
import tensorflow as tf

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print(f"Initial Training images shape: {x_train.shape}")
print(f"Initial Training labels shape: {y_train.shape}")
print(f"Initial Test images shape: {x_test.shape}")
print(f"Initial Test labels shape: {y_test.shape}")

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print("Pixel values normalized.")

# Convert labels to one-hot encoding
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
print(f"Training labels shape after one-hot encoding: {y_train.shape}")
print(f"Test labels shape after one-hot encoding: {y_test.shape}")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

input_shape = x_train.shape[1:]

model = Sequential()

model.add(Input(shape=input_shape))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
print("CNN model architecture defined successfully.")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("CNN model compiled successfully.")
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=64,
                    validation_data=(x_test, y_test))
print("Model training completed.")
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

if len(y_test.shape) > 1:
    y_true = np.argmax(y_test, axis=1)
else:
    y_true = y_test

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

# 🔹 Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
