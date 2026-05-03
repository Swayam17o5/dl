import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
# STEP 1: Load MNIST dataset
# =============================================================================
print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Initial Training images shape : {x_train.shape}")
print(f"Initial Training labels shape : {y_train.shape}")
print(f"Initial Test images shape     : {x_test.shape}")
print(f"Initial Test labels shape     : {y_test.shape}")

# =============================================================================
# STEP 2: Preprocess
# =============================================================================

# Reshape to add channel (grayscale → 1 channel)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

print("\nPixel values normalized.")

# One-hot encoding
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test, num_classes)

# =============================================================================
# STEP 3: Build CNN (Simplified for MNIST)
# =============================================================================
input_shape = (28, 28, 1)

model = Sequential([
    Input(shape=input_shape),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax'),
])

model.summary()

# =============================================================================
# STEP 4: Compile
# =============================================================================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# =============================================================================
# STEP 5: Train
# =============================================================================
print("\nTraining the model...")
history = model.fit(
    x_train, y_train,
    epochs=5,   # MNIST trains fast
    batch_size=64,
    validation_data=(x_test, y_test),
)

# =============================================================================
# STEP 6: Evaluate
# =============================================================================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

# =============================================================================
# STEP 7: Predictions
# =============================================================================
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

class_names = [str(i) for i in range(10)]

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
