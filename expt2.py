import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("TensorFlow Version:", tf.__version__)
print("Keras Version:", keras.__version__)

# 1. Generate a simple synthetic dataset for a binary classification task
np.random.seed(42)
num_samples = 1000
num_features = 10

X = np.random.rand(num_samples, num_features).astype(np.float32)
y = (np.sum(X[:, :5], axis=1) > np.sum(X[:, 5:], axis=1)).astype(np.float32)

print(f"Generated dataset: X.shape={X.shape}, y.shape={y.shape}")

# 2. Define a simple sequential Keras model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(num_features,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid') # Output layer for binary classification
])

# Display model summary
print("\nModel Summary:")
model.summary()

# 3. Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\nModel compiled successfully.")

# 4. Train the model
epochs = 15
batch_size = 32

history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

print(f"\nModel training completed for {epochs} epochs.")

# 5. Evaluate the trained model
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\nModel Evaluation on training data - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# 6. Use the trained model to make predictions
# Predict on a small portion of the data
num_predictions = 5
sample_indices = np.random.choice(num_samples, num_predictions, replace=False)
X_new = X[sample_indices]
y_true = y[sample_indices]

predictions = model.predict(X_new)

print(f"\nPredictions for {num_predictions} samples:")
for i in range(num_predictions):
    predicted_probability = predictions[i][0]
    predicted_class = 1 if predicted_probability > 0.5 else 0
    true_class = int(y_true[i])
    print(f"Sample {i+1}: Predicted Probability = {predicted_probability:.4f}, Predicted Class = {predicted_class}, True Class = {true_class}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

print("PyTorch Version:", torch.__version__)

# 1. Generate a simple synthetic dataset for a binary classification task
np.random.seed(42)
num_samples = 1000
num_features = 10

# Features X (float32)
X_np = np.random.rand(num_samples, num_features).astype(np.float32)
# Labels y (binary 0 or 1, float32 for BCELoss)
y_np = (np.sum(X_np[:, :5], axis=1) > np.sum(X_np[:, 5:], axis=1)).astype(np.float32)

# Convert numpy arrays to PyTorch tensors
X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np).unsqueeze(1) # Add a dimension for binary classification target

print(f"Generated dataset: X.shape={X.shape}, y.shape={y.shape}")

# 2. Define a simple PyTorch neural network for binary classification
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.2)
        self.layer_2 = nn.Linear(64, 32)
        self.dropout_2 = nn.Dropout(0.2)
        self.layer_out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.dropout_1(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout_2(x)
        x = self.sigmoid(self.layer_out(x))
        return x

# Instantiate the model
model = BinaryClassifier(num_features)

print("\nModel Architecture:")
print(model)

# 3. Define Loss Function and Optimizer
criterion = nn.BCELoss() # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nLoss function and optimizer defined successfully.")

# 4. Create DataLoader for training
batch_size = 32
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. Train the model
epochs = 15

print(f"\nStarting model training for {epochs} epochs...")
for epoch in range(epochs):
    model.train() # Set model to training mode
    for inputs, labels in dataloader:
        optimizer.zero_grad() # Zero the gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Optional: Print epoch-wise loss (for brevity, removed for final output)
    # if (epoch + 1) % 5 == 0:
    #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("\nModel training completed.")

# 6. Evaluate the trained model
model.eval() # Set model to evaluation mode
with torch.no_grad(): # Disable gradient calculation for evaluation
    outputs = model(X)
    loss = criterion(outputs, y)
    predicted_classes = (outputs > 0.5).float() # Threshold probabilities to get classes
    accuracy = (predicted_classes == y).float().mean()

print(f"\nModel Evaluation on training data - Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

# 7. Use the trained model to make predictions
num_predictions = 5
sample_indices = np.random.choice(num_samples, num_predictions, replace=False)
X_new = X[sample_indices]
y_true = y_np[sample_indices] # Use original numpy labels for easy comparison

with torch.no_grad():
    predictions = model(X_new)

print(f"\nPredictions for {num_predictions} samples:")
for i in range(num_predictions):
    predicted_probability = predictions[i].item()
    predicted_class = 1 if predicted_probability > 0.5 else 0
    true_class = int(y_true[i])
    print(f"Sample {i+1}: Predicted Probability = {predicted_probability:.4f}, Predicted Class = {predicted_class}, True Class = {true_class}")


print("Ensuring MXNet is installed and applying compatibility patch...")

# Ensure MXNet is installed. This will likely install numpy 1.26.4 or similar.
#!pip install mxnet --quiet

# Apply compatibility patch for numpy.bool
import numpy as np
if not hasattr(np, 'bool'):
    np.bool_ = bool
    np.bool = bool

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader, ArrayDataset

print("MXNet Version:", mx.__version__)
print("NumPy Version:", np.__version__)

# 1. Generate a simple synthetic dataset for a binary classification task
np.random.seed(42)
num_samples = 1000
num_features = 10

X_np = np.random.rand(num_samples, num_features).astype(np.float32)
y_np = (np.sum(X_np[:, :5], axis=1) > np.sum(X_np[:, 5:], axis=1)).astype(np.float32)

# Convert numpy arrays to MXNet NDArrays
X = nd.array(X_np)
y = nd.array(y_np).reshape((-1, 1)) # Reshape for binary classification target

print(f"Generated dataset: X.shape={X.shape}, y.shape={y.shape}")

# 2. Define a simple Gluon Sequential model for binary classification
model = nn.Sequential()
with model.name_scope():
    model.add(nn.Dense(64, activation='relu'))
    model.add(nn.Dropout(0.2))
    model.add(nn.Dense(32, activation='relu'))
    model.add(nn.Dropout(0.2))
    model.add(nn.Dense(1, activation='sigmoid')) # Output layer for binary classification

# 3. Initialize model parameters
model.initialize(mx.init.Xavier(), ctx=mx.cpu())

print("\nModel Architecture:")
print(model)

# 4. Define Loss Function and Optimizer
criterion = gluon.loss.SigmoidBinaryCrossEntropyLoss() # Binary Cross-Entropy Loss with Sigmoid
optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.001})

print("\nLoss function and optimizer defined successfully.")

# 5. Create DataLoader for training
batch_size = 32
dataset = ArrayDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 6. Train the model
epochs = 15

print(f"\nStarting model training for {epochs} epochs...")
for epoch in range(epochs):
    cumulative_loss = 0
    for data, label in dataloader:
        with autograd.record():
            output = model(data)
            loss = criterion(output, label)
        loss.backward()
        optimizer.step(batch_size)
        cumulative_loss += nd.mean(loss).asscalar()

print("\nModel training completed.")

# 7. Evaluate the trained model
output_eval = model(X)
loss_eval = criterion(output_eval, y)

predicted_probabilities = output_eval.asnumpy()
predicted_classes = (predicted_probabilities > 0.5).astype(np.float32)
accuracy = np.mean(predicted_classes == y.asnumpy())

print(f"\nModel Evaluation on training data - Loss: {nd.mean(loss_eval).asscalar():.4f}, Accuracy: {accuracy:.4f}")

# 8. Use the trained model to make predictions
num_predictions = 5
sample_indices = np.random.choice(num_samples, num_predictions, replace=False)
X_new = X[sample_indices]
y_true = y_np[sample_indices]

predictions = model(X_new)

print(f"\nPredictions for {num_predictions} samples:")
for i in range(num_predictions):
    predicted_probability = predictions[i].asscalar()
    predicted_class = 1 if predicted_probability > 0.5 else 0
    true_class = int(y_true[i])
    print(f"Sample {i+1}: Predicted Probability = {predicted_probability:.4f}, Predicted Class = {predicted_class}, True Class = {true_class}")
