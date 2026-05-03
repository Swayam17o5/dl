# MCP Neuron Function
def mcp_neuron(x1, x2, w1, w2, threshold):
    net = x1*w1 + x2*w2
    if net >= threshold:
        return 1
    else:
        return 0


inputs = [(0,0), (0,1), (1,0), (1,1)]

# -------- AND GATE --------
print("AND Gate using MCP Neuron")
print("x1  x2  Output")

w1 = 1
w2 = 1
threshold = 2

for x1, x2 in inputs:
    output = mcp_neuron(x1, x2, w1, w2, threshold)
    print(x1, " ", x2, "   ", output)


# -------- OR GATE --------
print("\nOR Gate using MCP Neuron")
print("x1  x2  Output")

threshold = 1

for x1, x2 in inputs:
    output = mcp_neuron(x1, x2, w1, w2, threshold)
    print(x1, " ", x2, "   ", output)

# MCP Neuron for Loan Approval
def loan_approval(income, credit_score):
    w1 = 1
    w2 = 1
    threshold = 2

    net = income*w1 + credit_score*w2

    if net >= threshold:
        return "Loan Approved"
    else:
        return "Loan Rejected"


# Test all possible cases
inputs = [(0,0), (0,1), (1,0), (1,1)]

print("Income  Credit  Decision")
for income, credit in inputs:
    result = loan_approval(income, credit)
    print(income, "     ", credit, "    ", result)

# Step activation function
def step(net):
    return 1 if net >= 0 else 0


# Perceptron training function
def train_perceptron(data, targets, lr=1, epochs=10):
    w1, w2, bias = 0, 0, 0   # initial weights

    print(f"Initial weights: w1={w1}, w2={w2}, bias={bias}\n")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}:")
        for i, ((x1, x2), t) in enumerate(zip(data, targets)):
            net = x1*w1 + x2*w2 + bias
            y = step(net)
            error = t - y

            print(f"  Input: ({x1}, {x2}), Target: {t}")
            print(f"    Net: {x1}*{w1} + {x2}*{w2} + {bias} = {net}")
            print(f"    Output (y): {y}")
            print(f"    Error (t-y): {error}")

            w1 += lr * error * x1
            w2 += lr * error * x2
            bias += lr * error

            print(f"    Updated weights: w1={w1}, w2={w2}, bias={bias}\n")

    return w1, w2, bias

# AND gate training data
inputs = [(0,0), (0,1), (1,0), (1,1)]
targets_and = [0, 0, 0, 1]

w1, w2, b = train_perceptron(inputs, targets_and)

print("AND Gate using Perceptron Learning")
print("x1  x2  Output")

for x1, x2 in inputs:
    y = step(x1*w1 + x2*w2 + b)
    print(x1, " ", x2, "   ", y)

    # OR gate training data
targets_or = [0, 1, 1, 1]

w1, w2, b = train_perceptron(inputs, targets_or)

print("\nOR Gate using Perceptron Learning")
print("x1  x2  Output")

for x1, x2 in inputs:
    y = step(x1*w1 + x2*w2 + b)
    print(x1, " ", x2, "   ", y)

    import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_neurons = 2
hidden_neurons = 2
output_neurons = 1

np.random.seed(123)

# Weights and biases for hidden layer
wh = np.random.uniform(size=(input_neurons, hidden_neurons)) # (2, 2)
bh = np.random.uniform(size=(1, hidden_neurons)) # (1, 2)

# Weights and biases for output layer
wo = np.random.uniform(size=(hidden_neurons, output_neurons)) # (2, 1)
bo = np.random.uniform(size=(1, output_neurons)) # (1, 1)

learning_rate = 0.1
epochs = 10000

print("\n--- Training XOR Gate using Multi-layer Perceptron (Backpropagation) ---")
print(f"Initial Weights Hidden (wh):\n{wh}")
print(f"Initial Bias Hidden (bh):\n{bh}")
print(f"Initial Weights Output (wo):\n{wo}")
print(f"Initial Bias Output (bo):\n{bo}\n")

for epoch in range(epochs):
    # Forward Propagation
    # Hidden layer
    net_h = np.dot(X, wh) + bh
    act_h = sigmoid(net_h)

    # Output layer
    net_o = np.dot(act_h, wo) + bo
    output = sigmoid(net_o)

    # Backpropagation
    # Calculate error
    error_output = y - output
    d_output = error_output * sigmoid_derivative(output)

    # Calculate error for hidden layer
    error_hidden = d_output.dot(wo.T)
    d_hidden = error_hidden * sigmoid_derivative(act_h)

    # Update weights and biases
    wo += act_h.T.dot(d_output) * learning_rate
    bo += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(d_hidden) * learning_rate
    bh += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        loss = np.mean(np.square(error_output))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\n--- Training Complete ---")
print(f"Final Weights Hidden (wh):\n{wh}")
print(f"Final Bias Hidden (bh):\n{bh}")
print(f"Final Weights Output (wo):\n{wo}")
print(f"Final Bias Output (bo):\n{bo}\n")

print("\n--- XOR Gate using MLP Predictions ---")
print("Input   Expected Output   Predicted Output (Rounded)")

net_h_test = np.dot(X, wh) + bh
act_h_test = sigmoid(net_h_test)

net_o_test = np.dot(act_h_test, wo) + bo
predictions = sigmoid(net_o_test)

for i in range(len(X)):
    print(f"{X[i]}        {y[i][0]}              {np.round(predictions[i][0]):.0f} ({predictions[i][0]:.4f})")

    import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XNOR training data
X_xnor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xnor = np.array([[1], [0], [0], [1]])

input_neurons = 2
hidden_neurons = 4
output_neurons = 1

np.random.seed(123)

wh_xnor = np.random.uniform(size=(input_neurons, hidden_neurons))
bh_xnor = np.random.uniform(size=(1, hidden_neurons))

# Weights and biases for output layer
wo_xnor = np.random.uniform(size=(hidden_neurons, output_neurons))
bo_xnor = np.random.uniform(size=(1, output_neurons))

learning_rate = 0.2
epochs = 50000

print("\n--- Training XNOR Gate using Multi-layer Perceptron (Backpropagation) ---")
print(f"Initial Weights Hidden (wh_xnor):\n{wh_xnor}")
print(f"Initial Bias Hidden (bh_xnor):\n{bh_xnor}")
print(f"Initial Weights Output (wo_xnor):\n{wo_xnor}")
print(f"Initial Bias Output (bo_xnor):\n{bo_xnor}\n")

for epoch in range(epochs):
    # Hidden layer
    net_h_xnor = np.dot(X_xnor, wh_xnor) + bh_xnor
    act_h_xnor = sigmoid(net_h_xnor)

    # Output layer
    net_o_xnor = np.dot(act_h_xnor, wo_xnor) + bo_xnor
    output_xnor = sigmoid(net_o_xnor)

    error_output_xnor = y_xnor - output_xnor
    d_output_xnor = error_output_xnor * sigmoid_derivative(output_xnor)

    # Calculate error for hidden layer
    error_hidden_xnor = d_output_xnor.dot(wo_xnor.T)
    d_hidden_xnor = error_hidden_xnor * sigmoid_derivative(act_h_xnor)

    # Update weights and biases
    wo_xnor += act_h_xnor.T.dot(d_output_xnor) * learning_rate
    bo_xnor += np.sum(d_output_xnor, axis=0, keepdims=True) * learning_rate
    wh_xnor += X_xnor.T.dot(d_hidden_xnor) * learning_rate
    bh_xnor += np.sum(d_hidden_xnor, axis=0, keepdims=True) * learning_rate

    if epoch % 5000 == 0: # Print loss less frequently due to increased epochs
        loss = np.mean(np.square(error_output_xnor))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\n--- Training Complete ---")
print(f"Final Weights Hidden (wh_xnor):\n{wh_xnor}")
print(f"Final Bias Hidden (bh_xnor):\n{bh_xnor}")
print(f"Final Weights Output (wo_xnor):\n{wo_xnor}")
print(f"Final Bias Output (bo_xnor):\n{bo_xnor}\n")

print("\n--- XNOR Gate using MLP Predictions ---")
print("Input   Expected Output   Predicted Output (Rounded)")

net_h_test_xnor = np.dot(X_xnor, wh_xnor) + bh_xnor
act_h_test_xnor = sigmoid(net_h_test_xnor)

net_o_test_xnor = np.dot(act_h_test_xnor, wo_xnor) + bo_xnor
predictions_xnor = sigmoid(net_o_test_xnor)

for i in range(len(X_xnor)):
    print(f"{X_xnor[i]}        {y_xnor[i][0]}              {np.round(predictions_xnor[i][0]):.0f} ({predictions_xnor[i][0]:.4f})")