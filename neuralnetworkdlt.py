import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Generate random dataset
np.random.seed(1)
X = np.random.rand(1000, 3)  # 1000 samples, 3 input features
y = np.array([[1] if x[0] + x[1] + x[2] > 1.5 else [0] for x in X])  # simple rule

# Network architecture
input_size = 3
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 1000

# Weight and bias initialization
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # Error
    error = y - final_output

    if epoch % 100 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch} Loss: {loss:.4f}")

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Final evaluation
predictions = (final_output > 0.5).astype(int)
accuracy = np.mean(predictions == y)
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")
