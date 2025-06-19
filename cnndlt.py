import numpy as np

# Sample data (10 images of 1x28x28)
X = np.random.randn(10, 1, 28, 28)  # (samples, channels, height, width)
y = np.random.randint(0, 3, size=(10,))  # 3 classes

# One-hot encode
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y_encoded = one_hot(y, 3)

# Convolution operation
def conv2d(x, kernel):
    h, w = x.shape
    kh, kw = kernel.shape
    output = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(x[i:i+kh, j:j+kw] * kernel)
    return output

# Max pooling (2x2)
def maxpool2d(x, size=2):
    h, w = x.shape
    output = np.zeros((h//size, w//size))
    for i in range(0, h, size):
        for j in range(0, w, size):
            output[i//size, j//size] = np.max(x[i:i+size, j:j+size])
    return output

# ReLU
def relu(x):
    return np.maximum(0, x)

# Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Cross-entropy loss
def cross_entropy(pred, target):
    return -np.sum(target * np.log(pred + 1e-9))

# Initialize weights
kernel = np.random.randn(3, 3)  # 3x3 kernel
fc1_w = np.random.randn(13*13, 64)
fc2_w = np.random.randn(64, 3)

# Forward pass
def forward(x):
    x = x[0]  # Assume single channel
    out = conv2d(x, kernel)
    out = relu(out)
    out = maxpool2d(out)
    out_flat = out.flatten()
    fc1 = relu(np.dot(out_flat, fc1_w))
    out_final = softmax(np.dot(fc1, fc2_w))
    return out_final

# Run on all samples
for i in range(len(X)):
    output = forward(X[i])
    loss = cross_entropy(output, y_encoded[i])
    print(f"Sample {i+1}: Prediction = {np.argmax(output)}, True = {y[i]}, Loss = {loss:.4f}")
