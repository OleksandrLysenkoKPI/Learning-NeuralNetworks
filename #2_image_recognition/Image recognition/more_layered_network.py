import numpy as np
import struct

def read_mnist_images(filename):
    with open(filename, 'rb') as file:
        magic, num, rows, cols = struct.unpack('>IIII', file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)

    return images

def read_mnist_labels(file_path):
    with open(file_path, 'rb') as file:
        magic_number, num_labels = struct.unpack('>II', file.read(8))
        labels = np.fromfile(file, dtype=np.uint8)

    return labels

train_images = read_mnist_images('MNIST dataset/train-images.idx3-ubyte')
train_labels = read_mnist_labels('MNIST dataset/train-labels.idx1-ubyte')

test_images = read_mnist_images('MNIST dataset/t10k-images.idx3-ubyte')
test_labels = read_mnist_labels('MNIST dataset/t10k-labels.idx1-ubyte')

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_images_flatten = train_images.reshape(train_images.shape[0], -1)
test_images_flatten = test_images.reshape(test_images.shape[0], -1)
x_train = train_images_flatten
x_test = test_images_flatten

def one_hot_encoding(labels):
    one_hot_labels = np.zeros((labels.shape[0], 10))
    one_hot_labels[np.arange(labels.shape[0]), labels] = 1

    return one_hot_labels

y_train = one_hot_encoding(train_labels)
y_test = one_hot_encoding(test_labels)

# Uses He initialization for weight initialization
def initialize_parameters(input_size, hidden_size1, hidden_size2, output_size):
    W1 = np.random.randn(hidden_size1, input_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((hidden_size1, 1))

    W2 = np.random.randn(hidden_size2, hidden_size1) * np.sqrt(2. / hidden_size1)
    b2 = np.zeros((hidden_size2, 1))

    W3 = np.random.randn(output_size, hidden_size2) * np.sqrt(2. / hidden_size2)
    b3 = np.zeros((output_size, 1))

    return W1, b1, W2, b2, W3, b3

def RelU(Z):
    return np.maximum(0, Z)

def RelU_deriv(A):
    return A > 0

def softmax(Z):
    # Shift values by subtracting the maximum for numerical stability
    shiftZ = Z - np.max(Z, axis=0, keepdims=True)
    exps = np.exp(shiftZ)
    return exps / np.sum(exps, axis=0, keepdims=True)

def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    X = X.T

    Z1 = np.dot(W1, X) + b1
    A1 = RelU(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = RelU(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)

    return A3, A2, A1

def compute_loss(Y, Y_hat):
    m = Y.shape[0] # Number of samples
    # Compute cross-entropy loss
    loss = -1 / m * np.sum(Y.T * np.log(Y_hat))
    return loss

def compute_gradients(X, Y, A3, A2, A1, W3, W2):
    m = X.shape[0]

    dZ3 = A3 - Y.T
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = np.dot(W3.T, dZ3) * RelU_deriv(A2)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * RelU_deriv(A1)
    dW1 = (1 / m) * np.dot(dZ1, X)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

def update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    return W1, b1, W2, b2, W3, b3

input_size = x_train.shape[1]
hidden_size1 = 128
hidden_size2 = 64
output_size = 10

num_epochs = 100
learning_rate = 0.5

W1, b1, W2, b2, W3, b3 = initialize_parameters(input_size, hidden_size1, hidden_size2, output_size)

for i in range(num_epochs):
    A3, A2, A1 = forward_propagation(x_train, W1, b1, W2, b2, W3, b3)
    loss = compute_loss(y_train, A3)

    dW1, db1, dW2, db2, dW3, db3 = compute_gradients(x_train, y_train, A3, A2, A1, W3, W2)

    W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)

    print(f"Epoch {i}: loss = {loss:.6f}")

test_predictions, _, _ = forward_propagation(x_test, W1, b1, W2, b2, W3, b3)

predicted_labels = np.argmax(test_predictions, axis=0)

accuracy = np.mean(predicted_labels == test_labels)
print("Accuracy: {:.2f}%".format(accuracy * 100))

test_loss = compute_loss(y_test, test_predictions)
print(f"Loss when compared to test set = {test_loss}")