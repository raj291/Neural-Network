import numpy as np


def initialize_network(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
    weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
    biases_hidden = np.zeros((1, hidden_size))
    biases_output = np.zeros((1, output_size))
    return weights_input_hidden, weights_hidden_output, biases_hidden, biases_output

W1, W2, b1, b2 = initialize_network(784, 128, 10)


def relu(x):
    return np.maximum(0, x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def one_hot(labels, num_classes):
    one_hot_label = np.zeros(num_classes, dtype=int)
    one_hot_label[labels] = 1
    return one_hot_label

def cross_entropy_loss(output, label):
    label = one_hot(label , 10)
    prob = np.sum(output * label)
    loss = -np.log(prob)
    return loss

def forward_pass(traning_data , W1, W2, b1, b2):    
    traning_data = traning_data.flatten()
    hidden_raw = traning_data @ W1 + b1
    hidden = relu(hidden_raw)
    output_raw = hidden @ W2 + b2
    output = softmax(output_raw)
    return output , hidden, hidden_raw


def relu_derivative(x):
    return (x > 0).astype(float)

def backpropogation(input , hidden , hidden_raw , output , label , W2, W1, b1,b2,learning_rate):
    output_error = output - one_hot(label,10)
    gradient_W2 = hidden.T @ output_error
    gradient_b2 = output_error.flatten()
    hidden_error = output_error @ W2.T
    hidden_error = hidden_error * relu_derivative(hidden_raw)
    gradient_W1 = input.flatten().reshape(1, -1).T @ hidden_error
    gradient_b1 = hidden_error.flatten()

    W1 = W1 - learning_rate * gradient_W1
    W2 = W2 - learning_rate * gradient_W2
    b1 = b1 - learning_rate * gradient_b1
    b2 = b2 - learning_rate * gradient_b2

    return W1, W2 , b1 , b2