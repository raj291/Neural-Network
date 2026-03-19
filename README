# Neural Network from Scratch — MNIST Digit Recogniser

A handwritten digit classifier built from scratch using only Python and NumPy. No PyTorch. No TensorFlow. Just math.

This project was built to understand how neural networks actually work — from raw pixel data all the way to a prediction — by implementing every piece by hand.

---

## What It Does

Takes a 28×28 pixel image of a handwritten digit and predicts which digit it is (0 through 9).

It learns to do this by training on 60,000 handwritten digit images from the MNIST dataset, adjusting its internal weights through backpropagation until it gets good at recognising patterns.

---

## How It Works

### Architecture

```
Input Layer  →  784 neurons  (28×28 pixels flattened)
Hidden Layer →  128 neurons  (learns features like curves and edges)
Output Layer →   10 neurons  (one per digit, outputs probabilities)
```

### The Process

1. **Forward Pass** — An image flows through the network layer by layer. Each layer multiplies the input by its weights, adds a bias, and passes the result through an activation function. The final output is 10 probabilities that sum to 1.

2. **Loss Calculation** — Cross-entropy loss measures how wrong the prediction was. A confident correct prediction produces a small loss. A wrong prediction produces a large loss.

3. **Backpropagation** — The error signal travels backwards through the network. Calculus (the chain rule) figures out how much each weight contributed to the mistake.

4. **Gradient Descent** — Every weight is nudged slightly in the direction that reduces the loss. Repeat 60,000 times per epoch.

### Key Concepts

| Concept | What it does |
|---|---|
| Weights | Store what the network has learned |
| Biases | Give each neuron an adjustable threshold |
| ReLU | Hidden layer activation — fires if positive, silent if negative |
| Softmax | Converts raw output scores into probabilities |
| Cross-entropy loss | Measures prediction error |
| Backpropagation | Calculates which weights caused the error |
| Gradient descent | Updates weights to reduce the error |
| Epoch | One full pass through all 60,000 training images |

---

## Folder Structure

```
CNN/
├── train_data/
│   ├── train-images-idx3-ubyte
│   └── train-labels-idx1-ubyte
├── test_data/
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── load_data.py
├── network.py
├── train.py
└── test.py
```

---

## Dataset

MNIST — Modified National Institute of Standards and Technology database.

- 60,000 training images
- 10,000 test images
- Every image is 28×28 pixels, grayscale
- Every pixel is a value from 0 (black) to 255 (white)

Download from: http://yann.lecun.com/exdb/mnist/

---

## Files

### `load_data.py`
Reads the raw MNIST binary files (IDX format) and loads them into numpy arrays. Normalises pixel values from 0–255 down to 0–1 for stable training.

Produces four arrays:
- `training_data` — shape (60000, 28, 28)
- `training_label` — shape (60000,)
- `testing_data` — shape (10000, 28, 28)
- `testing_label` — shape (10000,)

### `network.py`
Contains every function the neural network needs:

- `relu(x)` — activation function for the hidden layer
- `relu_derivative(x)` — used during backpropagation
- `softmax(x)` — converts output scores to probabilities
- `one_hot(label, num_classes)` — converts a digit label to a vector
- `cross_entropy_loss(output, label)` — measures prediction error
- `initialize_network(input_size, hidden_size, output_size)` — creates random weights and zero biases
- `forward_pass(image, W1, W2, b1, b2)` — runs an image through the network
- `backpropagation(...)` — computes gradients and updates weights

### `train.py`
The training loop. Imports data and network functions, then runs forward pass → loss → backpropagation for every image across every epoch. Prints average loss per epoch.

### `test.py`
Evaluates the trained network on 10,000 images it has never seen before. Reports final accuracy.

---

## How to Run

**Step 1 — Install dependencies**
```
pip install numpy
```

**Step 2 — Download the MNIST dataset**

Download all 4 files from http://yann.lecun.com/exdb/mnist/ and extract them into the correct folders.

**Step 3 — Train the network**
```
python train.py
```

You will see the loss printed after each epoch. It should decrease over time:
```
Epoch 1, Loss: 0.250
Epoch 2, Loss: 0.102
Epoch 3, Loss: 0.071
Epoch 4, Loss: 0.053
...
```

**Step 4 — Test the network**
```
python test.py
```

---

## Expected Results

| Epoch | Approximate Loss |
|---|---|
| 1 | 0.25 |
| 5 | 0.03 |
| 10 | 0.02 |
| 20 | 0.01 |

A well trained network achieves above 95% accuracy on the test set.

---

## Why This Is AI

This is machine learning — a subset of AI where a system learns from data instead of being programmed with explicit rules. The network was never told what a 7 looks like. It discovered that on its own by seeing thousands of examples and adjusting its weights to minimise its mistakes.

The hidden layer spontaneously learns to detect features like edges, curves, and loops — the building blocks of handwritten digits. This is called representation learning and it is the foundation of modern deep learning.