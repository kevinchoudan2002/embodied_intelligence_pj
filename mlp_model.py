import numpy as np

from activate_functions import relu, relu_grad, sigmoid, sigmoid_grad, tanh_grad
from encode_loss_calc import softmax


def get_activation(name):
    normalized = name.strip().lower()
    activations = {
        "relu": (relu, relu_grad),
        "sigmoid": (sigmoid, sigmoid_grad),
        "tanh": (np.tanh, tanh_grad),
    }
    if normalized not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[normalized], normalized


def get_epoch_lr(initial_lr, lr_decay, epoch):
    return initial_lr / (1.0 + lr_decay * (epoch - 1))


def accuracy(probs, y_true):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y_true)


class MLP3Layer:
    def __init__(self, input_dim, hidden_dim, output_dim, activation="relu", seed=42):
        (self.activation_fn, self.activation_grad), self.activation_name = get_activation(activation)

        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        scale3 = np.sqrt(2.0 / hidden_dim)

        self.W1 = (rng.standard_normal((input_dim, hidden_dim)) * scale1).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden_dim, hidden_dim)) * scale2).astype(np.float32)
        self.b2 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W3 = (rng.standard_normal((hidden_dim, output_dim)) * scale3).astype(np.float32)
        self.b3 = np.zeros((1, output_dim), dtype=np.float32)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = self.activation_fn(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.activation_fn(z2)
        logits = a2 @ self.W3 + self.b3
        probs = softmax(logits)
        cache = (x, z1, a1, z2, a2, logits, probs)
        return probs, cache

    def backward(self, cache, dlogits):
        x, z1, a1, z2, a2, _, _ = cache

        dW3 = a2.T @ dlogits
        db3 = np.sum(dlogits, axis=0, keepdims=True)

        da2 = dlogits @ self.W3.T
        dz2 = da2 * self.activation_grad(z2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.activation_grad(z1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3

    def step(self, grads, lr, weight_decay=0.0):
        dW1, db1, dW2, db2, dW3, db3 = grads
        if weight_decay > 0.0:
            dW1 = dW1 + weight_decay * self.W1
            dW2 = dW2 + weight_decay * self.W2
            dW3 = dW3 + weight_decay * self.W3
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
