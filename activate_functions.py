import numpy as np


def relu(x):
	return np.maximum(0.0, x)


def relu_grad(x):
	return (x > 0).astype(np.float32)


def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def sigmoid_grad(x):
	s = sigmoid(x)
	return s * (1.0 - s)


def tanh_grad(x):
	t = np.tanh(x)
	return 1.0 - t * t