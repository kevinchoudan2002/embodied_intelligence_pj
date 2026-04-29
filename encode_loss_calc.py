import numpy as np


def one_hot(y, num_classes):
	encoded = np.zeros((y.shape[0], num_classes), dtype=np.float32)
	encoded[np.arange(y.shape[0]), y] = 1.0
	return encoded

def softmax(logits):
	shifted = logits - np.max(logits, axis=1, keepdims=True)
	exp_scores = np.exp(shifted)
	return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def cross_entropy_loss(probs, y_true):
	n = y_true.shape[0]
	picked = probs[np.arange(n), y_true]
	return -np.mean(np.log(np.clip(picked, 1e-12, 1.0)))

def mse_loss(probs, y_true):
	y_onehot = one_hot(y_true, probs.shape[1])
	return np.mean(np.sum((probs - y_onehot) ** 2, axis=1))

def dsoftmax_from_dprobs(probs, dprobs):
	inner = np.sum(dprobs * probs, axis=1, keepdims=True)
	return probs * (dprobs - inner)

def compute_loss_and_dlogits(probs, y_true, loss_name):
	n = y_true.shape[0]
	y_onehot = one_hot(y_true, probs.shape[1])

	if loss_name == "cross_entropy":
		loss = cross_entropy_loss(probs, y_true)
		dlogits = (probs - y_onehot) / n
	elif loss_name == "mse":
		loss = mse_loss(probs, y_true)
		dprobs = 2.0 * (probs - y_onehot) / n
		dlogits = dsoftmax_from_dprobs(probs, dprobs)
	else:
		raise ValueError(f"Unsupported loss: {loss_name}")

	return loss, dlogits

def l2_penalty(model):
	return 0.5 * (
		np.sum(model.W1 * model.W1)
		+ np.sum(model.W2 * model.W2)
		+ np.sum(model.W3 * model.W3)
	)