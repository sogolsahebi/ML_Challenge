import numpy as np


# Multi-class Logistic Regression Model

# Softmax
def softmax(z):
    """
    Softmax function used instead of sigmoid since we are dealing with
    multiple classes
    """
    axis_max = np.max(z, axis=1, keepdims=True)
    e_z = np.exp(z - axis_max)
    return e_z / np.sum(e_z, axis=1, keepdims=True)


def compute_entropy_loss(X, t, W):
    """
    Function to compute the cross-entropy loss
    """
    samples_n = X.shape[0]
    z = np.dot(X, W)
    prob = softmax(z)
    loss = -np.sum(t * np.log(
        prob + 1e-5)) / samples_n  # Just to make sure that we don't log(0)
    return loss


def compute_grad(X, t, W):
    """
    Computes the gradient of cross entropy loss
    """
    samples_n = X.shape[0]
    z = np.dot(X, W)
    prob = softmax(z)
    grad = np.dot(X.T, (prob - t)) / samples_n
    return grad


def accuracy(predictions, labels):
    return np.mean(predictions == np.argmax(labels, axis=1))


def grad_descent(X_train, t_train, alpha=0.1, n_iter=30,
                 initial_weights=None):
    if initial_weights is None:
        W = np.zeros((X_train.shape[1], t_train.shape[1]))

    else:
        W = initial_weights

    for _ in range(n_iter):
        grad_w = compute_grad(X_train, t_train, W)
        W = W - alpha * grad_w
    return W
