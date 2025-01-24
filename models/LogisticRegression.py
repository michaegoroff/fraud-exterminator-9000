import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.001, iters=1000, weights=None, bias=None):
        self.lr = learning_rate
        self.iters = iters
        self.weights = weights
        self.bias = bias
        self.losses = []

    # Activation: sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y_true, y_pred):
        e = 1e-9
        y1 = y_true * np.log(y_pred + e)
        y2 = (1 - y_true) * np.log(1 - y_pred + e)
        return -np.mean(y1 + y2)

    def feed_forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        A = self.sigmoid(z)
        return A

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init weights & biases
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.iters):
            A = self.feed_forward(X)
            self.losses.append(self.compute_loss(y, A))
            dz = A - y
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)
            # update weights & biases
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        threshold = 0.5
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(y_hat)
        if isinstance(y_predicted, np.ndarray):
            y_predicted_cls = [1 if i > threshold else 0 for i in y_predicted]
            return np.array(y_predicted_cls)
        else:
            y_predicted_cls = 1 if y_predicted > threshold else 0
            return y_predicted_cls


    def predict_proba(self, X):
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(y_hat)
        return np.array(y_predicted)
