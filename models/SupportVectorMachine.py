import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_=0.01, iters=1000, weights=None, bias=None):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.iters = iters
        self.weights = weights
        self.bias = bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)
        #initialize weights & biases
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iters):
            for i, Xi in enumerate(X):
                if y[i] * (np.dot(Xi, self.weights) - self.bias) >= 1:
                    self.weights -= self.learning_rate * (2 * self.lambda_ * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_ * self.weights - np.dot(Xi, y[i]))
                    self.bias -= self.learning_rate * y[i]
        return self.weights, self.bias

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) - self.bias
        if isinstance(y_predicted, np.ndarray):
            y_predicted_cls = [1 if val > 0 else 0 for val in y_predicted]
            return y_predicted_cls
        else:
            y_predicted_cls = 1 if y_predicted > 0 else 0
            return y_predicted_cls

    def predict_proba(self, X):
        pred = np.dot(X, self.weights) - self.bias
        #normalize from 0 to 1
        pred_norm = []
        for val in pred:
            val = (val - np.min(pred))/(np.max(pred) - np.min(pred))
            pred_norm.append(val)
        pred_norm = np.array(pred_norm)
        return pred_norm
