import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

class custom_GD(BaseEstimator, TransformerMixin):
    def __init__(self, learning_rate=0.01, tolerance=1e-6, max_iterations=1000):  # hyperparameters
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.uniform(-1, 1, n_features)  # weights randomly set between -1 and 1
        # self.weights = np.zeros(n_features)

        for iteration in range(self.max_iterations):
            predictions = X.dot(self.weights)  # calculate predictions
            errors = predictions - y.iloc[:, 0]   # compute error
            gradient_w = 2 * X.T.dot(errors) / n_samples  # gradient for weights

            # Update weights and bias
            self.weights -=  self.learning_rate * gradient_w

            # check for convergence
            if np.linalg.norm(gradient_w) < self.tolerance:
                print(f"Convergence reached after {iteration + 1} iterations.")
                return self
        return self

    def predict(self, X):
        return X.dot(self.weights)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

