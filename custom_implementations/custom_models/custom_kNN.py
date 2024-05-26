import numpy as np
from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class custom_kNN(BaseEstimator, TransformerMixin):
    def __init__(self, k=1, distance_function='manhattan'):
        self.y_train = None
        self.x_train = None
        self.k = k
        self.distance_function = distance_function
        self.tree = None

    def fit(self, X_train, y_train):
        self.x_train = X_train.values
        self.y_train = y_train.values
        self.tree = BallTree(self.x_train, leaf_size=30, metric=self.distance_function)
        return self

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def predict(self, X_test):
        X_test = X_test.values
        y_pred = np.zeros(X_test.shape[0])
        for i, x in tqdm(enumerate(X_test), total=X_test.shape[0], position=0, leave=True):
            y_pred[i] = self._predict(x)
        return y_pred

    def _predict(self, x):
        _, indices = self.tree.query([x], k=self.k)
        k_nearest_targets = self.y_train[indices[0]]
        return np.mean(k_nearest_targets)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
