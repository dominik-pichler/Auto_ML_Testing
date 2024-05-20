import numpy as np
from collections import Counter
from scipy.spatial import distance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class custom_kNN(BaseEstimator, TransformerMixin):
    def __init__(self, k=3, distance_function='manhattan'):
        self.y_train = None
        self.x_train = None
        self.k = k
        self.distance_function = distance_function

    def fit(self, X_train, y_train):
        self.x_train = X_train
        self.y_train = y_train
        return self

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def calculate_distance(self, x1: list, x2: list, type: str) -> float:
        """
        This function determines the distance between two distance vectors for a KNN Implementation

        :param x1: distance vector
        :param x2: distance vector
        :param type: type of distance metric that should be used
        :return: float distance between the two vectors
        """
        if type == 'euclidean':
            return distance.euclidean(x1, x2)
        elif type == 'hamming':
            return distance.hamming(x1, x2)
        elif type == 'manhattan':
            return distance.cityblock(x1, x2)
        elif type == 'minkowski':
            return distance.minkowski(x1, x2)
        elif type == 'chebyshev':
            return distance.chebyshev(x1, x2)
        elif type == 'cosine':
            return distance.cosine(x1, x2)
        else:
            raise ValueError("ERROR: Invalid Distance Metric specified! ")

    def predict(self, X_test):
        y_pred = [self._predict(x.values) for _, x in tqdm(X_test.iterrows(), total=X_test.shape[0], position=0, leave=True)]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.calculate_distance(x, x_train, self.distance_function) for _, x_train in self.x_train.iterrows()]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the target values of the k nearest neighbor training samples
        k_nearest_targets = [self.y_train.iloc[i] for i in k_indices]
        # Return the mean of the target values of the k nearest neighbors
        return np.mean(k_nearest_targets)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
