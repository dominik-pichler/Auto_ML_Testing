class custom_kNN:
    def __init__(self, param1=1, param2=2):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y):
        # Custom training logic
        print("Fitting the model with data")
        return self

    def predict(self, X):
        # Custom prediction logic
        print("Predicting with the model")
        return [0] * len(X)

    def score(self, X, y):
        # Custom scoring logic
        print("Scoring the model")
        return 0.5  # Dummy score
