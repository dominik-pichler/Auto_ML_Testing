from numpy import log
def log_transform(X):
    return log(X + 1e-10)  # Add a small constant to prevent taking log of zero or negative values
