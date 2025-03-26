import numpy as np


class LinearRegression:
    """
    A simple implementation of Linear Regression using Gradient Descent.
    """

    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initialize model parameters and hyperparameters.
        """
        self.lr = lr  # Learning rate for gradient descent
        self.n_iters = n_iters  # Maximum number of iterations
        self.weights = None  # Weights (coefficients), initialized during training
        self.bias = None  # Bias term, initialized during training

    def fit(self, X, y):
        """
        Train the model using gradient descent.

        Parameters:
        X (numpy.ndarray): Training data of shape (n_samples, n_features).
        y (numpy.ndarray): True labels/targets of shape (n_samples,).
        """
        # Extract number of samples and features
        n_samples, n_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Perform gradient descent optimization
        for _ in range(self.n_iters):
            # Linear prediction: y_pred = X * weights + bias
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients for weights and bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update model parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict target values for input features.

        Parameters:
        X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
        numpy.ndarray: Predicted values of shape (n_samples,).
        """
        # Use trained weights and bias to compute predictions
        return np.dot(X, self.weights) + self.bias