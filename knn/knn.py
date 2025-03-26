import numpy as np
from collections import Counter


# Function to calculate the Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Implementation of the K-Nearest Neighbors (KNN) classifier
class KNN:
    def __init__(self, k=3):
        """
        Initializes the KNN classifier.

        Parameters:
        k (int): The number of nearest neighbors to consider for classification.
        """
        self.k = k

    def fit(self, X, y):
        """
        Stores the training data.

        Parameters:
        X (numpy.ndarray): Training data features.
        y (numpy.ndarray): Target labels corresponding to the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts the class labels for the given data points.

        Parameters:
        X (numpy.ndarray): Data points to be classified.

        Returns:
        numpy.ndarray: Predicted class labels.
        """
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        """
        Predicts the class label for a single data point.

        Parameters:
        x (numpy.ndarray): A single data point to classify.

        Returns:
        int: The predicted class label.
        """
        # Compute distances from the input point to all points in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Find the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Retrieve the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Perform a majority vote among the k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)

        # Return the label with the highest frequency
        return most_common[0][0]