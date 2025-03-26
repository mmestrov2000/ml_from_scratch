import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from knn import KNN  # Importing custom KNN implementation from knn.py


# Function to calculate the accuracy of predictions
def accuracy(y_true, y_pred):
    """
    Computes the accuracy of predicted labels.

    Parameters:
    y_true (numpy.ndarray): Array of true class labels.
    y_pred (numpy.ndarray): Array of predicted class labels.

    Returns:
    float: Fraction of correctly classified samples (accuracy score).
    """
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target  # Features (X) and target labels (y)

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Set the number of neighbors for KNN
k = 3

# Create an instance of the KNN classifier with the given value of k
clf = KNN(k=k)

# Train the KNN classifier on the training data
clf.fit(X_train, y_train)

# Predict the class labels for the test data
predictions = clf.predict(X_test)

# Calculate and print the classification accuracy
print("KNN classification accuracy:", accuracy(y_test, predictions))