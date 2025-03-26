import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from linear_regression import LinearRegression


# Define a function to calculate Mean Squared Error (MSE)
def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Generate synthetic regression dataset with noise
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Initialize the Linear Regression model with learning rate and number of iterations
regressor = LinearRegression(lr=0.01, n_iters=1000)

# Train the model on the training data
regressor.fit(X_train, y_train)

# Generate predictions on the test data
predictions = regressor.predict(X_test)

# Print the Mean Squared Error for the test set
print(MSE(y_test, predictions))

# Create a figure for visualizing the data and regression line
fig = plt.figure(figsize=(8, 6))

# Scatter plot of the original data
plt.scatter(X[:, 0], y, color='blue', marker='o', s=30)

# Extract model parameters (weights and bias) from the LinearRegression class
weight = regressor.weights[0]  # Assuming weights is a 1D array
bias = regressor.bias  # Assuming bias is a scalar

# Generate a range of X values to plot the regression line
x_min, x_max = X.min(), X.max()
x_range = np.linspace(x_min, x_max, 100)  # 100 evenly spaced points between min and max

# Calculate corresponding y values for the regression line
line_predictions = weight * x_range + bias

# Plot the regression line across the full data range
plt.plot(x_range, line_predictions, color='red', label="Regression Line")

# Add legend and display the plot
plt.legend()
plt.show()