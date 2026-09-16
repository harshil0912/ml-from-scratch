import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


"""
Linear Regression using Scikit-Learn

Goal:
Predict house price based on house area.

Steps:
1. Prepare the dataset
2. Split data into training and testing sets
3. Create the model
4. Train the model
5. Make predictions
6. Evaluate the model
7. Visualize the regression line
"""


# Dataset
# X = input feature (Area)
# y = target variable (Price)
X = np.array([
    800, 1000, 1200, 1400, 1600, 1800,
    2000, 2200, 2400, 2600, 2800, 3000
]).reshape(-1, 1)

y = np.array([
    38, 52, 45, 63, 55, 74,
    68, 86, 78, 96, 88, 108
])


# Split the data into training and testing sets
# 80% → training
# 20% → testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create the Linear Regression model
model = LinearRegression()


# Train the model
# Scikit-Learn learns the weight and bias from the training data
model.fit(X_train, y_train)


# Make predictions on the test data
y_pred = model.predict(X_test)


# Get the learned parameters
# coef_ = weight / slope
# intercept_ = bias / intercept
weight = model.coef_[0]
bias = model.intercept_

print("Weight:", weight)
print("Bias:", bias)


# Evaluate the model
# MSE measures the average squared prediction error
mse = mean_squared_error(y_test, y_pred)

# R² measures how much variation in the target
# is explained by the model
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)


# Predict the price of a new house
area = np.array([[2500]])

predicted_price = model.predict(area)

print("Predicted Price for 2500 sq ft:",
      predicted_price[0], "Lakhs")


# Plot the original data points
plt.scatter(X, y, label="Actual Data")


# Generate points for drawing the regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# Predict prices for the points on the line
y_line = model.predict(X_line)

# Plot the regression line
plt.plot(X_line, y_line, label="Regression Line")


# Add labels and title
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (Lakhs)")
plt.title("Linear Regression: House Price Prediction")
plt.legend()

plt.show()
