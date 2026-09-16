
import numpy as np
"""
Linear Regression From Scratch

The model learns a straight-line relationship:

    ŷ = wx + b

where:
    w = weight / slope
    b = bias / intercept

The model learns w and b by using Gradient Descent
to minimize the Mean Squared Error (MSE).
"""


# --------------------------------------------------
# 1. Training Data
# --------------------------------------------------

# X = input feature
# In our example, X represents the area of a house.
X = np.array([1, 2, 3, 4, 5], dtype=float)

# y = actual target value
# In this example, these are the actual values we want to predict.
y = np.array([2, 4, 6, 8, 10], dtype=float)


# --------------------------------------------------
# 2. Model Parameters
# --------------------------------------------------

# We don't know the correct weight and bias initially.
# So we start with 0 and let Gradient Descent learn them.
weight = 0.0
bias = 0.0


# --------------------------------------------------
# 3. Hyperparameters
# --------------------------------------------------

# Learning rate controls how big each update should be.
# A smaller value means smaller steps.
# A larger value means larger steps.
learning_rate = 0.01

# Number of times the model will repeat the learning process.
epochs = 1000


# --------------------------------------------------
# 4. Training the Model
# --------------------------------------------------

for epoch in range(epochs):

    # ----------------------------------------------
    # Step 1: Make Predictions
    # ----------------------------------------------

    # Our Linear Regression equation is:
    #
    #     ŷ = wx + b
    #
    # Using the current weight and bias, we calculate
    # a prediction for every value in X.
    y_pred = weight * X + bias


    # ----------------------------------------------
    # Step 2: Calculate Error
    # ----------------------------------------------

    # Error = Prediction - Actual Value
    #
    # This tells us how far each prediction is
    # from the actual target.
    error = y_pred - y


    # ----------------------------------------------
    # Step 3: Calculate Gradients
    # ----------------------------------------------

    # The gradients tell us how the MSE changes
    # when we change the weight and bias.
    #
    # Remeber the mse formula in the notes 
  
    # dw = derivative of MSE with respect to weight
    # db = derivative of MSE with respect to bias
    #
    # These gradients tell Gradient Descent
    # which direction the parameters should move.

    # The dw means How much does the MSE change when we change the weight w
    dw = (2 / len(X)) * np.sum(X * error) # When u derivate it u will this formula 
    



    db = (2 / len(X)) * np.sum(error)
    # Why doesn't db have X?
    # Because changing the bias moves every prediction by the same amount.
   

    # ----------------------------------------------
    # Step 4: Update Parameters
    # ----------------------------------------------

    # Gradient Descent update rule:
    #
    #     new parameter = old parameter
    #                     - learning rate × gradient
    #
    # We subtract the gradient because we want
    # to move in the direction that reduces the error.

    weight = weight - learning_rate * dw
    bias = bias - learning_rate * db


# --------------------------------------------------
# 5. Final Predictions
# --------------------------------------------------

# After training, weight and bias have been learned.
# We use the learned values to make the final predictions.
predictions = weight * X + bias


# --------------------------------------------------
# 6. Display Results
# --------------------------------------------------

print("Weight:", weight)
print("Bias:", bias)
print("Predictions:", predictions)
