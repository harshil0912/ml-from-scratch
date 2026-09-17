import numpy as np

"""
Logistic Regression From Scratch

Logistic Regression is used for binary classification.

The model follows this process:

    z = wx + b
        ↓
    Sigmoid
        ↓
    Probability
        ↓
    Cross-Entropy Loss
        ↓
    Gradient Descent
        ↓
    Update weight and bias

The goal is to learn the weight (w) and bias (b)
that produce good class probabilities.
"""


# --------------------------------------------------
# 1. Training Data
# --------------------------------------------------

# X = input feature
# Here, X represents Hours Studied.
X = np.array([1, 2, 3, 4, 5, 6], dtype=float)

# y = actual target
# 0 = Fail
# 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1], dtype=float)


# --------------------------------------------------
# 2. Model Parameters
# --------------------------------------------------

# Start with initial values.
# These values will be updated during training.
weight = 0.0
bias = 0.0


# --------------------------------------------------
# 3. Hyperparameters
# --------------------------------------------------

# Learning rate controls how large each update is.
learning_rate = 0.1

# Number of times the model goes through
# the training process.
epochs = 1000


# --------------------------------------------------
# 4. Sigmoid Function
# --------------------------------------------------

def sigmoid(z):
    """
    Converts the linear output into a probability
    between 0 and 1.

        sigmoid(z) = 1 / (1 + e^(-z))
    """
    return 1 / (1 + np.exp(-z))


# --------------------------------------------------
# 5. Training the Model
# --------------------------------------------------

for epoch in range(epochs):

    # Step 1: Calculate the linear output
    #
    # z = wx + b
    z = weight * X + bias

    # Step 2: Apply the Sigmoid function
    #
    # Converts z into probabilities.
    #
    # Example:
    # 0.90 → 90% probability of Class 1
    y_pred = sigmoid(z)


    # --------------------------------------------------
    # Step 3: Calculate Cross-Entropy Loss
    # --------------------------------------------------

    # Cross-Entropy measures how wrong
    # the predicted probabilities are.
    #
    # Correct and confident prediction → small loss
    # Wrong and confident prediction    → large loss
    #
    # L = -[y log(y_pred) +
    #       (1-y) log(1-y_pred)]

    # Clip probabilities to prevent log(0),
    # which would produce -infinity.
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

    loss = -(1 / len(X)) * np.sum(
        y * np.log(y_pred_clipped)
        + (1 - y) * np.log(1 - y_pred_clipped)
    )


    # --------------------------------------------------
    # Step 4: Calculate the Gradient
    # --------------------------------------------------

    # The gradient tells us how the loss changes
    # when we change the weight and bias.
    #
    # For Sigmoid + Cross-Entropy, the gradient
    # simplifies to:
    #
    # prediction - actual

    error = y_pred - y

    # Gradient of loss with respect to weight.
    #
    # X is multiplied here because changing the
    # weight affects the prediction according to X.
    dw = (1 / len(X)) * np.sum(X * error)

    # Gradient of loss with respect to bias.
    #
    # Bias shifts every prediction equally,
    # so X is not multiplied here.
    db = (1 / len(X)) * np.sum(error)


    # --------------------------------------------------
    # Step 5: Gradient Descent
    # --------------------------------------------------

    # Move the parameters in the direction
    # that reduces the loss.
    #
    # New parameter = Old parameter - Learning Rate × Gradient

    weight = weight - learning_rate * dw
    bias = bias - learning_rate * db


# --------------------------------------------------
# 6. Final Probabilities
# --------------------------------------------------

# Use the learned weight and bias
# to calculate the final probabilities.
z = weight * X + bias

probabilities = sigmoid(z)


# --------------------------------------------------
# 7. Convert Probabilities into Classes
# --------------------------------------------------

# We use 0.5 as the decision threshold.
#
# Probability >= 0.5 → Class 1 (Pass)
# Probability <  0.5 → Class 0 (Fail)

predictions = (probabilities >= 0.5).astype(int)


# --------------------------------------------------
# 8. Display Results
# --------------------------------------------------

print("Weight:", weight)
print("Bias:", bias)

print("\nProbabilities:")
print(probabilities)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y)

print("\nFinal Cross-Entropy Loss:", loss)
