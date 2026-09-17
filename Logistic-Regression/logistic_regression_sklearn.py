import numpy as np
from sklearn.linear_model import LogisticRegression


# Training Data
# X = Hours Studied
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# y = Actual result
# 0 = Fail
# 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1])


# Create the Logistic Regression model
model = LogisticRegression()


# Train the model
# Sklearn handles the loss calculation
# and optimization internally.
model.fit(X, y)


# Predict the class for a new student
# Student studied for 5 hours.
new_student = np.array([[5]])

prediction = model.predict(new_student)

print("Prediction:", prediction[0])


# Get the probability of each class
probability = model.predict_proba(new_student)

print("Probability of Fail:", probability[0][0])
print("Probability of Pass:", probability[0][1])
