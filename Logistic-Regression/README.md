# Logistic Regression

Logistic Regression is a **supervised learning algorithm** used for **classification**.

Unlike Linear Regression, which predicts a continuous numerical value, Logistic Regression predicts the **probability of an input belonging to a class**.

---

## 📌 Example

Suppose we want to predict whether a student will pass based on the number of hours they studied:

| Hours Studied | Result |
| ------------: | ------ |
|             1 | Fail   |
|             2 | Fail   |
|             3 | Fail   |
|             4 | Pass   |
|             5 | Pass   |
|             6 | Pass   |

Here:

* **X** → Hours Studied
* **y** → Pass / Fail

Since there are two possible classes:

```text
0 → Fail
1 → Pass
```

This is called **Binary Classification**.

---

# 1. The Problem with Linear Regression

We might initially think that Linear Regression can be used for classification.

Linear Regression produces a continuous output such as:

```text
0.2
0.7
1.4
-0.3
2.1
```

However, for classification we want a **probability between 0 and 1**.

For example:

```text
0.20 → 20% probability of Pass
0.70 → 70% probability of Pass
0.95 → 95% probability of Pass
```

So we need a function that converts the output of a linear equation into a value between **0 and 1**.

This is where the **Sigmoid Function** comes in.

---

# 2. The Linear Part

Logistic Regression first calculates the same linear equation used in Linear Regression:

$$
z = wx + b
$$

Where:

| Symbol | Meaning       |
| ------ | ------------- |
| `x`    | Input feature |
| `w`    | Weight        |
| `b`    | Bias          |
| `z`    | Linear output |

The **weight** controls how strongly the input affects the prediction.

The **bias** shifts the output independently of the input.

However, `z` itself is **not the final prediction**.

---

# 3. Sigmoid Function

The linear output `z` is passed through the **Sigmoid Function**:

$$
\hat{y} = \frac{1}{1 + e^{-z}}
$$

The Sigmoid Function converts any value of `z` into a value between **0 and 1**.

### Example

```text
z = -5 → probability close to 0
z =  0 → probability = 0.5
z =  5 → probability close to 1
```

The output can be interpreted as the **probability of belonging to Class 1**.

---

# 4. From Probability to Class

The Sigmoid Function gives us a **probability**, not the final class.

For binary classification, we commonly use a threshold of **0.5**.

```text
Probability >= 0.5 → Class 1
Probability <  0.5 → Class 0
```

### Example

```text
0.82 → Class 1 → Pass
0.31 → Class 0 → Fail
```

Therefore, the complete prediction process is:

```text
Input
  ↓
Linear Equation
  ↓
z = wx + b
  ↓
Sigmoid Function
  ↓
Probability
  ↓
Threshold
  ↓
Final Class
```

---

# 5. How Does Logistic Regression Learn?

Initially, the model does not know the correct values of the weight and bias.

We can start with:

```text
w = 0
b = 0
```

The model then repeatedly performs the following steps:

1. Calculate `z`
2. Calculate the probability using Sigmoid
3. Compare the prediction with the actual value
4. Calculate the loss
5. Calculate the gradients
6. Update the weight and bias
7. Repeat the process

This learning process is performed using **Gradient Descent**.

---

# 6. Loss Function

For Logistic Regression, we use **Binary Cross-Entropy Loss**, also called **Log Loss**.

The loss measures how different the predicted probability is from the actual class.

### When the actual class is `1`

```text
Prediction close to 1 → Small loss
Prediction close to 0 → Large loss
```

### When the actual class is `0`

```text
Prediction close to 0 → Small loss
Prediction close to 1 → Large loss
```

The model uses this loss to determine how the **weight and bias should be updated**.

The Binary Cross-Entropy Loss for one example is:

$$
L = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]
$$

For `n` training examples:

$$
J = -\frac{1}{n}\sum_{i=1}^{n}
[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
$$

---

# 7. Gradient Descent

Gradient Descent is used to find better values for the **weight and bias**.

For Logistic Regression, the gradients are:

$$
dw = \frac{1}{n}\sum X(\hat{y}-y)
$$

$$
db = \frac{1}{n}\sum(\hat{y}-y)
$$

The parameters are then updated using:

$$
w = w - \text{learning rate} \times dw
$$

$$
b = b - \text{learning rate} \times db
$$

This process is repeated for multiple **epochs** until the model learns suitable values for `w` and `b`.

---

# 8. Linear Regression vs Logistic Regression

The major difference is what happens **after the linear equation**.

### Linear Regression

```text
Input
  ↓
wx + b
  ↓
Prediction
```

### Logistic Regression

```text
Input
  ↓
wx + b
  ↓
Sigmoid
  ↓
Probability
  ↓
Threshold
  ↓
Class
```

| Feature        | Linear Regression  | Logistic Regression  |
| -------------- | ------------------ | -------------------- |
| Task           | Regression         | Classification       |
| Output         | Continuous value   | Probability          |
| Activation     | None               | Sigmoid              |
| Typical output | Any real number    | 0 to 1               |
| Example        | House price        | Pass / Fail          |
| Loss           | Mean Squared Error | Binary Cross-Entropy |

---

# 9. Key Terms

| Term                 | Meaning                                         |
| -------------------- | ----------------------------------------------- |
| **X**                | Input feature                                   |
| **y**                | Actual class                                    |
| **w**                | Learned weight                                  |
| **b**                | Learned bias                                    |
| **z**                | Linear output before Sigmoid                    |
| **ŷ**                | Predicted probability                           |
| **Sigmoid**          | Converts `z` into a value between 0 and 1       |
| **Threshold**        | Converts probability into a class               |
| **Log Loss**         | Measures how wrong the predicted probability is |
| **Gradient Descent** | Learns the weight and bias                      |
| **Epoch**            | One complete pass through the training data     |
| **Learning Rate**    | Controls the size of each parameter update      |

---

# 10. Simple Example

Suppose:

```text
X = Hours Studied
y = Pass / Fail
```

The model first calculates:

$$
z = wx + b
$$

Suppose the model gets:

$$
z = 2
$$

Apply the Sigmoid Function:

$$
\hat{y} = \frac{1}{1+e^{-2}}
$$

Therefore:

$$
\hat{y} \approx 0.88
$$

This means:

```text
88% probability of Pass
```

Now apply the threshold:

```text
0.88 >= 0.5
```

Therefore:

```text
Class 1 → Pass
```

---

# 11. Complete Learning Process

The entire training process can be summarized as:

```text
             Training Data
                  ↓
            Initialize w, b
                  ↓
            Calculate z
              z = wx + b
                  ↓
             Apply Sigmoid
                  ↓
        Calculate Probability
                  ↓
           Calculate Log Loss
                  ↓
          Calculate Gradients
                  ↓
        Update w and b
                  ↓
              Repeat
                  ↓
           Trained Model
```

---

# 12. The Complete Idea

Logistic Regression can be understood as:

```text
Linear Equation
      +
Sigmoid Function
      +
Binary Cross-Entropy Loss
      +
Gradient Descent
      ↓
Binary Classification Model
```

The core mathematical pipeline is:

$$
z = wx+b
$$

$$
\hat{y} = \frac{1}{1+e^{-z}}
$$

```text
Probability
     ↓
Threshold
     ↓
Class
```

---

# Implementation

This repository contains a **Logistic Regression implementation from scratch using NumPy**, including:

* Linear calculation
* Sigmoid activation
* Binary Cross-Entropy / Log Loss
* Gradient calculation
* Gradient Descent
* Parameter updates
* Binary classification

No machine-learning library is used to implement the core algorithm.

The goal is to understand **what is happening inside Logistic Regression rather than simply calling `LogisticRegression()` from scikit-learn**.
