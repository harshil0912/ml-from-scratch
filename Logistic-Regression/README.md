# Logistic Regression

Logistic Regression is a **supervised learning algorithm used for classification**.

Unlike Linear Regression, which predicts a continuous numerical value, Logistic Regression predicts the **probability of an input belonging to a class**.

For example, we might want to predict whether a student will pass based on the number of hours they studied.

| Hours Studied | Result |
|---------------|--------|
| 1             | Fail   |
| 2             | Fail   |
| 3             | Fail   |
| 4             | Pass   |
| 5             | Pass   |
| 6             | Pass   |

Here:

```text
X = Hours Studied
y = Pass / Fail
```

Because there are only two possible classes:

```text
0 → Fail
1 → Pass
```

This is called **Binary Classification**.

---

# What Are We Actually Trying to Do?

Imagine we have training data showing how many hours students studied and whether they passed.

We want our model to learn a pattern like:

```text
Less studying
      ↓
Lower probability of passing

More studying
      ↓
Higher probability of passing
```

But instead of directly predicting:

```text
Pass
Fail
```

the model first predicts a **probability**.

For example:

```text
1 hour  → 0.02
2 hours → 0.08
3 hours → 0.25
4 hours → 0.68
5 hours → 0.91
6 hours → 0.98
```

These values are useful because they tell us **how confident the model is**.

---

# Why Can't We Just Use Linear Regression?

Linear Regression produces a straight line:

$$
\hat{y} = wx + b
$$

The problem is that a straight line can produce **any numerical value**.

For example:

```text
-2
-0.5
0.4
1.2
3
```

But probability should be between:

```text
0 and 1
```

We don't want predictions such as:

```text
-0.4 → -40% probability
1.7  → 170% probability
```

So we need something that takes the linear output and **squashes it between 0 and 1**.

That function is the **Sigmoid Function**.

---

# The Sigmoid Function

The Sigmoid Function is:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

It takes any value from negative infinity to positive infinity and converts it into a value between `0` and `1`.

Some examples:

| $z$ | $\sigma(z)$ |
|-----|-------------|
| -5  | ≈ 0.007     |
| -2  | ≈ 0.119     |
| 0   | 0.500       |
| 2   | ≈ 0.881     |
| 5   | ≈ 0.993     |

The important idea is:

```text
Very negative z → probability close to 0

z = 0 → probability = 0.5

Very positive z → probability close to 1
```

The shape of the Sigmoid Function looks like an **S-curve**.

It smoothly transforms our unlimited linear output into a probability.

---

# Visualizing the Sigmoid

The horizontal axis represents `z`.

The vertical axis represents the probability.

As `z` increases:



<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/0bd8ba53-6f4c-438e-b1dc-c0cf11968c92" />


Notice something important:

The sigmoid never needs to manually say:

```python
if z > something:
    probability = 1
else:
    probability = 0
```

Instead, it gives us a **smooth probability**.

---

# Where Does z Come From?

Before applying the Sigmoid Function, Logistic Regression calculates:

$$
z = wx + b
$$

This is exactly the same linear equation we saw in Linear Regression.

Then:

$$
\hat{y} = \sigma(z)
$$

So the complete mathematical process is:

$$
z = wx + b
$$

$$
\hat{y} = \frac{1}{1 + e^{-z}}
$$

In simple terms:

```text
Input
  ↓
Linear Equation
  ↓
z = wx + b
  ↓
Sigmoid
  ↓
Probability
```

---

# Probability Does Not Mean Class Yet

Suppose the model predicts:

$$
\hat{y} = 0.82
$$

This means:

```text
82% probability of Class 1
```

But we still need to convert that probability into an actual class.

We commonly use a threshold of `0.5`.

```text
Probability >= 0.5 → Class 1

Probability < 0.5 → Class 0
```

For example:

```text
0.82 → Class 1
0.31 → Class 0
0.51 → Class 1
0.12 → Class 0
```

So:

```text
Probability
     ↓
  Threshold
     ↓
   Class
```

---

# Now Comes the Important Question

So far, we know how Logistic Regression makes a prediction.

But how does the model know whether its prediction is **good or bad**?

Let's use a real example.

Suppose we have a student who studied for **5 hours**.

The actual result is:

```text
Pass
```

Since we represent:

```text
0 → Fail
1 → Pass
```

the actual value is:

$$
y = 1
$$

The model does not directly say:

```text
Pass
```

Instead, it produces a **probability of Pass**.

For example, the model might predict:

$$
\hat{y} = 0.90
$$

This means:

```text
90% probability of Pass
```

Since the actual result was Pass (`y = 1`), this is a pretty good prediction.

But how do we mathematically measure **how good** it is?

That's the job of the **Loss Function**.

---

# Binary Cross-Entropy / Log Loss

For binary classification, Logistic Regression commonly uses:

**Binary Cross-Entropy Loss**

also called:

**Log Loss**

For one training example:

$$
L = -\left[y\log(\hat{y}) + (1-y)\log(1-\hat{y})\right]
$$

Don't worry about memorizing the formula yet.

Let's actually use it.

---

# Let's See What Cross-Entropy Does

We have the same student:

```text
Hours Studied = 5

Actual Result = Pass

y = 1
```

Now imagine the model makes a prediction:

$$
\hat{y} = 0.90
$$

The model is saying:

```text
90% probability of Pass
```

Because the student actually **passed**, this is a good prediction.

Since:

$$
y = 1
$$

the formula becomes:

$$
L = -\log(\hat{y})
$$

Substituting:

$$
L = -\log(0.90)
$$

Therefore:

$$
L \approx 0.105
$$

That's a **small loss**.

---

# Now Let's Make the Model Less Confident

Imagine the same student, with the same actual result:

```text
Actual Result = Pass

y = 1
```

But this time the model predicts:

$$
\hat{y} = 0.50
$$

The model is basically saying:

```text
50% probability of Pass
```

It is unsure.

The loss becomes:

$$
L = -\log(0.50)
$$

Therefore:

$$
L \approx 0.693
$$

Notice what happened:

```text
Prediction = 0.90
Loss       ≈ 0.105

Prediction = 0.50
Loss       ≈ 0.693
```

The prediction became less confident about the correct answer, so the loss increased.

---

# Now Let's Make It Confidently Wrong

Same student.

Same actual result:

```text
Actual Result = Pass

y = 1
```

But now the model predicts:

$$
\hat{y} = 0.10
$$

That means:

```text
10% probability of Pass
```

The model is strongly leaning toward **Fail**, even though the student actually passed.

Calculate the loss:

$$
L = -\log(0.10)
$$

Therefore:

$$
L \approx 2.303
$$

Now look at the progression:

| Model Prediction | Meaning | Loss |
|------------------|---------|------|
| `0.90`           | 90% chance of Pass | `0.105` |
| `0.50`           | 50% chance of Pass | `0.693` |
| `0.10`           | 10% chance of Pass | `2.303` |

This is the important behavior of Cross-Entropy:

```text
Prediction moves toward the correct answer
              ↓
           Loss decreases
```

```text
Prediction moves toward the wrong answer
              ↓
           Loss increases
```

And notice something even more important:

```text
Very confident + Correct
        ↓
    Very small loss
```

while:

```text
Very confident + Wrong
        ↓
    Very large loss
```

---

# Why Is This Different From Just Checking Correct / Wrong?

Suppose the actual result is:

```text
Pass

y = 1
```

Two models make these predictions:

```text
Model A → 0.51
Model B → 0.99
```

After applying the `0.5` threshold, both predict:

```text
Class 1 → Pass
```

So if we only checked:

```text
Correct or Wrong?
```

both models would look equally good.

But they are not equally confident.

```text
Model A → 51% probability of Pass
Model B → 99% probability of Pass
```

Cross-Entropy captures this difference.

The model that assigns a **higher probability to the correct class** receives a **smaller loss**.

This gives Logistic Regression something much more useful than just:

```text
Correct / Wrong
```

It gives us a continuous measure of **how good the probability prediction is**.

---

# What If the Actual Class Is 0?

The exact same idea works when the actual class is `0`.

Suppose another student:

```text
Actual Result = Fail

y = 0
```

Now imagine the model predicts:

$$
\hat{y} = 0.10
$$

Remember:

```text
ŷ = probability of Class 1
```

So:

```text
0.10 → 10% probability of Pass
```

That means the model is giving:

```text
90% probability of Fail
```

This is a good prediction because the actual result was Fail.

The loss is:

$$
L = -\log(1-\hat{y})
$$

Substituting:

$$
L = -\log(1-0.10)
$$

$$
L = -\log(0.90)
$$

Therefore:

$$
L \approx 0.105
$$

Again, we get a small loss.

But if the model predicts:

$$
\hat{y} = 0.90
$$

it is saying:

```text
90% probability of Pass
```

even though the student actually failed.

That produces a large loss:

$$
L = -\log(1-0.90)
$$

$$
L = -\log(0.10)
$$

Therefore:

$$
L \approx 2.303
$$

So the same principle applies:

```text
Actual = 0

Prediction close to 0 → Small loss
Prediction close to 1 → Large loss
```

---

# The Intuition Behind Cross-Entropy

Now we can finally understand what Cross-Entropy is doing.

The model gives us a probability:

```text
        Prediction
             ↓
       How confident?
             ↓
       Is that confidence
        in the right class?
             ↓
            Loss
```

For the correct class:

```text
High probability
      ↓
Small loss
```

For the wrong class:

```text
High probability
      ↓
Large loss
```

So Cross-Entropy encourages the model to do two things:

1. Predict the **correct class**
2. Give the correct class a **high probability**

---

# The Big Picture

Now we can connect everything we've learned so far.

The model starts with the input:

```text
Hours Studied
      ↓
Linear Equation
      ↓
z = wx + b
      ↓
Sigmoid
      ↓
Probability
      ↓
Cross-Entropy
      ↓
Loss
```

The model then uses the loss to learn better values for its parameters.

```text
Loss
 ↓
Gradient Descent
 ↓
Update weight and bias
 ↓
Repeat
```

So the complete Logistic Regression learning process is:

```text
Input Features
      ↓
Linear Equation
      ↓
z = wx + b
      ↓
Sigmoid
      ↓
Probability
      ↓
Cross-Entropy Loss
      ↓
Gradients
      ↓
Gradient Descent
      ↓
Update Weight & Bias
      ↓
Repeat
```

Now let's see this process in code in the next Python file.
