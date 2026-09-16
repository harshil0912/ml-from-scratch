# Linear Regression

Linear Regression comes under **Supervised Learning**.

Supervised learning is a type of machine learning where a model learns from **labeled data**, meaning each input has a known correct output (target).
Regression is a type of supervised learning used to predict a continuous numerical value from one or more input features.

## Example of Labeled Data

| Area (X) | Actual Price (y) |
|---|---|
| 1000 sq ft | ₹50L |
| 1500 sq ft | ₹70L |
| 2000 sq ft | ₹90L |
| 3000 sq ft | ₹100L |

---

## Let's Understand Linear Regression Practically

Let's say we have a dataset like this:

| Area (X) — sq ft | Actual Price (y) — ₹ Lakhs |
|---:|---:|
| 800 | 38 |
| 1000 | 52 |
| 1200 | 45 |
| 1400 | 63 |
| 1600 | 55 |
| 1800 | 74 |
| 2000 | 68 |
| 2200 | 86 |
| 2400 | 78 |
| 2600 | 96 |
| 2800 | 88 |
| 3000 | 108 |

Now, how would we predict the actual price for a particular area given by a user?

Let's try to tackle this using **Linear Regression**.

First, let's plot the data.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/3d1cbb87-4a73-455d-baaf-39a72b7beb08" />

---

## Finding a Line

Now let's try to understand it professionally, but in a simple way.

We are looking for a line that **fits the data as closely as possible**.

The line should generally pass between the points and have the **smallest possible error** when compared with the actual data.

To measure how good a line is, we use **Mean Squared Error (MSE)**.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/7639fb87-11a5-49c2-86ed-6feaa8325fa4" />

---

## Mean Squared Error (MSE)

MSE is simply the **average of the squared differences between the actual value and the model's predicted value**.

In simple words:

> We calculate how far each prediction is from the actual value, square those errors, and then take their average.

The formula is:

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Where:

- `n` = number of data points
- `yᵢ` = actual value
- `ŷᵢ` = predicted value
- `yᵢ - ŷᵢ` = error / residual
- `(yᵢ - ŷᵢ)²` = squared error

### Intuition

It is simply the difference between the actual value and the model's predicted value, squared.

Intuitively, we can say that we are looking for a line with the **least error**, meaning its predictions are as close as possible to the actual values.

Also professionally, here are some **key important words**:

**Residual:** The difference between what the model predicted and what actually happened.

---

Now fundamentally, the line we are producing is a simple straight line following the linear equation:

$$
y = mx + c
$$

In the Linear Regression equation:

$$
\hat{y} = wx + b
$$

where:

- **X = Area** (Feature)
- **y = Actual Price** (Target)
- **w = Learned Weight / Slope**
- **b = Learned Bias / Intercept**

**Weight (`w`)**

The weight determines **how much the prediction changes when the input changes**. In a linear model, it represents the **slope of the line**.

*(Think of it like a seesaw adjusting the direction and steepness of the line.)*

**Bias (`b`)**

The bias determines **where the line starts vertically**. It shifts the entire line **up or down** and represents the **y-intercept**.

*(Think of it as moving the whole line up or down without changing its slope.)*

### For Our House-Price Example

$$
\text{Predicted Price} = w(\text{Area}) + b
$$

- **Weight →** controls how strongly **Area affects Price**
- **Bias →** adjusts the baseline price independently of Area

Both **`w`** and **`b`** are **learned from the data** using **Gradient Descent** in our implementation.

Now let's understand **how it works practically** in the following coding series.

