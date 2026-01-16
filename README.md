# supervised_machine_learning_regression_and_classification
Course 1 of 3 Machine Learning Specialization by Andrew Ng
# Machine Learning Specialization by Andrew Ng - Course 1 Summary

## **Course Title:** Supervised Machine Learning: Regression and Classification
### **Platform:** Coursera
### **Instructor:** Dr. Andrew Ng
### **Course Duration:** 3 Weeks

---

## **Week 1: Introduction to Machine Learning**

### **Module 1: Welcome to the Machine Learning Specialization**
- **Overview of the Specialization:** Three-course series covering fundamental ML concepts
- **Course 1 Focus:** Supervised Learning (Regression and Classification)
- **Prerequisites:** Basic coding experience (Python helpful but not required)
- **Tools Introduced:** Python, Jupyter Notebooks, NumPy, scikit-learn

### **Module 2: Introduction to Machine Learning**
- **What is Machine Learning?**
  - Formal definition: "A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E."
  - Practical perspective: ML systems learn to perform tasks without explicit programming

- **Types of Machine Learning:**
  1. **Supervised Learning** (Course 1 focus)
  2. **Unsupervised Learning** (Covered in later courses)
  3. **Reinforcement Learning** (Briefly mentioned)

- **Key Concepts:**
  - **Training Data:** Dataset used to train the model
  - **Features/Input Variables (x):** The input data
  - **Target/Output Variable (y):** What we're trying to predict
  - **Model/Function (f):** The mapping from input to output: `f(x) = y`

### **Module 3: Regression Model**
- **Linear Regression with One Variable (Univariate Linear Regression)**
  - **Model Representation:** `f(x) = w*x + b`
    - `w`: weight (slope/parameter)
    - `b`: bias (intercept/parameter)
  - **Notation:**
    - `x`: input/feature
    - `y`: output/target/label
    - `m`: number of training examples
    - `(x, y)`: single training example
    - `(xⁱ, yⁱ)`: i-th training example

- **Cost Function (Mean Squared Error - MSE):**
  - Measures how well the model fits the data
  - Formula: `J(w,b) = (1/2m) * Σ(f(xⁱ) - yⁱ)²`
  - Goal: Find parameters `w` and `b` that minimize `J(w,b)`

### **Module 4: Gradient Descent Algorithm**
- **Purpose:** Optimization algorithm to minimize the cost function
- **General Form:**
  ```
  Repeat until convergence {
    w = w - α * (∂J(w,b)/∂w)
    b = b - α * (∂J(w,b)/∂b)
  }
  ```
  - `α`: Learning rate (controls step size)
  - Partial derivatives: Slope of the cost function

- **Key Concepts:**
  - **Simultaneous Update:** Update all parameters simultaneously
  - **Learning Rate:**
    - Too small: Slow convergence
    - Too large: May fail to converge or diverge
  - **Convex Function:** Bowl-shaped cost function (guarantees finding global minimum for linear regression)

---

## **Week 2: Regression with Multiple Features & Classification**

### **Module 1: Multiple Linear Regression**
- **Extending to Multiple Features:**
  - Model: `f(x) = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
  - Vector notation: `f(x) = w·x + b` (dot product)
  - `n`: Number of features
  - `xⱼ`: j-th feature
  - `wⱼ`: j-th parameter (weight)

- **Vectorization for Efficiency:**
  - Using NumPy for efficient computation
  - Benefits of vectorization over for-loops
  - Code examples demonstrating performance difference

### **Module 2: Gradient Descent in Practice**
- **Feature Scaling:**
  - **Why scale?** Helps gradient descent converge faster
  - **Methods:**
    1. **Mean Normalization:** `x = (x - μ) / (max - min)`
    2. **Z-score Normalization (Standardization):** `x = (x - μ) / σ`
       - `μ`: Mean of the feature
       - `σ`: Standard deviation of the feature

- **Learning Rate Selection:**
  - Debugging gradient descent
  - **Learning rate too small:** Very slow convergence
  - **Learning rate too large:** Cost may not decrease (or diverge)
  - **Rule of thumb:** Try values like 0.001, 0.01, 0.1, 1

### **Module 3: Classification with Logistic Regression**
- **Binary Classification Problems:**
  - Examples: Email spam (0/1), tumor malignant/benign
  - Output: `y ∈ {0, 1}`

- **Logistic Regression Model:**
  - **Sigmoid/Logistic Function:** `g(z) = 1 / (1 + e⁻ᶻ)`
  - Model: `f(x) = g(w·x + b)`
  - Interpretation: Outputs probability that `y = 1` given `x`
  - Decision boundary: Predict `y = 1` if `f(x) ≥ 0.5`

- **Cost Function for Logistic Regression:**
  - **Log Loss Function:** `J(w,b) = -(1/m) * Σ[yⁱ log(f(xⁱ)) + (1-yⁱ) log(1-f(xⁱ))]`
  - Convex function (ensures gradient descent finds global minimum)

- **Gradient Descent for Logistic Regression:**
  - Same update rules as linear regression
  - Different `f(x)` function (sigmoid vs. linear)

---

## **Week 3: Advanced Topics & Practical Aspects**

### **Module 1: Overfitting and Regularization**
- **The Problem of Overfitting:**
  - **Underfitting (High Bias):** Model too simple, doesn't fit data well
  - **Overfitting (High Variance):** Model too complex, fits training data too well but fails to generalize
  - **Just Right:** Balanced model complexity

- **Addressing Overfitting:**
  1. **Collect more training data**
  2. **Feature selection** (reduce number of features)
  3. **Regularization** (reduce size/values of parameters)

- **Regularization:**
  - **Modified Cost Function:** Adds penalty for large parameters
  - For Linear Regression: `J(w,b) = (1/2m) * Σ(f(xⁱ)-yⁱ)² + (λ/2m) * Σwⱼ²`
  - For Logistic Regression: `J(w,b) = -(1/m) * Σ[...] + (λ/2m) * Σwⱼ²`
  - `λ`: Regularization parameter (controls trade-off between fitting data and keeping parameters small)

### **Module 2: Gradient Descent with Regularization**
- **Regularized Linear Regression:**
  - Gradient descent update: `wⱼ = wⱼ - α * [(1/m) * Σ(f(xⁱ)-yⁱ) * xⱼⁱ + (λ/m) * wⱼ]`
  - Bias term `b` is not regularized

- **Regularized Logistic Regression:**
  - Similar modification to gradient descent updates
  - Implementation details and code examples

### **Module 3: Practical Advice and Course Summary**
- **Machine Learning Process:**
  1. Define the problem
  2. Collect and prepare data
  3. Choose model/algorithm
  4. Train the model
  5. Evaluate performance
  6. Tune/improve model

- **Model Evaluation:**
  - Training set, validation set, test set
  - Bias-variance tradeoff analysis
  - Introduction to learning curves

- **Summary of Key Skills Learned:**
  - Implement and train linear regression models
  - Implement and train logistic regression models for classification
  - Apply gradient descent to optimize model parameters
  - Use feature scaling to improve convergence
  - Apply regularization to prevent overfitting
  - Work with multiple features using vectorization

---

## **Key Programming Tools & Libraries Introduced**
- **NumPy:** For efficient numerical computations and vectorization
- **Matplotlib:** For data visualization
- **scikit-learn:** For machine learning utilities
- **Jupyter Notebooks:** Interactive coding environment

## **Assessment Structure**
- **Quizzes:** Multiple-choice questions testing conceptual understanding
- **Programming Assignments:** Hands-on implementation in Python
- **Labs:** Interactive coding exercises with immediate feedback

## **Main Takeaways**
1. **Foundational Understanding:** Solid grasp of supervised learning basics
2. **Mathematical Intuition:** Understanding of cost functions and optimization
3. **Practical Implementation:** Ability to implement algorithms from scratch
4. **Problem-Solving Framework:** Structured approach to ML problems
