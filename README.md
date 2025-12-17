# Implementation of Primal and Dual SVM
### Noa Mediavilla & Rebeca Torrecilla

![Language](https://img.shields.io/badge/Python-3.8%2B-blue)
![Subject](https://img.shields.io/badge/Course-Mathematical_Optimization-green)
![Institution](https://img.shields.io/badge/University-UPC-blue)

## Project Overview
The main objective is the manual implementation and analysis of **Support Vector Machines (SVM)** from an optimization perspective. The project explores both the **Primal** and **Dual** formulations, verifies the Strong Duality theorem, and implements **Kernels** (Gaussian/RBF) to handle non-linearly separable data.

## Mathematical Formulation
The project solves the SVM optimization problem using quadratic programming.

### 1. Primal Problem (Soft-Margin)
We minimize the classification error while maximizing the margin:
$$ \min_{w, \gamma, s} \frac{1}{2}w^T w + \nu e^T s $$
Subject to:
$$ -Y(Aw + \gamma e) - s + e \le 0 $$
$$ -s \le 0 $$

### 2. Dual Problem
We maximize the Lagrangian multipliers to find the support vectors:
$$ \max_{\lambda} \lambda^T e - \frac{1}{2} \lambda^T Y A A^T Y \lambda $$
Subject to:
$$ \lambda^T Y e = 0 $$
$$ 0 \le \lambda \le \nu $$

### 3. Kernel Trick
For non-linear problems, we replaced the dot product with a **Gaussian Kernel (RBF)**:
$$ K(x, y) = \exp \left( - \frac{||x - y||^2}{2\sigma^2} \right) $$

## Experiments & Datasets
We tested the implementation on three different scenarios to analyze the impact of the regularization parameter ($\nu$) and the kernel choice.

### 1. Linearly Separable Data (Artificial)
*   **Goal:** Validate the correctness of the Primal and Dual implementations.
*   **Result:** Verified **Strong Duality** (Objective function values coincided for both formulations).

### 2. Banknote Authentication Dataset
*   **Description:** 1,400 training points / 600 test points. Features extracted via Wavelet Transform (Variance, Skewness, Kurtosis, Entropy).
*   **Findings:**
    *   For $\nu=0$, the model failed to learn (trivial solution).
    *   For $\nu \in [0.5, 0.75]$, the model achieved **~99.27% accuracy**.
    *   **Primal vs Dual:** Comparison showed similar execution times (~0.03s vs ~6s depending on $\nu$) and identical optimal values, confirming the robustness of the implementation.

### 3. Non-Linearly Separable Data (Swiss Roll)
*   **Challenge:** The data is shaped in a spiral/roll, making it impossible for a linear hyperplane to separate the classes.
*   **Linear SVM:** Failed to classify correctly (accuracy remained low regardless of $\nu$).
*   **Gaussian Kernel SVM:** Successfully "unrolled" the data in high-dimensional space, achieving **~99% accuracy**.

## Conclusions
1. **Duality**: The results experimentally confirmed the Strong Duality Theorem; primal and dual objective functions yielded identical values.
2. **Regularization** ($\nu$): A proper choice of $\nu$ is critical. Low values lead to underfitting (trivial solutions), while excessively high values cause overfitting and increased model complexity (more support vectors).
3. **Kernels**: Linear formulations are insufficient for complex geometries like the "Swiss Roll". The introduction of the Gaussian Kernel allowed the SVM to adapt to the spiral distribution without significantly increasing computational cost in the Dual formulation. 

