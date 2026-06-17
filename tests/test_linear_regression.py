"""
Correctness check: compare our from-scratch LinearRegression against
scikit-learn's implementation on the same synthetic dataset.
"""

import numpy as np
from sklearn.linear_model import LinearRegression as SKLearnLinearRegression
from classical_ml.linear_regression import LinearRegression


def test_linear_regression_matches_sklearn():
    np.random.seed(0)
    X = 2 * np.random.rand(150, 1)
    y = 3.0 * X.squeeze() + 5.0 + np.random.randn(150) * 0.3

    ours = LinearRegression(learning_rate=0.1, n_iterations=2000, method="gradient_descent")
    ours.fit(X, y)

    sk = SKLearnLinearRegression()
    sk.fit(X, y)

    assert abs(ours.weights[0] - sk.coef_[0]) < 0.05
    assert abs(ours.bias - sk.intercept_) < 0.05
    print("PASSED: LinearRegression matches scikit-learn within tolerance.")


if __name__ == "__main__":
    test_linear_regression_matches_sklearn()
