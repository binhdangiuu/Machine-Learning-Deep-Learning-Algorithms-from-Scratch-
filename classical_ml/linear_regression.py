"""
Linear Regression implemented from scratch using only NumPy.

Two training methods are provided:
    1. Gradient Descent   - iterative optimization, scales to large data.
    2. Normal Equation    - closed-form solution, exact but O(n^3) on features.

Math reference
--------------
Model:        y_hat = X @ w + b

Cost (MSE):   J(w, b) = (1 / 2m) * sum( (y_hat_i - y_i)^2 )
              where m = number of samples

Gradients (derived from J via chain rule):
              dJ/dw = (1/m) * X.T @ (y_hat - y)
              dJ/db = (1/m) * sum(y_hat - y)

Update rule (Gradient Descent):
              w := w - learning_rate * dJ/dw
              b := b - learning_rate * dJ/db

Normal Equation (closed-form, sets gradient of J to zero and solves directly):
              theta = (X_b.T @ X_b)^(-1) @ X_b.T @ y
              where X_b = X with an extra column of 1s prepended (for the bias term)
"""

import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, method="gradient_descent"):
        """
        Parameters
        ----------
        learning_rate : float
            Step size for gradient descent updates. Ignored if method="normal_equation".
        n_iterations : int
            Number of gradient descent steps. Ignored if method="normal_equation".
        method : str
            Either "gradient_descent" or "normal_equation".
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None
        self.cost_history = []  # tracks MSE at every iteration, useful for plotting

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.method == "gradient_descent":
            self._fit_gradient_descent(X, y)
        elif self.method == "normal_equation":
            self._fit_normal_equation(X, y)
        else:
            raise ValueError(f"Unknown method: '{self.method}'. Use 'gradient_descent' or 'normal_equation'.")
        return self

    def _fit_gradient_descent(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            # Mean Squared Error cost, recorded for diagnostics / plotting later
            cost = (1 / (2 * n_samples)) * np.sum(error ** 2)
            self.cost_history.append(cost)

            # Gradients of the cost function w.r.t weights and bias
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # Parameter update step
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def _fit_normal_equation(self, X, y):
        n_samples = X.shape[0]
        X_b = np.hstack([np.ones((n_samples, 1)), X])  # prepend bias column of 1s

        # pinv (pseudo-inverse) is used instead of inv to avoid crashing
        # if (X_b.T @ X_b) happens to be singular / non-invertible.
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.weights + self.bias

    def mse(self, X, y):
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        return np.mean((y_pred - y) ** 2)


if __name__ == "__main__":
    # ---- Demo: generate synthetic linear data and verify both training methods ----
    np.random.seed(42)
    n_samples = 200
    X = 2 * np.random.rand(n_samples, 1)
    true_weight, true_bias = 3.5, 4.0
    noise = np.random.randn(n_samples) * 0.5
    y = true_weight * X.squeeze() + true_bias + noise

    print(f"True parameters used to generate data: weight={true_weight}, bias={true_bias}\n")

    print("=== Gradient Descent ===")
    model_gd = LinearRegression(learning_rate=0.1, n_iterations=1000, method="gradient_descent")
    model_gd.fit(X, y)
    print(f"Learned weight={model_gd.weights[0]:.4f}, bias={model_gd.bias:.4f}")
    print(f"Final MSE={model_gd.mse(X, y):.4f}")

    print("\n=== Normal Equation ===")
    model_ne = LinearRegression(method="normal_equation")
    model_ne.fit(X, y)
    print(f"Learned weight={model_ne.weights[0]:.4f}, bias={model_ne.bias:.4f}")
    print(f"Final MSE={model_ne.mse(X, y):.4f}")

    # Optional sanity check against scikit-learn, if installed.
    try:
        from sklearn.linear_model import LinearRegression as SKLearnLinearRegression

        sk_model = SKLearnLinearRegression()
        sk_model.fit(X, y)
        print("\n=== scikit-learn (for comparison only) ===")
        print(f"weight={sk_model.coef_[0]:.4f}, bias={sk_model.intercept_:.4f}")
    except ImportError:
        print("\n(scikit-learn not installed - skipping comparison)")
