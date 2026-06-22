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
    def __init__(self, lr = 0.01, n_iters = 1000): #set everythinnn
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        pass
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0
        for i in range (self.n_iters):
            y_pred = X @ w + b
            error = y_pred - y
            dw = (1/n_samples)*(X.T @ error) #many vectors -> cannot use np.sum()
            db = (1/n_samples)*np.sum(error)
            w -= self.lr*dw   #update w and b in the opposite direction of the gradient to minimize cost/loss
            b -= self.lr*db
        self.w = w #w and b go to optimal value after looping yayyyy
        self.b = b
        pass

    def predict(self, X):
        return X @ self.w + self.b #@ is matrix multiplicationnn
        pass

#Test the implementation HERE
if __name__ == "__main__":
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 3.5 * X.squeeze() + 4.0 + np.random.randn(100) * 0.5

    model = LinearRegression(lr=0.1, n_iters=1000)
    model.fit(X, y)

    print(f"weight: {model.w[0]:.4f}  (expected ≈ 3.5)")
    print(f"bias:   {model.b:.4f}  (expected ≈ 4.0)")

