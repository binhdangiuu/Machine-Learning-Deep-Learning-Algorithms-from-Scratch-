"""
Logistic Regression implemented from scratch using only NumPy.

STATUS: DONE

Math reference (for when you implement this):
    Model:   y_hat = sigmoid(X @ w + b),  sigmoid(z) = 1 / (1 + exp(-z))
    Cost:    Binary Cross-Entropy
             J(w, b) = -(1/m) * sum( y*log(y_hat) + (1-y)*log(1-y_hat) )
    Gradients (same form as linear regression, because of how sigmoid + BCE combine):
             dJ/dw = (1/m) * X.T @ (y_hat - y)
             dJ/db = (1/m) * sum(y_hat - y)
"""

import numpy as np

# TODO: implement LogisticRegression class (binary + optional softmax for multiclass)
class LogisticRegression:
    def __init__(self, lr = 0.01, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        pass
    
    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))
        pass
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0
        for i in range (self.n_iters):
            y_pred = self._sigmoid(X @ w + b)
            error = y_pred - y
            dw = 1/n_samples*(X.T@error)
            db = 1/n_samples*np.sum(error)
            w -= self.lr*dw
            b -= self.lr*db
        self.w = w
        self.b = b
        pass
    
    def predict(self, X):
        z = X @ self.w + self.b
        y_pred = self._sigmoid(z)
        return (y_pred >= 0.5).astype(int) #threshold at 0.5
        pass

#Test the implementation HERE
if __name__ == "__main__":
    np.random.seed(42)

    X0 = np.random.randn(100, 2) + np.array([2, 2])    # from (1,1)
    X1 = np.random.randn(100, 2) + np.array([-2, -2])  # from (-1,-1)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(100), np.ones(100)])

    indices = np.random.permutation(200)
    X_train, X_test = X[indices[:160]], X[indices[160:]]
    y_train, y_test = y[indices[:160]], y[indices[160:]]

    model = LogisticRegression(lr=0.1, n_iters=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}  (expected > 0.90)")
        
