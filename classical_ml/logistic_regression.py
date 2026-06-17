"""
Logistic Regression implemented from scratch using only NumPy.

STATUS: TODO - not yet implemented.

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
