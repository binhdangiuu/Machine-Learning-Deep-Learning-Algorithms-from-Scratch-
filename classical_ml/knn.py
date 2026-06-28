"""
K-Nearest Neighbors implemented from scratch using only NumPy.

STATUS: TODO - not yet implemented.

Idea: for each query point, compute distance to all training points,
take the k closest, and predict by majority vote (classification)
or average (regression). No training phase - all computation happens at predict time.
"""

import numpy as np

# TODO: implement KNN class with vectorized distance computation
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        pass
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        pass
    
    def predict(self, X):
        #distance calculation
        distances = []
        for x in X:
            #x_train is a sample, X_train is an array of samples
            distance = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            distances.append(distance)
        distances = np.array(distances)
        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_labels = self.y_train[k_indices].astype(int)
        return np.array([np.bincount(k_labels[i]).argmax() for i in range(len(k_labels))])
if __name__ == "__main__":
    np.random.seed(42)

    X0 = np.random.randn(100, 2) + np.array([2, 2])
    X1 = np.random.randn(100, 2) + np.array([-2, -2])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(100), np.ones(100)])

    indices = np.random.permutation(200)
    X_train, X_test = X[indices[:160]], X[indices[160:]]
    y_train, y_test = y[indices[:160]], y[indices[160:]]

    model = KNN(k=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}  (expected > 0.90)")
