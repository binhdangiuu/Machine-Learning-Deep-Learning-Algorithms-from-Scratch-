"""
K-Means clustering implemented from scratch using only NumPy.

STATUS: TODO - not yet implemented.

Idea: iteratively (1) assign each point to its nearest centroid,
(2) recompute centroids as the mean of points assigned to them,
until centroids stop moving (convergence).
"""

import numpy as np

# TODO: implement KMeans class (random init + K-Means++ init)
class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        pass
    
    def fit(self, X):
        n_samples, n_features = X.shape
        #Randomly initialize centroids
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]
        for i in range(self.max_iters):
            #Calculate distance and Assign each point to the nearest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            #Recompute centroids as the mean of points assigned to them
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.k)])
            #Check for convergence (if centroids do not change)
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        pass
    
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
        pass
    
    def inertia(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return np.sum((X - self.centroids[labels])**2)
        pass
if __name__ == "__main__":
    np.random.seed(42)

    X0 = np.random.randn(100, 2) + np.array([2, 2])
    X1 = np.random.randn(100, 2) + np.array([-2, -2])
    X2 = np.random.randn(100, 2) + np.array([2, -2])
    X = np.vstack([X0, X1, X2])

    model = KMeans(k=3, max_iters=100)
    model.fit(X)

    print(f"Centroids:\n{model.centroids}")
    print(f"Inertia: {model.inertia(X):.4f}")