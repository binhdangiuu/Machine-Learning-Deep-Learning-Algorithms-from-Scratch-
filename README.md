# MLFromScratch

> Machine Learning & Deep Learning algorithms implemented from scratch using only NumPy — no scikit-learn, no PyTorch autograd. Built to understand the math and mechanics behind the libraries we use every day.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![NumPy](https://img.shields.io/badge/Dependencies-NumPy%20only-orange)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## Table of Contents

- [Why This Project](#why-this-project)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithms Implemented](#algorithms-implemented)
  - [Classical Machine Learning](#1-classical-machine-learning)
  - [Neural Network from Scratch](#2-neural-network-from-scratch)
  - [Mini Autograd Engine](#3-mini-autograd-engine)
- [Benchmarks](#benchmarks)
- [Roadmap](#roadmap)
- [References](#references)
- [License](#license)

## Why This Project

Most courses and tutorials teach you to call `model.fit()` or `loss.backward()` without ever showing what happens inside. This repository is my attempt to open that black box: every algorithm here is implemented from first principles, using nothing but NumPy for matrix operations and linear algebra.

The goals are:

- Understand the math behind classical ML algorithms by deriving and implementing them directly.
- Understand backpropagation by computing gradients by hand instead of relying on autograd.
- Understand how frameworks like PyTorch work internally by building a small autograd engine.
- Validate correctness by benchmarking every implementation against scikit-learn / PyTorch equivalents.

## Repository Structure

```
MLFromScratch/
├── classical_ml/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── kmeans.py
│   ├── decision_tree.py
│   ├── naive_bayes.py
│   └── pca.py
├── neural_network/
│   ├── layers.py
│   ├── activations.py
│   ├── losses.py
│   ├── optimizers.py
│   └── network.py
├── autograd/
│   ├── engine.py          # core Value class + backward()
│   └── nn.py               # small NN built on top of the engine
├── notebooks/
│   ├── 01_linear_regression.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── ...
│   └── 0X_autograd_demo.ipynb
├── tests/
│   └── test_*.py            # correctness checks vs scikit-learn
├── data/
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/MLFromScratch.git
cd MLFromScratch
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies are intentionally minimal: `numpy`, `matplotlib` for visualization, and `scikit-learn` / `torch` only inside `tests/` for benchmarking — never inside the core implementations.

## Usage

Each algorithm can be run standalone, for example:

```bash
python classical_ml/linear_regression.py
```

Or explored interactively through the notebooks in `notebooks/`, which include visualizations such as decision boundaries, loss curves, and clustering results.

## Algorithms Implemented

### 1. Classical Machine Learning

| Algorithm | Status | Notes |
|---|---|---|
| Linear Regression | ✅ | Gradient Descent + Normal Equation |
| Logistic Regression | ✅ | Binary & multiclass (softmax) |
| K-Nearest Neighbors | ✅ | Vectorized distance computation |
| K-Means | ✅ | Random & K-Means++ initialization |
| Decision Tree | ✅ | Gini / Entropy split criteria |
| Naive Bayes | ✅ | Gaussian variant |
| PCA | ✅ | Via eigen-decomposition of covariance matrix |

Each implementation includes a docstring with the underlying math (cost function, gradient derivation) and a comparison script against the equivalent scikit-learn model to validate correctness.

### 2. Neural Network from Scratch

A fully-connected Neural Network built without any deep learning framework:

- Forward pass implemented manually for each layer.
- Backpropagation derived and implemented by hand (chain rule applied explicitly, not via autograd).
- Activations: ReLU, Sigmoid, Tanh, Softmax.
- Loss functions: MSE, Cross-Entropy.
- Optimizers: SGD, Momentum, Adam.
- Trained and evaluated on MNIST, with accuracy compared against an equivalent PyTorch model.

### 3. Mini Autograd Engine

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), this is a small reverse-mode automatic differentiation engine built from scratch:

- A `Value` class that tracks operations and builds a computational graph.
- `backward()` implemented via topological sort + chain rule.
- Support for basic operations (`+`, `*`, `**`, `tanh`, `relu`, etc.).
- A tiny neural network built on top of the engine to demonstrate end-to-end training without PyTorch.

This is the part of the project that best demonstrates understanding of how modern deep learning frameworks compute gradients internally.

## Benchmarks

Every custom implementation is validated against its standard-library counterpart on the same dataset and metric, to confirm correctness rather than just "code that runs." Results and comparison tables are documented inside each notebook in `notebooks/`.

## Roadmap

- [x] Classical ML algorithms
- [x] Neural Network with manual backpropagation
- [ ] Mini autograd engine
- [ ] CNN from scratch (convolution + pooling, no framework)
- [ ] Write-up / blog post summarizing key learnings

## References

- Andrej Karpathy, [micrograd](https://github.com/karpathy/micrograd)
- Andrew Ng, Machine Learning Specialization
- Ian Goodfellow et al., *Deep Learning*
- scikit-learn and PyTorch documentation (used for benchmarking only)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Author: [Your Name] — Computer Science student (AI track), Ho Chi Minh City University of Technology (HCMUT - Bach Khoa).*
