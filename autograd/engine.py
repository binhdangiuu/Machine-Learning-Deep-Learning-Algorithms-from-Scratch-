"""
Mini autograd engine, inspired by Andrej Karpathy's micrograd.

STATUS: TODO - not yet implemented. (Giai doan 3 trong roadmap)

Core idea: a Value class wraps a scalar/array and tracks the operation
that produced it. Calling .backward() walks the computational graph in
reverse topological order, applying the chain rule at each node.
"""

# TODO: implement Value class with +, *, **, tanh, relu, and backward()
