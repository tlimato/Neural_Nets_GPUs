# Author      : Tyson Limato
# Date        : 2025-6-18
# File Name   : layers.py
from mpi4py import MPI
import pandas as pd
import numpy as np
import cupy as cp

class Layer:
    """
    Abstract base class for neural network layers.

    This class defines the interface that all custom layers must implement
    to be compatible with the rest of the model's forward and backward passes.

    Methods:
    --------
    forward(x: np.ndarray) -> np.ndarray
        Computes the output of the layer for a given input.

    backward(grad: np.ndarray)
        Computes the gradient of the loss with respect to the layer's input
        using the gradient from the next layer.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray):
        raise NotImplementedError


# ------------------ Dense Layer (CPU) ------------------
class Dense(Layer):
    """
    A fully connected (dense) layer for a neural network.

    This layer performs a linear transformation: y = W·x + b

    Parameters:
    -----------
    in_dim : int
        Number of input features.
    out_dim : int
        Number of output neurons.
    lr : float
        Learning rate used for parameter updates (default: 0.001).

    Methods:
    --------
    forward(x: np.ndarray) -> np.ndarray
        Computes the output of the layer for a given input `x`.

    backward(grad: np.ndarray) -> tuple
        Computes gradients with respect to input, weights, and biases.

    apply(dw: np.ndarray, db: np.ndarray)
        Updates weights and biases using the given gradients and learning rate.
    """

    def __init__(self, in_dim, out_dim, lr=0.001):
        # Initialize weights and biases with uniform distribution in [-0.5, 0.5]
        self.weights = np.random.uniform(-0.5, 0.5, (out_dim, in_dim))
        self.biases  = np.random.uniform(-0.5, 0.5, out_dim)
        self.lr = lr  # Learning rate for updates

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass through the dense layer.
        """
        self.x = x  # Store input for use in backward pass
        return self.weights.dot(x) + self.biases

    def backward(self, grad: np.ndarray):
        """
        Perform the backward pass (gradient computation).

        Parameters:
        -----------
        grad : np.ndarray
            Gradient from the next layer (with respect to this layer's output).

        Returns:
        --------
        tuple (dx, dw, db)
            dx: Gradient with respect to input x
            dw: Gradient with respect to weights
            db: Gradient with respect to biases
        """
        dx = self.weights.T.dot(grad)          # Gradient w.r.t input
        dw = np.outer(grad, self.x)            # Gradient w.r.t weights
        db = grad.copy()                       # Gradient w.r.t biases
        return dx, dw, db

    def apply(self, dw: np.ndarray, db: np.ndarray):
        """
        Apply the computed gradients to update weights and biases.
        """
        self.weights -= self.lr * dw
        self.biases  -= self.lr * db

# ------------------ Dense Layer (GPU) ------------------
class DenseGPU(Layer):
    """
    A fully connected (dense) layer for GPU using CuPy.
    Supports both per-sample (1D) and batched (2D) inputs.
    """
    def __init__(self, in_dim, out_dim, lr=0.001):
        self.weights = cp.random.uniform(-0.5, 0.5, (out_dim, in_dim))
        self.biases  = cp.random.uniform(-0.5, 0.5, out_dim)
        self.lr = lr

    def forward(self, x):
        # ensure GPU array and store input for backward
        x = cp.asarray(x)
        self.x = x
        if x.ndim == 1:
            # Single-sample: (out_dim, in_dim) @ (in_dim,) -> (out_dim,)
            return self.weights @ x + self.biases
        elif x.ndim == 2:
            # Batched: (batch, in_dim) @ (in_dim, out_dim) -> (batch, out_dim)
            return x @ self.weights.T + self.biases[None, :]
        else:
            raise ValueError(f"DenseGPU.forward: unsupported input ndim={x.ndim}")

    def backward(self, grad):
        grad = cp.asarray(grad)
        if grad.ndim == 1:
            # Single-sample backward
            dx = self.weights.T @ grad
            dw = cp.outer(grad, self.x)
            db = grad.copy()
        else:
            # Batched backward
            dx = grad @ self.weights
            dw = grad.T @ self.x
            db = cp.sum(grad, axis=0)
        return dx, dw, db

    def apply(self, dw, db):
        self.weights -= self.lr * dw
        self.biases  -= self.lr * db

# ------------------ Activation (GPU) ------------------
class ReLUGPU(Layer):
    """
    ReLU activation for GPU using CuPy.
    Applies f(x) = max(0, x) elementwise, supporting batch dims.
    """
    def forward(self, x):
        x = cp.asarray(x)
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        grad_gpu = cp.asarray(grad)
        return grad_gpu * self.mask


# ------------------ Activation (CPU) ------------------
class ReLU(Layer):
    """
    ReLU (Rectified Linear Unit) activation layer.

    Applies the element-wise function: f(x) = max(0, x)

    Methods:
    --------
    forward(x: np.ndarray) -> np.ndarray
        Applies the ReLU activation to the input.

    backward(grad: np.ndarray) -> np.ndarray
        Computes the gradient of the loss with respect to the input.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the ReLU activation.

        Parameters:
        -----------
        x : np.ndarray
            Input array.

        Returns:
        --------
        np.ndarray
            Output array where negative values are replaced with 0.
        """
        self.mask = x > 0  # Boolean mask where input > 0
        return x * self.mask  # Zero out negative elements

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for the ReLU activation.
        """
        return grad * self.mask  # Pass gradient only where input was > 0