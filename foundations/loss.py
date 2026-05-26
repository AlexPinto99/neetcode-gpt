import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # We add a small epsilon (1e-7) to y_pred to avoid log(0)
        n = len(y_true)
        bce = -(1/n)*np.sum(y_true* np.log(y_pred+(1e-7)) + (1-y_true)*np.log(1+(1e-7)-y_pred))
        return round(bce, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # We add a small epsilon (1e-7) to y_pred to avoid log(0)
        n = len(y_true)
        cce = -(1/n)*np.sum(np.sum(y_true* np.log(y_pred+(1e-7))))
        return round(cce, 4)
