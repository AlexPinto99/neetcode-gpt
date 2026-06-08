import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        n_features = X.shape[1]
        n_samples = X.shape[0]
        w = np.zeros(n_features)
        b = 0
        for i in range(epochs):
            y_hat = X @ w + b
            MSE = (1/n_samples)*np.sum((y_hat-y)**2)
            dW = (2/n_samples)*X.T@(y_hat-y)
            db = (2/n_samples)*np.sum(y_hat-y)
            w = w - lr*dW
            b = b - lr*db
        return (np.round(w,5), np.round(b,5))
