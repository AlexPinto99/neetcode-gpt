import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Rounding to 5 decimal places
        prediction = np.dot(X, weights)
        return np.round(prediction, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Computing the mean squared error between predictions and ground truth
        # Rounding to 5 decimal places
        n = len(ground_truth)
        mse = 1/n * np.sum(np.square(model_prediction - ground_truth))
        return np.round(mse, 5)
