import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x = np.array(x)
        gamma = np.array(gamma)
        mean_sqr = np.mean(x**2)
        rms = np.sqrt(mean_sqr + eps)
        x_hat = x/rms
        output = gamma*x_hat
        return np.round(output, decimals=4)
