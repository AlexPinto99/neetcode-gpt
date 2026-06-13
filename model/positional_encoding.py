import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        PE = np.zeros((seq_len, d_model))
        pos = np.arange(seq_len)[:,np.newaxis]
        i_even = np.arange(0, d_model, 2)
        angles = pos/10000**(i_even/d_model)
        PE[:, 0::2] = np.sin(angles)
        PE[:, 1::2] = np.cos(angles)
        return np.round(PE, 5)

