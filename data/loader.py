import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch

        # We set the seed for reproducibility
        torch.manual_seed(0)

        # Generating the starting indices, considering we cannot go over the data length
        starting_indices = torch.randint(0, len(data) - context_length, (batch_size,))
        
        # We extract the slices and then stack them in 2D tensors and then return
        X = torch.stack([data[start:start+context_length] for start in starting_indices])
        Y = torch.stack([data[start+1:start+context_length+1] for start in starting_indices])
        return (X, Y)

        
