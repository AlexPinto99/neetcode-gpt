import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        all_text_samples = positive + negative
        vocab = sorted({token for text_sample in all_text_samples for token in text_sample.split()})
        word_to_id = {token: index + 1 for index,token in enumerate(vocab)}
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        encoded_sequences = [
            torch.tensor([word_to_id[token] for token in sample.split()])
            for sample in all_text_samples
        ]
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return nn.utils.rnn.pad_sequence(encoded_sequences, batch_first=True)
