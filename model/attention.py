import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim

        self.key = nn.Linear(self.embedding_dim, self.attention_dim, bias=False)
        self.query = nn.Linear(self.embedding_dim, self.attention_dim, bias=False)
        self.value = nn.Linear(self.embedding_dim, self.attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)
        
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        attention_score = (Q @ K.transpose(-2, -1)) / math.sqrt(self.attention_dim)

        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        sequence_length = attention_score.shape[2]
        mask = torch.tril(torch.ones(sequence_length, sequence_length))
        masked_scores = attention_score.masked_fill(mask==0, float('-inf'))

        # 4. Apply softmax(dim=2) to masked scores
        attention_weights = masked_scores.softmax(dim = 2)

        # 5. Return (scores @ V) rounded to 4 decimal places
        return torch.round(attention_weights @ V, decimals = 4)

