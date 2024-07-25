import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from token_embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding
from single_query_attention import SingleQueryAttention

class Attention(SingleQueryAttention):
    """
    Attention is a module that performs the attention mechanism, extending the SingleQueryAttention class.
    
    Methods:
    --------
    forward(current_tokens: torch.Tensor, context_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Performs the forward pass of the attention mechanism.
    """
    def __init__(self, input_dim: int, attention_dim: int, output_dim: int) -> None:
        """
        Initialises the Attention module.

        Parameters:
        -----------
        input_dim : int
            The dimension of the input tokens.
        attention_dim : int
            The dimension of the attention mechanism.
        output_dim : int
            The dimension of the output tokens.
        """
        super().__init__(input_dim, attention_dim, output_dim)

    def forward(self, current_tokens: torch.Tensor, context_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Attention module.

        Performs the attention mechanism on the input tokens.

        Parameters:
        -----------
        current_tokens : torch.Tensor
            A tensor of current tokens for which attention is calculated.
        context_tokens : torch.Tensor
            A tensor of context tokens against which attention is calculated.
        mask : Optional[torch.Tensor], optional
            An optional mask tensor to apply to the attention scores.

        Returns:
        --------
        torch.Tensor
            A tensor representing the attended output.
        """
        q = self.query(current_tokens)
        k = self.key(context_tokens)
        v = self.value(context_tokens)
        s = torch.einsum("ijk,ilk->ijl", [q, k])
        if mask is not None:
            s = s + mask
        alpha = nn.functional.softmax(s / np.sqrt(self.attention_dim), dim=-1)
        v = torch.einsum("ijk,ilm->ilk", [v, alpha])
        return v
    
if __name__ == "__main__":
    # parameters
    vocabulary_size = 1000
    batch_size = 16
    max_sequence_length = 100
    embedding_dim = 32
    mask = torch.tril(torch.ones(max_sequence_length, max_sequence_length // 3))
    # create artificial tokens sequences 
    current_tokens = torch.randint(0, vocabulary_size, (batch_size, max_sequence_length))
    context_tokens = torch.randint(0, vocabulary_size, (batch_size, max_sequence_length // 3))
    # processing layers
    token_embedding = TokenEmbedding(vocabulary_size, embedding_dim)
    positional_embedding = PositionalEmbedding(max_sequence_length, embedding_dim)
    attention = Attention(embedding_dim, embedding_dim, embedding_dim)
    # processing
    current_tokens_embeddings = token_embedding(current_tokens) + positional_embedding(max_sequence_length)
    context_tokens_embeddings = token_embedding(context_tokens) + positional_embedding(max_sequence_length // 3)
    output = attention(current_tokens_embeddings, context_tokens_embeddings)
    # (batch_size, max_sequence_length, embedding_dim)