import torch
import torch.nn as nn
from typing import Any

class PositionalEmbedding(nn.Module):
    """
    PositionalEmbedding is a module that generates positional embeddings for sequence positions.
    
    Attributes:
    -----------
    max_sequence_length : int
        The maximum length of the sequences.
    embedding_dim : int
        The dimension of the embedding vectors.
    embedding : nn.Parameter
        The learnable positional embedding matrix initialised with a standard normal distribution.

    Methods:
    --------
    forward(position: int) -> torch.Tensor:
        Retrieves the positional embeddings for the given positions.
    """
    def __init__(self, max_sequence_length: int, embedding_dim: int, **kwargs: Any) -> None:
        """
        Initialises the PositionalEmbedding module.

        Parameters:
        -----------
        max_sequence_length : int
            The maximum length of the sequences.
        embedding_dim : int
            The dimension of the embedding vectors.
        **kwargs : dict
            Additional keyword arguments (not used in this implementation).
        """
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.embedding = nn.Parameter(torch.normal(0, 1, (self.max_sequence_length, self.embedding_dim)))

    def forward(self, position: int) -> torch.Tensor:
        """
        Forward pass for the PositionalEmbedding module.

        Retrieves the positional embeddings for the given positions.

        Parameters:
        -----------
        position : int
            The position up to which the embeddings are retrieved.

        Returns:
        --------
        torch.Tensor
            A tensor of positional embeddings up to the specified position.
        """
        return self.embedding[:position, :]
    
if __name__ == "__main__":
    embedding_dim = 32
    max_sequence_length = 100
    positional_embedding = PositionalEmbedding(max_sequence_length, embedding_dim)
    # lookup the positional embedding of a random position in the sequence
    positional_embedding(max_sequence_length) 
    # (max_sequence_length, embedding_dim)