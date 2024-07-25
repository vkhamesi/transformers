import torch
import torch.nn as nn
from typing import Any

class TokenEmbedding(nn.Module):
    """
    TokenEmbedding is a module that converts token IDs into corresponding embedding vectors.
    
    Attributes:
    -----------
    vocabulary_size : int
        The size of the vocabulary (number of unique tokens).
    embedding_dim : int
        The dimension of the embedding vectors.
    embedding : nn.Embedding
        The learnable lookup matrix that maps token IDs to embedding vectors.

    Methods:
    --------
    forward(token_id: torch.Tensor) -> torch.Tensor:
        Converts token IDs into embedding vectors.
    """
    def __init__(self, vocabulary_size: int, embedding_dim: int, **kwargs: Any) -> None:
        """
        Initialises the TokenEmbedding module.

        Parameters:
        -----------
        vocabulary_size : int
            The size of the vocabulary (number of unique tokens).
        embedding_dim : int
            The dimension of the embedding vectors.
        **kwargs : dict
            Additional keyword arguments to pass to the nn.Embedding layer.
        """
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_dim, **kwargs)
    
    def forward(self, token_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TokenEmbedding module.

        Performs a lookup in the embedding matrix for the given token IDs.

        Parameters:
        -----------
        token_id : torch.Tensor
            A tensor of token IDs to be converted into embedding vectors.

        Returns:
        --------
        torch.Tensor
            A tensor of embedding vectors corresponding to the input token IDs.
        """
        return self.embedding(token_id)
    
if __name__ == "__main__":
    vocabulary_size = 1000
    embedding_dim = 32
    token_embedding = TokenEmbedding(vocabulary_size, embedding_dim)
    # artificial sequences of tokens
    batch_size = 16
    max_sequence_length = 100
    example_sequences = torch.randint(0, vocabulary_size, (batch_size, max_sequence_length))
    # lookup the embedding of each token for each sequence
    example_embeddings = token_embedding(example_sequences) 
    # (batch_size, max_sequence_length, embedding_dim)