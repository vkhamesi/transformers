import torch
import torch.nn as nn
from typing import Any

class TokenUnembedding(nn.Module):
    """
    TokenUnembedding is a module that converts embedding vectors back to token probabilities.
    
    Attributes:
    -----------
    vocabulary_size : int
        The size of the vocabulary (number of unique tokens).
    embedding_dim : int
        The dimension of the embedding vectors.
    unembedding_matrix : nn.Linear
        The linear layer that maps embedding vectors to vocabulary logits.
    softmax : nn.Softmax
        The softmax layer to convert logits to probabilities.

    Methods:
    --------
    forward(embedding: torch.Tensor) -> torch.Tensor:
        Performs the forward pass to convert embeddings to token probabilities.
    """
    def __init__(self, vocabulary_size: int, embedding_dim: int, **kwargs: Any) -> None:
        """
        Initialises the TokenUnembedding module.

        Parameters:
        -----------
        vocabulary_size : int
            The size of the vocabulary (number of unique tokens).
        embedding_dim : int
            The dimension of the embedding vectors.
        **kwargs : dict
            Additional keyword arguments to pass to the nn.Linear layer.
        """
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.unembedding_matrix = nn.Linear(self.embedding_dim, self.vocabulary_size, **kwargs)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TokenUnembedding module.

        Converts embedding vectors to token probabilities.

        Parameters:
        -----------
        embedding : torch.Tensor
            A tensor of embedding vectors to be converted to token probabilities.

        Returns:
        --------
        torch.Tensor
            A tensor of token probabilities corresponding to the input embeddings.
        """
        logits = self.unembedding_matrix(embedding)
        token = self.softmax(logits)
        return token
    
if __name__ == "__main__":
    vocabulary_size = 1000
    embedding_dim = 32
    batch_size = 16
    token_unemb = TokenUnembedding(vocabulary_size, embedding_dim)
    embeddings = torch.normal(0, 1, (batch_size, embedding_dim))
    tokens = token_unemb(embeddings)
    # (max_sequence_length, vocabulary_size)