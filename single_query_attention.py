import torch
import torch.nn as nn
import numpy as np
from token_embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding

class SingleQueryAttention(nn.Module):
    """
    SingleQueryAttention is a module that performs single query attention mechanism.
    
    Attributes:
    -----------
    input_dim : int
        The dimension of the input tokens.
    attention_dim : int
        The dimension of the attention mechanism.
    output_dim : int
        The dimension of the output tokens.
    key : nn.Linear
        The learnable linear transformation for keys.
    query : nn.Linear
        The learnable linear transformation for queries.
    value : nn.Linear
        The learnable linear transformation for values.

    Methods:
    --------
    forward(current_token: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        Performs the forward pass of the single query attention mechanism.
    """
    def __init__(self, input_dim: int, attention_dim: int, output_dim: int) -> None:
        """
        Initialises the SingleQueryAttention module.

        Parameters:
        -----------
        input_dim : int
            The dimension of the input tokens.
        attention_dim : int
            The dimension of the attention mechanism.
        output_dim : int
            The dimension of the output tokens.
        """
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.output_dim = output_dim
        self.key = nn.Linear(self.input_dim, self.attention_dim)
        self.query = nn.Linear(self.input_dim, self.attention_dim)
        self.value = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, current_token: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SingleQueryAttention module.

        Performs the single query attention mechanism on the input tokens.

        Parameters:
        -----------
        current_token : torch.Tensor
            The current token tensor for which attention is calculated.
        context_tokens : torch.Tensor
            The context tokens tensor against which attention is calculated.

        Returns:
        --------
        torch.Tensor
            A tensor representing the attended output.
        """
        q = self.query(current_token)
        k = self.key(context_tokens)
        v = self.value(context_tokens)
        s = torch.einsum("ijk,ilk->ilk", [q, k])
        alpha = nn.functional.softmax(s / np.sqrt(self.attention_dim), dim=-1)
        v = torch.einsum("ijk,ijk->ik", [alpha, v])
        return v[:, None, :]

if __name__ == "__main__":
    # parameters
    vocabulary_size = 1000
    batch_size = 16
    sequence_length = 100
    embedding_dim = 32
    # create artificial tokens sequences 
    current_token = torch.randint(0, vocabulary_size, (batch_size, 1))
    context_tokens = torch.randint(0, vocabulary_size, (batch_size, sequence_length))
    # processing layers
    token_embedding = TokenEmbedding(vocabulary_size, embedding_dim)
    positional_embedding = PositionalEmbedding(sequence_length, embedding_dim)
    single_query_attention = SingleQueryAttention(embedding_dim, embedding_dim, embedding_dim)
    # processing
    current_token_embeddings = token_embedding(current_token) + positional_embedding(1)
    context_tokens_embeddings = token_embedding(context_tokens) + positional_embedding(sequence_length)
    output = single_query_attention(current_token_embeddings, context_tokens_embeddings) 
    # (batch_size, 1, embedding_dim)