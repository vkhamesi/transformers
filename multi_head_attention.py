import torch
import torch.nn as nn
from typing import Any, Optional
from token_embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding
from attention import Attention

class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention is a module that performs multi-head attention mechanism.
    
    Attributes:
    -----------
    num_heads : int
        The number of attention heads.
    input_dim : int
        The dimension of the input tokens.
    attention_dim : int
        The dimension of the attention mechanism.
    output_dim : int
        The dimension of the output tokens.
    heads : nn.ModuleList
        A list of Attention modules for each head.
    output_projection : nn.Linear
        The linear layer for projecting concatenated outputs from all heads.

    Methods:
    --------
    forward(current_tokens: torch.Tensor, context_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Performs the forward pass of the multi-head attention mechanism.
    """
    def __init__(self, num_heads: int, input_dim: int, attention_dim: int, output_dim: int, **kwargs: Any) -> None:
        """
        Initialises the MultiHeadAttention module.

        Parameters:
        -----------
        num_heads : int
            The number of attention heads.
        input_dim : int
            The dimension of the input tokens.
        attention_dim : int
            The dimension of the attention mechanism.
        output_dim : int
            The dimension of the output tokens.
        **kwargs : dict
            Additional keyword arguments to pass to the Attention modules.
        """
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.output_dim = output_dim
        self.heads = nn.ModuleList([
            Attention(input_dim, attention_dim, output_dim, **kwargs) for _ in range(self.num_heads)
        ])
        self.output_projection = nn.Linear(self.num_heads * self.output_dim, self.output_dim)

    def forward(self, current_tokens: torch.Tensor, context_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the MultiHeadAttention module.

        Performs the multi-head attention mechanism on the input tokens.

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
        yh = [head(current_tokens, context_tokens, mask) for head in self.heads]
        y = torch.cat(yh, axis=2)
        v = self.output_projection(y)
        return v
    
if __name__ == "__main__":
    # parameters
    vocabulary_size = 1000
    batch_size = 16
    sequence_length = 100
    embedding_dim = 32
    num_heads = 4
    # create artificial tokens sequences 
    current_tokens = torch.randint(0, vocabulary_size, (batch_size, sequence_length))
    context_tokens = torch.randint(0, vocabulary_size, (batch_size, sequence_length))
    # processing layers
    token_embedding = TokenEmbedding(vocabulary_size, embedding_dim)
    positional_embedding = PositionalEmbedding(sequence_length, embedding_dim)
    multi_head_attention = MultiHeadAttention(num_heads, embedding_dim, embedding_dim, embedding_dim)
    # processing
    current_tokens_embeddings = token_embedding(current_tokens) + positional_embedding(sequence_length)
    context_tokens_embeddings = token_embedding(context_tokens) + positional_embedding(sequence_length)
    output = multi_head_attention(current_tokens_embeddings, context_tokens_embeddings)
    # (batch_size, max_sequence_length, embedding_dim)