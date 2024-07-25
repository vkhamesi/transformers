import torch
import torch.nn as nn
from token_embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding
from multi_head_attention import MultiHeadAttention
from resnet import ResNet
from layer_norm import LayerNorm
from token_unembedding import TokenUnembedding

class EncoderTransformer(nn.Module):
    """
    EncoderTransformer is a module that implements the encoder part of a Transformer model.

    Attributes:
    -----------
    max_sequence_length : int
        The maximum length of the input sequences.
    layer_encoder : int
        The number of encoder layers.
    num_heads : int
        The number of attention heads.
    embedding_dim : int
        The dimension of the embedding.
    mlp_dim : int
        The dimension of the feedforward network.
    output_dim : int
        The dimension of the output.
    vocabulary_size : int
        The size of the vocabulary.

    Methods:
    --------
    forward(z: torch.Tensor) -> torch.Tensor:
        Performs a forward pass through the encoder Transformer model.
    """
    def __init__(self, max_sequence_length: int, layer_encoder: int, num_heads: int, 
                 embedding_dim: int, mlp_dim: int, output_dim: int, vocabulary_size: int) -> None:
        """
        Initialises the EncoderTransformer module.

        Parameters:
        -----------
        max_sequence_length : int
            The maximum length of the input sequences.
        layer_encoder : int
            The number of encoder layers.
        num_heads : int
            The number of attention heads.
        embedding_dim : int
            The dimension of the embedding.
        mlp_dim : int
            The dimension of the feedforward network.
        output_dim : int
            The dimension of the output.
        vocabulary_size : int
            The size of the vocabulary.
        """
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.layer_encoder = layer_encoder
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        self.vocabulary_size = vocabulary_size

        # create layers
        self.token_embedding = TokenEmbedding(self.vocabulary_size, self.embedding_dim)
        self.positional_embedding = PositionalEmbedding(self.max_sequence_length, self.embedding_dim)
        self.encoder_layers = nn.ModuleList()

        # encoder network
        for i in range(self.layer_encoder):
            # residual attention
            multi_head_attention = MultiHeadAttention(num_heads, embedding_dim, embedding_dim, embedding_dim)
            resnet_attention = ResNet(multi_head_attention)
            self.encoder_layers.add_module(f"encoder_attention_{i}", resnet_attention)
            # layer normalisation
            layer_norm = LayerNorm(self.embedding_dim)
            self.encoder_layers.add_module(f"encoder_layer_norm0_{i}", layer_norm)
            # multi-layer perceptron
            mlp = nn.Sequential(
                nn.Linear(embedding_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, embedding_dim)
            )
            resnet_mlp = ResNet(mlp)
            self.encoder_layers.add_module(f"encoder_mlp_{i}", resnet_mlp)
            # layer normalisation
            layer_norm = LayerNorm(self.embedding_dim)
            self.encoder_layers.add_module(f"encoder_layer_norm1_{i}", layer_norm)

        self.output_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.output_dim),
            nn.GELU(),
            LayerNorm(self.output_dim)
        )
        self.token_unembedding = TokenUnembedding(self.vocabulary_size, self.output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the EncoderTransformer module.

        Performs a forward pass through the encoder Transformer model.

        Parameters:
        -----------
        z : torch.Tensor
            The input tensor for the encoder.

        Returns:
        --------
        torch.Tensor
            The output tensor after processing through the Transformer encoder.
        """
        # max_sequence_length per batch
        lz = z.size()[1]
        z = self.token_embedding(z) + self.positional_embedding(lz)[None, :, :]
        for name, module in self.encoder_layers.named_children():
            if "attention" in name:
                z = module(z, z)
            else:
                z = module(z)
        z = self.output_projection(z)
        probs = self.token_unembedding(z)
        return probs
        
if __name__ == "__main__":
    # parameters
    batch_size = 16
    vocabulary_size = 1000
    max_sequence_length = 100
    num_heads = 4
    layer_encoder = 3
    embedding_dim = 32
    mlp_dim = 64
    output_dim = 128
    # artificial tokens sequences
    z = torch.randint(0, vocabulary_size, (batch_size, max_sequence_length))
    # model
    transformer = EncoderTransformer(max_sequence_length, layer_encoder, num_heads, 
                                     embedding_dim, mlp_dim, output_dim, vocabulary_size)
    output = transformer(z)
    # (batch_size, len(z), vocabulary_size)