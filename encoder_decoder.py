import torch
import torch.nn as nn
from token_embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding
from multi_head_attention import MultiHeadAttention
from resnet import ResNet
from layer_norm import LayerNorm
from token_unembedding import TokenUnembedding

class EncoderDecoderTransformer(nn.Module):
    """
    EncoderDecoderTransformer is a module that implements a Transformer model with an encoder-decoder architecture.
    
    Attributes:
    -----------
    max_sequence_length : int
        The maximum length of the input sequences.
    layer_encoder : int
        The number of encoder layers.
    layer_decoder : int
        The number of decoder layers.
    num_heads : int
        The number of attention heads.
    embedding_dim : int
        The dimension of the embedding.
    mlp_dim : int
        The dimension of the feedforward network.
    vocabulary_size : int
        The size of the vocabulary.

    Methods:
    --------
    forward(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Performs a forward pass through the encoder-decoder Transformer model.
    """
    def __init__(self, max_sequence_length: int, layer_encoder: int, layer_decoder: int, num_heads: int, 
                 embedding_dim: int, mlp_dim: int, vocabulary_size: int) -> None:
        """
        Initialises the EncoderDecoderTransformer module.

        Parameters:
        -----------
        max_sequence_length : int
            The maximum length of the input sequences.
        layer_encoder : int
            The number of encoder layers.
        layer_decoder : int
            The number of decoder layers.
        num_heads : int
            The number of attention heads.
        embedding_dim : int
            The dimension of the embedding.
        mlp_dim : int
            The dimension of the feedforward network.
        vocabulary_size : int
            The size of the vocabulary.
        """
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.layer_encoder = layer_encoder
        self.layer_decoder = layer_decoder
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.vocabulary_size = vocabulary_size

        # create layers
        self.token_embedding = TokenEmbedding(self.vocabulary_size, self.embedding_dim)
        self.positional_embedding = PositionalEmbedding(self.max_sequence_length, self.embedding_dim)
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # encoder network
        for i in range(self.layer_encoder):
            # residual attention
            multi_head_attention = MultiHeadAttention(num_heads, embedding_dim, embedding_dim, embedding_dim)
            resnet_attention = ResNet(multi_head_attention)
            self.encoder_layers.add_module(f"encoder_attention_{i}", resnet_attention)
            # layer normalisation
            layer_norm = LayerNorm(self.embedding_dim)
            self.encoder_layers.add_module(f"encoder_layer_norm0_{i}", layer_norm)
            # mlp
            mlp = nn.Sequential(
                nn.Linear(embedding_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, embedding_dim)
            )
            resnet_mlp = ResNet(mlp)
            self.encoder_layers.add_module(f"encoder_mlp_{i}", resnet_mlp)
            # layer normalisation
            layer_norm = LayerNorm(self.embedding_dim)
            self.encoder_layers.add_module(f"encoder_layer_norm1_{i}", layer_norm)

        # decoder network
        for i in range(self.layer_decoder):
            # residual attention
            multi_head_attention = MultiHeadAttention(num_heads, embedding_dim, embedding_dim, embedding_dim)
            resnet_attention = ResNet(multi_head_attention)
            self.decoder_layers.add_module(f"decoder_attention0_{i}", resnet_attention)
            # layer normalisation
            layer_norm = LayerNorm(self.embedding_dim)
            self.decoder_layers.add_module(f"decoder_layer_norm2_{i}", layer_norm)
            # residual attention
            multi_head_attention = MultiHeadAttention(num_heads, embedding_dim, embedding_dim, embedding_dim)
            resnet_attention = ResNet(multi_head_attention)
            self.decoder_layers.add_module(f"decoder_attention1_{i}", resnet_attention)
            # layer normalisation
            layer_norm = LayerNorm(self.embedding_dim)
            self.decoder_layers.add_module(f"decoder_layer_norm3_{i}", layer_norm)
            # multi-layer perceptron
            mlp = nn.Sequential(
                nn.Linear(embedding_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, embedding_dim)
            )
            resnet_mlp = ResNet(mlp)
            self.decoder_layers.add_module(f"decoder_mlp_{i}", resnet_mlp)
            # layer normalisation
            layer_norm = LayerNorm(self.embedding_dim)
            self.decoder_layers.add_module(f"decoder_layer_norm4_{i}", layer_norm)

        self.token_unembedding = TokenUnembedding(self.vocabulary_size, self.embedding_dim)
        self.register_buffer("mask", torch.tril(torch.ones(self.max_sequence_length, self.max_sequence_length))
                             .view(1, self.max_sequence_length, self.max_sequence_length))

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the EncoderDecoderTransformer module.

        Performs a forward pass through the encoder-decoder Transformer model.

        Parameters:
        -----------
        z : torch.Tensor
            The input tensor for the encoder, often referred to as the source.
        x : torch.Tensor
            The input tensor for the decoder, often referred to as the target.

        Returns:
        --------
        torch.Tensor
            The output tensor after processing through the Transformer model.
        """
        # max_sequence_length per batch
        lz = z.size()[1]
        z = self.token_embedding(z) + self.positional_embedding(lz)[None, :, :]
        for name, module in self.encoder_layers.named_children():
            if "attention" in name:
                z = module(z, z)
            else:
                z = module(z)
        # max_sequence_length per batch
        lx = x.size()[1]
        x = self.token_embedding(x) + self.positional_embedding(lx)[None, :, :]
        for name, module in self.decoder_layers.named_children():
            if "attention0" in name:
                x = module(x, x, 0)
            elif "attention1" in name:
                x = module(x, z)
            else:
                x = module(x)
        probs = self.token_unembedding(x)
        return probs
        
if __name__ == "__main__":
    # parameters
    batch_size = 16
    vocabulary_size = 1000
    max_sequence_length = 100
    num_heads = 4
    layer_encoder = 3
    layer_decoder = 3
    embedding_dim = 32
    mlp_dim = 64
    # artificial tokens sequences
    z = torch.randint(0, vocabulary_size, (batch_size, max_sequence_length))
    x = torch.randint(0, vocabulary_size, (batch_size, max_sequence_length // 3))
    # model
    transformer = EncoderDecoderTransformer(max_sequence_length, layer_decoder, layer_decoder, num_heads,
                                            embedding_dim, mlp_dim, vocabulary_size)
    output = transformer(z, x)
    # (batch_size, len(x), vocabulary_size)