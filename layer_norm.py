import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    LayerNorm is a module that applies layer normalisation to the input activations.

    Attributes:
    -----------
    shape : int
        The shape of the layer to be normalised.
    eps : int
        A small value to prevent division by zero.
    scale : nn.Parameter
        The learnable scale parameter for normalisation.
    offset : nn.Parameter
        The learnable offset parameter for normalisation.

    Methods:
    --------
    forward(activations: torch.Tensor) -> torch.Tensor:
        Applies layer normalisation to the input activations.
    """
    def __init__(self, shape: int, eps: float = 1e-5) -> None:
        """
        Initialises the LayerNorm module.

        Parameters:
        -----------
        shape : int
            The shape of the layer to be normalised.
        eps : float, optional
            A small value to prevent division by zero (default is 1e-5).
        """
        super().__init__()
        self.shape = shape
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(self.shape))
        self.offset = nn.Parameter(torch.zeros(self.shape))

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LayerNorm module.

        Applies layer normalisation to the input activations.

        Parameters:
        -----------
        activations : torch.Tensor
            A tensor of input activations to be normalised.

        Returns:
        --------
        torch.Tensor
            A tensor of normalised activations.
        """
        m = activations.mean()
        nu = activations.std()
        normalised_activations = ((activations - m) / (nu + self.eps)) * self.scale + self.offset
        return normalised_activations

if __name__ == "__main__":
    shape = 20
    activations = torch.normal(0, 1, (shape,))
    layer_norm = LayerNorm(shape)
    normalised_activations = layer_norm(activations)
    # (shape,)