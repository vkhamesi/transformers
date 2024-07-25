import torch
import torch.nn as nn
from typing import Optional

class ResNet(nn.Module):
    """
    ResNet is a module that implements a residual network block.
    
    Attributes:
    -----------
    module : nn.Module
        The module to which residual connections are applied.

    Methods:
    --------
    forward(x: torch.Tensor, y: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Applies the forward pass with residual connections.
    """
    def __init__(self, module: nn.Module) -> None:
        """
        Initialises the ResNet module.

        Parameters:
        -----------
        module : nn.Module
            The module to which residual connections are applied.
        """
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the ResNet module.

        Applies the forward pass with residual connections. If `y` is provided, it is passed as an argument 
        to the `module`, along with the optional `mask`. The input tensor `x` is added to the output 
        of the `module`.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor.
        y : Optional[torch.Tensor], optional
            An optional tensor that is passed to the `module` if provided.
        mask : Optional[torch.Tensor], optional
            An optional mask tensor that is passed to the `module` if provided.

        Returns:
        --------
        torch.Tensor
            The tensor resulting from applying the `module` and adding the residual connection.
        """
        if y is None:
            return self.module(x) + x
        else:
            if mask is None:
                return self.module(x, y) + x
            else:
                return self.module(x, y, mask) + x
            
if __name__ == "__main__":
    # parameters
    batch_size = 16
    embedding_dim = 32
    # define a simple module
    module = nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim),
        nn.ReLU(),
        nn.Linear(embedding_dim, embedding_dim)
    )
    # generate the corresponding ResNet module
    resnet = ResNet(module)
    # generate artifical data
    x = torch.normal(0, 1, (batch_size, embedding_dim))
    # forward pass through the ResNet block
    output = resnet(x)
    # (batch_size, embedding_dim)