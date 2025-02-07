class Sequential:
  """
  A class representing a sequential neural network.
  
  It has the following attributes:
    - layers: a list of layers in the sequential network.
  It has the following methods:
    - forward: performs a forward pass through the network.
  """
  def __init__(self, layers):
    """
    Initializes the network with a list of layers.
    
    Args:
      - layers: list of layers to be added to the network. Default is an empty list. Each layer must be an instance of the Layer class.
    """
    self.layers = layers
    
  def forward(self, x):
    """
    Performs a forward pass through the network.
    
    Args:
      - x: input tensor.
    """
    for layer in self.layers:
      x = layer.forward(x)
    return x