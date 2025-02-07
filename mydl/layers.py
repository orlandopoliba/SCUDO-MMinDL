import torch # Importing torch module

class Layer:
  """ 
  A class representing a layer in a neural network.
  """
  
  def __init__(self):
    """
    Constructor for the Layer class. Initializes the layer with an empty list of parameters.
    """
    self.parameters = {} # Initializing the parameters dictionary
    self.gradL_d = {} # Initializing the gradient of loss with respect to parameters dictionary
  
  def forward(self, x):
    """
    Forward pass through the layer.
    """
    raise NotImplementedError # Raising an error if the forward method is not implemented in the subclass

  def backward(self, dL_dy):
    raise NotImplementedError # Raising an error if the backward method is not implemented in the subclass
  
class Linear(Layer):
  """
  A class representing a linear layer in a neural network.
  """
  def __init__(self, fan_in, fan_out):
    """
    Constructor for the Linear class. Initializes the layer with random parameters.
    
    Args:
      - fan_in: number of input units.
      - fan_out: number of output units.
      
    Parameters in this layer have the following shapes:
    W: (fan_in, fan_out)
    b: (     1, fan_out)
    """
    super().__init__()
    self.parameters['W'] = torch.randn((fan_in,fan_out), dtype=torch.float32, requires_grad=False) 
    self.parameters['b'] = torch.zeros((1,fan_out), dtype=torch.float32, requires_grad=False)
  
  def forward(self, x):
    """
    Forward pass through the linear layer. Stores the input and the output tensor for the backward pass. 
    
    Args:
      - x: input tensor. Shape (n_samples, fan_in).
    Returns:
      - y: output tensor. Shape (n_samples, fan_out).
    """
    self.x = x # Storing the input tensor for the backward pass
    return x @ self.parameters['W'] + self.parameters['b']
  
  def backward(self, dL_dy):
    """
    
    """
    self.gradL_d['W'] = (dL_dy @ self.x).t()
    self.gradL_d['b'] = (dL_dy @ torch.ones(dL_dy.shape[1],1)).t()
    return self.parameters['W'] @ dL_dy # this is dL_dx
  
class Sigmoid(Layer):
  """ 
  A class representing a sigmoid activation layer in a neural network.
  """
  
  def __init__(self):
    """
    Constructor for the Sigmoid class.
    """
    super().__init__()
    
  def forward(self, x):
    """
    Forward pass through the sigmoid activation layer. Stores the output tensor for the backward pass.
    """
    self.y = 1/(1+torch.exp(-x))
    return self.y