class Optimizer: 
  
  def __init__(self):
    pass
  
  def update(self):
    raise NotImplementedError
  
class GD(Optimizer):
  """
  A class representing the gradient descent optimizer.
  """
  def __init__(self, learning_rate):
    """
    Initializes the optimizer with a learning rate.
    """
    self.learning_rate = learning_rate
    
  def update(self, network):
    """
    Implements the gradient descent update rule.
    
    Args:
      - network: a neural network instance.
    """
    for layer in network.layers:
      for key in layer.parameters.keys():
        layer.parameters[key] -= self.learning_rate*layer.gradL_d[key]