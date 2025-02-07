import torch 

class Loss:
  """
  A class representing a loss function.
  """
  def __init__(self):
    pass # No need to initialize anything
  
  def __call__(self, *args, **kwds):
    raise NotImplementedError
  
  def backward(self, *args, **kwds):
    raise NotImplementedError
  
class MSE(Loss):
  """
  A class representing the mean squared error loss.
  """
  def __init__(self):
    super().__init__()
  
  def __call__(self, y_pred, y_true):
    """
    When the object is called, it calculates the mean squared error loss.
    """
    return torch.mean((y_pred - y_true)**2)
  
  def backward(self, y_pred, y_true):
    n_samples = y_pred.shape[0]
    dL_dy_pred = 2*(y_pred - y_true).t()/n_samples
    return dL_dy_pred