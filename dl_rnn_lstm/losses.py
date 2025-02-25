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
    """
    Backward pass through the mean squared error loss. Returns the differential of the loss with respect to the prediction.
    
    Args:
      - y_pred: the predicted values. Shape (n_samples, n_features).
      - y_true: the true values. Shape (n_samples, n_features).
    """
    n_samples = y_pred.shape[0]
    dL_dy_pred = 2*(y_pred - y_true).t()/n_samples
    return dL_dy_pred
  
class CrossEntropy(Loss):
  """
  A class representing the cross entropy loss. It is assumed that the input to the loss are the logits. A softmax activation is applied to the logits before calculating the loss.
  """
  def __init__(self):
    super().__init__()
  
  def __call__(self, z_pred, y_true):
    """
    When the object is called, it calculates the cross entropy loss on the predicted logits.
    
    Args:
      - z_pred: the predicted logits. Shape (n_samples, n_classes).
      - y_true: the true labels. Shape (n_samples, n_classes).
    """
    q = torch.exp(z_pred - torch.max(z_pred, axis=1, keepdim=True).values)
    q = q/torch.sum(q, dim=1, keepdim=True)
    return - torch.mean(torch.sum(y_true * torch.log(q), dim=1))
  
  def backward(self, z_pred, y_true):
    """
    Backward pass through the cross entropy loss. Returns the differential of the loss with respect to the logits.
    
    Args:
      - z_pred: the predicted logits. Shape (n_samples, n_classes).
      - y_true: the true labels. Shape (n_samples, n_classes).
    """
    q = torch.exp(z_pred - torch.max(z_pred, axis=1, keepdim=True).values)
    q = q/torch.sum(q, dim=1, keepdim=True)
    n_samples = z_pred.shape[0]
    dL_dz_pred = (q - y_true).t()/n_samples
    return dL_dz_pred