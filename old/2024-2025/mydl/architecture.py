import torch

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
  
  def backward(self, y_true, loss):
    """
    Implements the backward pass through the network by propagating the loss differential. The backward pass in each layer is called in reverse order and stores the loss gradients with respect to parameters.
    
    Args:
      - y_true: true labels.
      - loss: loss function used to calculate the loss differential.
    """
    y_pred = self.layers[-1].y
    dL_dy = loss.backward(y_pred, y_true)
    for layer in reversed(self.layers):
      dL_dy = layer.backward(dL_dy)
      
  def train(self, x_train, y_train, loss, optimizer, n_epochs, batch_size=None):
    """
    Trains the network on the given data.
    
    Args:
      - x_train: input training data.
      - y_train: output training data.
      - loss: loss function used to calculate the loss differential.
      - optimizer: optimizer used to update the network parameters.
      - n_epochs: number of epochs to train the network.
      - batch_size: Batch size. Default is None, which means that the whole dataset is used as a single batch.
    """
    
    batch_size = batch_size if batch_size else x_train.shape[0]
    
    print('Training the network...')
    losses_train = []
    for epoch in range(n_epochs):
      permuted_indices = torch.randperm(x_train.shape[0])
      batches = torch.split(permuted_indices, batch_size)
      for batch in batches:
        x_batch = x_train[batch]
        y_batch = y_train[batch]
        y_batch_pred = self.forward(x_batch)
        current_loss_train = loss(y_batch_pred, y_batch).item()
        losses_train.append(current_loss_train)
        self.backward(y_batch, loss)
        optimizer.update(self)
    print('Training complete.')
    return losses_train