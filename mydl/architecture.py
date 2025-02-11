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
      
  def train(self, x_train, y_train, loss, optimizer, n_epochs):
    """
    Trains the network on the given data.
    
    Args:
      - x_train: input training data.
      - y_train: output training data.
      - loss: loss function used to calculate the loss differential.
      - optimizer: optimizer used to update the network parameters.
      - n_epochs: number of epochs to train the
    """
    print('Training the network...')
    losses_train = []
    for epoch in range(n_epochs):
      y_pred = self.forward(x_train)
      current_loss = loss(y_pred, y_train)
      losses_train.append(current_loss)
      self.backward(y_train, loss)
      optimizer.update(self)
    print('Training complete.')
    return losses_train