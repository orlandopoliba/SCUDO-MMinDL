# /// script
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "torch==2.9.1",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium", app_title="layers module")

with app.setup:
    import numpy
    import torch


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This file serves as a module in the `mydl` package.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The `Layer` class

    We introduce the `Layer` class. This will be a super class that defines the blueprint for all layers. Specific layers will inherit from this class.
    """)
    return


@app.class_definition
class Layer:
    """
    A base class for all layers in the neural network.
    """

    def __init__(self):
        """
        Initialize the Layer.
        """
        self.parameters = {}
        self.n_parameters = 0
        self.gradL_d = {}

    def forward(self,x):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, dL_dy):
        raise NotImplementedError("Backward method not implemented.")

    def __repr__(self):
        raise NotImplementedError("__repr__ method not implemented.")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The `Linear` layer

    We recall that a linear layer is structured in the following way.

    - The input is expected to be a tensor $x \in \mathbb{R}^{N \times M_\text{in}}$. Here, $N$ is the number of samples, and $M_\text{in}$ is the input feature dimension (in code, called `fan_in` following the jargon of logic gates.)
    - The layer is characterized by the following parameters: weights $W \in \mathbb{R}^{M_\text{in} \times M_\text{out}}$ and biases $b \in \mathbb{R}^{1 \times M_\text{out}}$. Here, $M_\text{out}$ is the output feature dimension (in code, called `fan_out`, following the jargon of logic gates.)
    - The weight matrix $W$ is initialized using random values drawn from a Gaussian distribution and the biases are initialized as zero. (**Note**: We are not discussing initialization of parameters at the moment.)
    - The output of the layer is given by $y = x W + b \in \mathbb{R}^{N \times M_\text{out}}$.

    **Comment**: We are using `requires_grad = False`. The power of `torch.tensor` is that is can be initialized with `requires_grad = True`, which allows for automatic differentiation  with the `autograd` module. Here, for pedagogical purposes, we are not leveraging this feature, since we will implement differentiation manually in later sections.

    **Comment**: Pay attention to the [broadcasting rules for tensors](https://docs.pytorch.org/docs/stable/notes/broadcasting.html).
    """)
    return


@app.class_definition
class Linear(Layer):

    """
    A class that represents a linear layer in a neural network.

    Parameters in this layer have the following shapes:

    W: (fan_in, fan_out)
    b: (1, fan_out)
    """

    def __init__(self, fan_in, fan_out):
        """
        Initialize the Linear layer with weights and biases.

        Args:
        - fan_in (int): The number of input features.
        - fan_out (int): The number of output features.
        """
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.parameters['W'] = torch.randn((fan_in, fan_out), dtype=torch.float32, requires_grad=False)
        self.parameters['b'] = torch.zeros((1, fan_out), dtype=torch.float32, requires_grad=False)
        self.n_parameters = fan_in * fan_out + fan_out

    def forward(self,x):
        """
        Forward pass of the Linear layer.

        Args:
        - x (torch.Tensor): Input tensor of shape (N, fan_in).

        Returns:
        - y (torch.Tensor): Output tensor of shape (N, fan_out).
        """
        self.x = x # stores input for backward pass
        self.y = x @ self.parameters['W'] + self.parameters['b']

        return self.y

    def backward(self, dL_dy):
        """
        Backward pass of the Linear layer.

        Stores the gradients of the loss with respect to the parameters into self.gradL_d.

        Shapes:
        - dL_dy: (fan_out, N)
        - dL_dx: (fan_in,  N)
        - gradL_d['W']: (fan_in, fan_out)
        - gradL_d['b']: (1, fan_out)

        Args:
        - dL_dy (torch.Tensor): Differential of the loss with respect to the output, shape (fan_out, N).

        Returns:
        - dL_dx (torch.Tensor): Gradient of the loss with respect to the input, shape (N, fan_in).
        """
        self.gradL_d['W'] = ( dL_dy @ self.x ).t()
        self.gradL_d['b'] = torch.sum(dL_dy, dim=1, keepdim=True).t()
        dL_dx = self.parameters['W'] @ dL_dy 
        return dL_dx

    def __repr__(self):
        """
        String representation of the Linear layer.

        Returns:
        - str: A string describing the Linear layer with its fan_in and fan_out.
        """
        return f"Linear(fan_in={self.fan_in}, fan_out={self.fan_out}) | N. parameters: {self.n_parameters}"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The `Sigmoid` layer

    The `Sigmoid` layer is the first example of a non-linear activation function that we will implement. The sigmoid function is defined as follows:

    $$
    \sigma(z) = \frac{1}{1 + e^{-z}}
    $$

    and this layer has no parameters.
    """)
    return


@app.class_definition
class Sigmoid(Layer):
    """
    A class representing a sigmoid activation layer in a neural network.
    """

    def __init__(self):
        """
        Initialize the Sigmoid layer. It has no parameters.
        """
        super().__init__()

    def forward(self,x):
        """
        Forward pass of the Sigmoid layer.

        Args:
        - x (torch.Tensor): Input tensor of any shape.
        Returns:
        - y (torch.Tensor): Output tensor of the same shape as input, after applying the sigmoid function.
        """
        self.y = 1 / (1 + torch.exp(-x))  # store input for backward pass
        return self.y

    def backward(self, dL_dy):
        """
        Backward pass of the Sigmoid layer. Returns the gradient of the loss with respect to the input.

        Shapes:
        - dL_dy: (fan_out, N)

        Args:
        - dL_dy (torch.Tensor): Differential of the loss with respect to the output.

        Returns:
        - dL_dx (torch.Tensor): Differential of the loss with respect to the input.
        """
        dL_dx = dL_dy * ( self.y * (1 - self.y) ).t()
        return dL_dx

    def __repr__(self):
        """
        String representation of the Sigmoid layer.

        Returns:
        - str: A string describing the Sigmoid layer.
        """
        return f"Sigmoid() | N. parameters: {self.n_parameters}"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The `Tanh` layer

    The `Tanh` layer has the same structure as the `Sigmoid` layer, but applies the hyperbolic tangent function instead. The hyperbolic tangent function is defined as follows:

    $$
    \tanh(t) = \frac{e^t - e^{-t}}{e^t + e^{-t}}
    $$

    and has the advantage of being zero-centered, which can help with optimization.
    """)
    return


@app.class_definition
class Tanh(Layer):
    """
    A class representing a hyperbolic tangent activation layer in a neural network.
    """

    def __init__(self):
        """
        Initialize the Tanh layer. It has no parameters.
        """
        super().__init__()

    def forward(self,x):
        """
        Forward pass of the Tanh layer.

        Args:
        - x (torch.Tensor): Input tensor of any shape.
        Returns:
        - y (torch.Tensor): Output tensor of the same shape as input, after applying the tanh function.
        """
        self.y = torch.tanh(x)  # store input for backward pass
        return self.y

    def backward(self, dL_dy):
        """
        Backward pass of the Tanh layer. Returns the gradient of the loss with respect to the input.

        Shapes:
        - dL_dy: (fan_out, N)

        Args:
        - dL_dy (torch.Tensor): Differential of the loss with respect to the output.

        Returns:
        - dL_dx (torch.Tensor): Differential of the loss with respect to the input.
        """
        dL_dx = dL_dy * ( 1 - self.y ** 2 ).t()
        return dL_dx

    def __repr__(self):
        """
        String representation of the Tanh layer.

        Returns:
        - str: A string describing the Tanh layer.
        """
        return f"Tanh() | N. parameters: {self.n_parameters}"


if __name__ == "__main__":
    app.run()
