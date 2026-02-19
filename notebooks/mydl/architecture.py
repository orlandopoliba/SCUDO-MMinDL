# /// script
# dependencies = [
#     "marimo",
#     "plotly==6.5.2",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium", app_title="architecture module")

with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This file serves as a module in the `mydl` package.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # A note on `marimo` notebooks

    Here we are using `marimo` notebooks for pedagocial reasons. They are nice because they are `.py` files. Functions and classes that are defined in a `marimo` notebook can be imported in other `.py` files, so we can treat this as a normal Python module. Typically, one should just write `.py` files for modules.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # The `Sequential` class

    We introduce here the `Sequential` class which represents a container for stacking multiple layers in a neural network.
    """)
    return


@app.class_definition
class Sequential:
    """
    A class representing a sequential container for stacking layers in a neural network.
    """

    def __init__(self, layers):
        """
        Initializes the Sequential container with a list of layers.

        Parameters:
        layers (list): A list of layer instances to be stacked sequentially.
        """
        self.layers = layers
        self.n_parameters = 0
        for i, layer in enumerate(layers):
            self.n_parameters += layer.n_parameters

    def forward(self,x):
        """
        Forward pass for the sequential neural network.

        Args:
        - x (torch.tensor): The initial input tensor.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss, y_train):
        """
        Backward pass for the sequential neural network.

        Args:
        - loss (mydl.losses.Loss): The loss instance to compute gradients from.
        - y_train (torch.tensor): The target training data.
        """

        y = self.layers[-1].y 
        dL_dy = loss.backward(y, y_train)
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)

    def train(self, x_train, y_train, loss, optimizer, n_epochs=1):
        """
        Method to train the sequential neural network on the given training data.

        Args:
        - x_train (torch.tensor): The input training data.
        - y_train (torch.tensor): The target training data.
        - loss (mydl.losses.Loss): The loss instance to compute the loss.
        - optimizer (mydl.optimizers.Optimizer): The optimizer instance to update the parameters.
        - n_epochs (int): The number of epochs to train the model.

        Returns:
        - losses_history (list): A list containing the loss value at each epoch.
        """
        losses_history = []
        for epoch in mo.status.progress_bar(
            range(n_epochs),
            title="Training Progress",
            completion_title="Training Complete",
            show_eta=True,
            show_rate=True,
            remove_on_exit=False
        ):
            epoch_loss = optimizer.update(x_train, y_train, loss)
            losses_history.append(epoch_loss.item())
        return losses_history

    def __repr__(self):
        """
        Returns a string representation of the Sequential container and its layers.
        """
        string = ""
        for layer in self.layers:
            string += repr(layer) + "\n"
        string += f"Total number of parameters: {self.n_parameters}\n"
        return string


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
