# /// script
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "torch==2.9.1",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(
    width="medium",
    app_title="optimizers module",
    css_file="../style.css",
    auto_download=["html"],
)

with app.setup:
    import torch


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The `Optimizer` class

    The last ingredient we need to train a neural network is a numerical optimization algorithm. For this reason we define the `Optimizer` class. We create a new file named `optimizers.py` inside the `mydl` folder. This file will contain the definition of the `Optimizer` class and its subclasses (at the moment, only `GD`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The base class `Optimizer`

    We create a base class `Optimizer`.
    """)
    return


@app.class_definition
class Optimizer:
    """
    A base class for optimizers in a neural network framework.
    """

    def __init__(self):
        pass 

    def update(self):
        """
        Update the parameters of the model.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def step(self):
        """
        Perform a single update step.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The gradient descent optimizer

    Let us implement a simple `GD` (Gradien descent) optimizer that inherits from the `Optimizer` class.
    """)
    return


@app.class_definition
class GD(Optimizer):
    """
    Gradient Descent optimizer.
    """

    def __init__(self, model, learning_rate=0.01, lambda_reg=0.0):
        """
        Initialize the GD optimizer.

        Args:
        - model: The neural network model to optimize.
        - learning_rate: The learning rate for the optimizer.
        - lambda_reg: Weight for L2 regularization term.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

    def update(self, x_train, y_train, loss):
        """
        Updates the parameters of the model.

        Args:
        - x_train: The input training data.
        - y_train: The target training data.
        - loss: The loss function to compute the loss.
        Returns:
        - epoch_loss: The loss value for the current epoch.
        """
        y = self.model.forward(x_train)
        epoch_loss = loss(y, y_train)
        self.model.backward(loss, y_train)
        self.step()
        return epoch_loss

    def step(self):
        """
        Update the parameters of the model using gradient descent.
        """
        for layer in self.model.layers:
            for key in layer.parameters.keys():
                layer.parameters[key] -= self.learning_rate * ( layer.gradL_d[key] + self.lambda_reg * layer.parameters[key] )


@app.cell
def _(mo):
    mo.md(r"""
    ## The Stochastic Gradient Descent optimizer

    We modify Gradient Descent to implement mini batch sampling.
    """)
    return


@app.class_definition
class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer, implemented with random reshuffling. 
    """

    def __init__(self, model, learning_rate=0.01, batch_size=32, lambda_reg=0.0):
        """
        Initialize the SGD optimizer.

        Args:
        - model: The neural network model to optimize.
        - learning_rate: The learning rate for the optimizer.
        - batch_size: The batch size in the random reshuffling. 
        - lambda_reg: Weight for L2 regularization term.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg

    def update(self, x_train, y_train, loss):
        """
        Updates the parameters of the model.

        Args:
        - x_train: The input training data.
        - y_train: The target training data.
        - loss: The loss function to compute the loss.
        Returns:
        - epoch_loss: The loss value for the current epoch.
        """

        n_training_samples = x_train.shape[0]
        shuffled_indices = torch.randperm(n_training_samples)

        avg_epoch_loss = 0.0

        for start in range(0,n_training_samples,self.batch_size):

            batch_indices = shuffled_indices[start: start + self.batch_size]
            xb_train = x_train[batch_indices]
            yb_train = y_train[batch_indices]

            yb = self.model.forward(xb_train)
            batch_loss = loss(yb, yb_train)
            avg_epoch_loss += batch_loss

            self.model.backward(loss, yb_train)
            self.step()

        avg_epoch_loss = avg_epoch_loss / ( n_training_samples // self.batch_size )

        return avg_epoch_loss

    def step(self):
        """
        Update the parameters of the model using gradient descent.
        """
        for layer in self.model.layers:
            for key in layer.parameters.keys():
                layer.parameters[key] -= self.learning_rate * ( layer.gradL_d[key] + self.lambda_reg*layer.parameters[key] )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
