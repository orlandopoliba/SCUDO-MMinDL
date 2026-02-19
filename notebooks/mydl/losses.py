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
    app_title="losses module",
    auto_download=["html"],
)

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
    # The `Loss` class

    The `Loss` class is a base class for defining loss functions. Its subclasses will be specific losses.
    """)
    return


@app.class_definition
class Loss:
    """
    A base class for defining loss functions.
    """

    def __init__(self):
        pass

    def __call__(self, y, y_train):
        """
        Compute the loss between true and predicted values.

        Args:
        - y: The predicted values.
        - y_train: The true values.

        Returns:
        The computed loss.
        """
        raise NotImplementedError("Subclasses must implement this method.")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The `MSE` loss

    This is the first loss we introduces, the Mean Squared Error. It is used, for example, in linear regression.
    """)
    return


@app.class_definition
class MSE(Loss):
    """
    A class for Mean Squared Error loss.
    """

    def __init__(self):
        """
        Initialize the MSE loss with true output values from the dataset.
        """
        super().__init__()

    def __call__(self, y, y_true):
        """
        Compute the Mean Squared Error loss between true and predicted values.

        Args:
        - y (torch.tensor): The predicted values.
        - y_true (torch.tensor): The true values from the dataset.

        Returns:
        - The computed Mean Squared Error loss.
        """
        return torch.mean((y - y_true) ** 2)

    def backward(self, y, y_true):
        """
        Backpropagate the gradient of the Mean Squared Error loss. Returns the differential of the loss with respect to y.

        Shape of the output: (1, n_samples)

        Args:
        - y (torch.tensor): The predicted values.
        - y_true (torch.tensor): The true values from the dataset.

        Returns:
        - The gradient of the loss with respect to y.
        """
        n_samples = y_true.shape[0]
        dL_dy = 2 * (y - y_true).t() / n_samples
        return dL_dy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The `CrossEntropy` loss

    This is the loss used in classification. Notice that it is implemented in such a way that a softmax layer is applied. This means that the input should be a logit, and not a probability.
    """)
    return


@app.class_definition
class CrossEntropy(Loss):
    """
    A class for Cross Entropy loss.
    """

    def __init__(self):
        """
        Initializes the Cross Entropy loss.
        """
        super().__init__()

    def __call__(self, x, y_true):
        """
        Compute the Cross Entropy loss between true and predicted values.

        Args:
        - x (torch.tensor): The predicted logits.
        - y_true (torch.tensor): The true values from the dataset in one-hot encoded format.
        """

        q = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return -torch.mean( torch.sum( y_true * torch.log(q), dim=1 ) )

    def backward(self, x, y_true):
        """
        Backpropagate the gradient of the Cross Entropy loss. Returns the differential of the loss with respect to x.

        Shape of the output: (n_classes, n_samples)

        Args:
        - x (torch.tensor): The predicted logits.
        - y_true (torch.tensor): The true values from the dataset in one-hot encoded format.

        Returns:
        - The gradient of the loss with respect to x.
        """
        q = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
        q = q / torch.sum(q, dim=1, keepdim=True)
        n_samples = y_true.shape[0]
        dL_dx = (q - y_true).t() / n_samples
        return dL_dx


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
