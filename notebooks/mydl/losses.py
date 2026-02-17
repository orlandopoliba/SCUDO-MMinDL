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
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    import torch


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return


@app.class_definition
class Loss: 

    def __init__(self):
        pass 

    def __call__(self, y, y_train):
        raise NotImplementedError()


@app.class_definition
class MSE(Loss):

    def __init__(self):
        super().__init__()

    def __call__(self, y, y_train):
        return torch.mean((y - y_train) ** 2)

    def backward(self, y, y_train):
        n_samples = y.shape[0]
        dL_dy = (2 / n_samples) * (y - y_train).t()
        return dL_dy


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
