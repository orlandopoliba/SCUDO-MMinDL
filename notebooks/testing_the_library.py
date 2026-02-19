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
app = marimo.App(
    width="medium",
    app_title="Testing the Library",
    css_file="style.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import torch
    return (torch,)


@app.cell
def _():
    from mydl.architecture import Sequential 
    return (Sequential,)


@app.cell
def _():
    from mydl.layers import Linear, Sigmoid
    return Linear, Sigmoid


@app.cell
def _(Linear, Sequential, Sigmoid):
    model = Sequential([
        Linear(5,3),
        Sigmoid(),
        Linear(3,1),
        Sigmoid()
    ])
    return (model,)


@app.cell
def _(torch):
    x_train = torch.randn((100,5))
    y_train = torch.randn((100,1))
    return x_train, y_train


@app.cell
def _():
    from mydl.losses import MSE
    return (MSE,)


@app.cell
def _(MSE):
    loss = MSE()
    return (loss,)


@app.cell
def _():
    from mydl.optimizers import GD
    return (GD,)


@app.cell
def _(GD, model):
    optimizer = GD(model, learning_rate=0.1)
    return (optimizer,)


@app.cell
def _(loss, model, optimizer, x_train, y_train):
    losses = model.train(x_train, y_train, loss, optimizer, n_epochs=100)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
