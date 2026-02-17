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
def _(model, x_train):
    y = model.forward(x_train)
    return (y,)


@app.cell
def _():
    from mydl.losses import MSE
    return (MSE,)


@app.cell
def _(MSE):
    loss = MSE()
    return (loss,)


@app.cell
def _(loss, y, y_train):
    loss(y, y_train)
    return


@app.cell
def _(loss, y, y_train):
    dL_dy = loss.backward(y, y_train)
    return (dL_dy,)


@app.cell
def _(dL_dy):
    dL_dy.shape
    return


@app.cell
def _(torch):
    A = torch.randn(10, 4)
    return (A,)


@app.cell
def _(A):
    A
    return


@app.cell
def _(A, torch):
    torch.sum(A, dim=1, keepdim=True)
    return


@app.cell
def _(Linear, torch):
    def _test(): 
        x = torch.randn(100, 5)
        linear = Linear(5, 3)
        y = linear.forward(x)
        dL_dy = torch.randn(3, 100)
        dL_dx = linear.backward(dL_dy)
        print(dL_dx.shape)
    _test()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
