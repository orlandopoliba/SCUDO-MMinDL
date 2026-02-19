# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.19.10",
#     "pandas==3.0.0",
#     "plotly==6.5.2",
#     "torch==2.9.1",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(
    width="medium",
    app_title="MyDL on MNIST",
    css_file="style.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using the `mydl` library on the MNIST dataset

    Let us train a model built with our library on the MNIST dataset.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import torch
    import plotly.graph_objects as go
    return go, pd, torch


@app.cell
def _(pd):
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    data_path = BASE_DIR / 'datasets' / 'mnist_train.csv'
    mnist_train = pd.read_csv(data_path)
    data_path = BASE_DIR / 'datasets' / 'mnist_test.csv'
    mnist_test = pd.read_csv(data_path)
    return mnist_test, mnist_train


@app.cell
def _(torch):
    def one_hot_encode(y, num_classes):
        n_samples = y.shape[0]
        return torch.eye(num_classes)[y].view(n_samples, num_classes)
    return (one_hot_encode,)


@app.cell
def _(mnist_test, mnist_train, one_hot_encode, torch):
    x_train_raw = torch.tensor(mnist_train.iloc[:, 1:].values, dtype=torch.float32)
    x_test_raw = torch.tensor(mnist_test.iloc[:, 1:].values, dtype=torch.float32)

    x_train = x_train_raw/255
    x_test = x_test_raw/255

    y_train_raw = torch.tensor(mnist_train.iloc[:,0].values, dtype=torch.int64).view(-1,1)
    y_test_raw = torch.tensor(mnist_test.iloc[:,0].values, dtype=torch.int64).view(-1,1)

    y_train = one_hot_encode(y_train_raw, 10)
    y_test = one_hot_encode(y_test_raw, 10)
    return x_test, x_train, y_test, y_train


@app.cell
def _():
    from mydl.architecture import Sequential
    from mydl.layers import Linear, Sigmoid
    from mydl.losses import CrossEntropy
    from mydl.optimizers import SGD
    return CrossEntropy, Linear, SGD, Sequential, Sigmoid


@app.cell
def _(Linear, Sequential, Sigmoid):
    model = Sequential([
        Linear(784, 128),
        Sigmoid(),
        Linear(128, 64),
        Sigmoid(),
        Linear(64, 32),
        Sigmoid(),
        Linear(32, 10)
    ])
    return (model,)


@app.cell
def _(CrossEntropy, SGD, model, x_train, y_train):
    loss = CrossEntropy()

    learning_rate = 1
    n_epochs = 50
    batch_size = 32

    optimizer = SGD(model, learning_rate=learning_rate, batch_size=batch_size)

    losses = model.train(x_train, y_train, loss, optimizer, n_epochs=n_epochs)
    return (losses,)


@app.cell
def _(go, losses):
    def plot_losses():

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(losses) + 1)),
                y=losses,
                mode='lines+markers',
                name='Training Loss'
            )
        )

        fig.update_layout(
            title='Training Loss over Epochs',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white'
        )

        return fig 

    plot_losses()
    return


@app.cell
def _(model, torch, x_test, y_test):
    q = model.forward(x_test)
    label_pred = torch.argmax(q, dim=1)
    label_true = torch.argmax(y_test, dim=1)
    accuracy = (label_pred == label_true).float().mean().item()
    return (accuracy,)


@app.cell
def _(accuracy):
    accuracy
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
