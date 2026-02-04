# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.19.7",
#     "matplotlib==3.10.8",
#     "pandas==3.0.0",
#     "torch==2.9.1",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(
    width="medium",
    app_title="Binary Logistic Regression",
    css_file="style.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We load the MNIST dataset as we did in the [previous notebook](02-using_MNIST.py). The dataset has already been split into training and test sets, so we load them in separate dataframes.
    """)
    return


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
def _(mnist_test, mnist_train):
    print(f"Number of training examples: {len(mnist_train)}")
    print(f"Number of testing examples: {len(mnist_test)}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
