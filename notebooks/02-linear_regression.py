# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.19.6",
#     "matplotlib==3.10.8",
#     "pandas==3.0.0",
#     "torch==2.9.1",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(
    width="columns",
    app_title="Linear Regression",
    css_file="style.css",
    auto_download=["html"],
)


@app.cell(column=0)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear regression

    We recall one of the most basic models in machine learning: linear regression. We apply it to the problem of predicting the price of a house.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset

    We use the dataset available on [Kaggle](https://www.kaggle.com/datasets/kirbysasuke/house-price-prediction-simplified-for-regression). Its format is a CSV file. We load it as a `DataFrame` using the `pandas` library and we explore it.
    """)
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    data_path = BASE_DIR / 'datasets' / 'Real_Estate.csv'
    data = pd.read_csv(data_path)
    return (data,)


@app.cell
def _(data):
    data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have a dataset with 414 observations and 7 columns. We will do a simple linear regression using the following features:
    - `House age` -> $X_0$
    - `Distance to the nearest MRT station` -> $X_1$
    - `Number of convenience stores` -> $X_2$
    - `Latitude` -> $X_3$
    - `Longitude` -> $X_4$

    trying to predict the
    - `House price of unit area` -> $Y$
    """)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recalling the linear regression model

    We want to predict the `House price of unit area` based on the other features with a linear model, _i.e._, a prediction based on the ansatz

    $$
    Y \approx w_0 X_0 + w_1 X_1 + w_2 X_2 + w_3 X_3 + w_4 X_4 + b.
    $$

    The objective is to find "good" values for the weights $w_0, w_1, w_2, w_3, w_4$ and the bias $b$, exploiting the available data.

    The linear regression model can be written compactly as

    $$
    f(X;w,b) = Xw + b,
    $$

    where $X = (X_0, X_1, \dots, X_{M-1}) \in \mathbb{R}^{1 \times M}$ is the random feature vector, $w \in \mathbb{R}^{M\times 1}$ is the weights vector, and $b \in \mathbb{R}$ is the bias. The objective is to find the weights $w$ and the bias $b$ that minimize the mean squared error (MSE) between the predictions and the real target, _i.e._,

    $$
    \min_{\substack{w \in \mathbb{R}^{M \times 1} \\ \ b \in \mathbb{R}}} L(w,b;\{(x_i,y_i)\}_{i=0,\dots,N-1})\, , \  \quad \text{where } L(w,b;\{(x_i,y_i)\}_{i=0,\dots,N-1}) = \frac{1}{N} \sum_{i=0}^{N-1} (x_i w + b - y_i)^2.
    $$

    The minimum can be found explicitly by solving the _normal equation_

    $$
    A \begin{pmatrix} w \\ b \end{pmatrix} = c ,
    $$

    where

    $$
    A = \frac{1}{N} \tilde x^\top \tilde x \in \mathbb{R}^{(M+1) \times (M+1)} , \quad c = \frac{1}{N} \tilde x^\top y , \quad \tilde x = \begin{pmatrix} x & 1 \end{pmatrix} \in \mathbb{R}^{N \times (M+1)} .
    $$
    """)
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    import torch
    return (torch,)


@app.cell
def _(data):
    features = ['House age', 'Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
    data[features] 
    return (features,)


@app.cell
def _(data):
    n_training_samples = int(0.8 * len(data))
    return (n_training_samples,)


@app.cell
def _(n_training_samples):
    n_training_samples
    return


@app.cell
def _(data, n_training_samples, torch):
    torch.manual_seed(42)
    permuted_indices = torch.randperm(len(data))
    training_indices = permuted_indices[:n_training_samples]
    test_indices = permuted_indices[n_training_samples:]
    return test_indices, training_indices


@app.cell
def _(
    data,
    features,
    n_training_samples,
    test_indices,
    torch,
    training_indices,
):
    x_train = torch.tensor(data[features].iloc[training_indices].values, dtype=torch.float32)
    x_test = torch.tensor(data[features].iloc[test_indices].values, dtype=torch.float32)
    y_train = torch.tensor(data['House price of unit area'].iloc[training_indices].values, dtype=torch.float32).view(n_training_samples, 1)
    y_test = torch.tensor(data['House price of unit area'].iloc[test_indices].values, dtype=torch.float32).view(-1, 1)
    return x_test, x_train, y_test, y_train


@app.cell
def _(x_train):
    x_train.shape
    return


@app.cell
def _(y_train):
    y_train.shape
    return


@app.cell(column=3)
def _(n_training_samples, torch, x_train):
    x_tilde = torch.cat((x_train, torch.ones(n_training_samples,1)), dim=1)
    return (x_tilde,)


@app.cell
def _(x_tilde):
    x_tilde.t()
    return


@app.cell
def _(n_training_samples, x_tilde):
    A = 1 / n_training_samples * x_tilde.t() @ x_tilde
    return (A,)


@app.cell
def _(A):
    A.shape
    return


@app.cell
def _(n_training_samples, x_tilde, y_train):
    c = 1 / n_training_samples * x_tilde.t() @ y_train
    return (c,)


@app.cell
def _(c):
    c.shape
    return


@app.cell
def _(A, c, torch):
    solution = torch.linalg.solve(A, c)
    return (solution,)


@app.cell
def _(solution):
    w = solution[:5]
    b = solution[5]
    return b, w


@app.cell
def _(w):
    w.shape
    return


@app.cell
def _(b):
    b
    return


@app.cell
def _():
    return


@app.cell(column=4)
def _(b, w, x_train, y_train):
    R2_train = 1 - ( ( y_train - ( x_train @ w + b ) )**2 ).sum() / ( ( y_train - y_train.mean() )**2 ).sum()
    print(R2_train)
    return


@app.cell
def _(b, w, x_test, y_test):
    R2_test = 1 - ( ( y_test - ( x_test @ w + b ) )**2 ).sum() / ( ( y_test - y_test.mean() )**2 ).sum()
    print(R2_test)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
