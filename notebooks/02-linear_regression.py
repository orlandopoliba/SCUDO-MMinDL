# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.19.7",
#     "pandas==3.0.0",
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


@app.cell(column=0, hide_code=True)
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
    A = \frac{1}{N} \tilde x^\top \tilde x \in \mathbb{R}^{M \times M} , \quad c = \frac{1}{N} \tilde x^\top y , \quad \tilde x = \begin{pmatrix} x & 1 \end{pmatrix} \in \mathbb{R}^{N \times (M+1)} .
    $$
    """)
    return


if __name__ == "__main__":
    app.run()
