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

__generated_with = "0.19.6"
app = marimo.App(
    width="columns",
    app_title="Using MNIST",
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
    # The MNIST dataset

    The **MNIST database** (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets. The creators felt that since NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students, it was not well-suited for machine learning experiments. (Source: [Wikipedia](https://en.wikipedia.org/wiki/MNIST_database))
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We download the dataset from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv). It is given in a simpler format than the original MNIST dataset, a CSV file.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    return pd, plt, torch


@app.cell
def _(pd):
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    data_path = BASE_DIR / 'datasets' / 'mnist_train.csv'
    mnist = pd.read_csv(data_path)
    return (mnist,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us inspect the dataset.
    """)
    return


@app.cell
def _(mnist):
    mnist.info()
    return


@app.cell
def _(mnist):
    mnist
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each row in the dataset has 785 columns. The first column, called `label`, is the digit that was drawn by the user. The rest of the columns are numbers from 0 to 255 that represent the grayscale level of each pixel in the 28x28 image.

    We can reorganize elements in the dataset as a 28x28 matrix.
    """)
    return


@app.cell
def _(mnist, torch):
    torch.tensor(mnist.iloc[0, 1:].values, dtype=torch.float32).view(28,28)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can plot the handwritten digits by coloring the pixels according to their grayscale level.
    """)
    return


@app.cell
def _(mnist, plt, torch):
    def _plot():
        i = torch.randint(0, mnist.shape[0], (1,)).item() # random sample in the dataset
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1) 
        ax.axis('off')
        image = torch.tensor(mnist.iloc[i, 1:].values, dtype=torch.float32).view(28,28)
        ax.imshow(image, cmap='gray_r')

        for y_index in torch.arange(28):
          for x_index in torch.arange(28):
            value = image[y_index, x_index].item()
            label = f"{value:.0f}"
            text_x = x_index
            text_y = y_index
            if value > 256/2:
              adapted_grey = (1,1,1)
            else:
              adapted_grey = (0,0,0)
            ax.text(text_x, text_y, label, color=adapted_grey, ha='center', va='center')
        plt.show()

    _plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us see some examples of labeled handwritten digits.
    """)
    return


@app.cell
def _(mnist, plt, torch):
    def _plot():
        fig = plt.figure(figsize=(10, 10))
        axs = []
        N = 4
        for i in range(N*N):
          axs.append(fig.add_subplot(N,N,i+1)) # we create an axis
        for i in range(N*N):
          axs[i].axis('off') # we turn off the axis
          axs[i].imshow(torch.tensor(mnist.iloc[i,1:].values, dtype=torch.float32).view(28, 28), cmap='gray_r') # we plot the image
          axs[i].set_title(mnist.iloc[i,0]) # we set the title of the image
        plt.show()

    _plot()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
