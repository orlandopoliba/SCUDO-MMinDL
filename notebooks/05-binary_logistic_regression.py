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
    width="columns",
    app_title="Binary Logistic Regression",
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
    # Logistic regression (binary classification) with MNIST

    Let us start with a simple example. We will use logistic regression to recognize if a digit is "1" or "0".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preparing the dataset

    We start by importing libraries.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    return pd, plt, torch


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us count the number of samples in each set.
    """)
    return


@app.cell
def _(mnist_test, mnist_train):
    print(f"Number of training examples: {len(mnist_train)}")
    print(f"Number of testing examples: {len(mnist_test)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us recall the structure of the dataset. We have 784 features (28x28 pixels) and 1 label column.
    """)
    return


@app.cell
def _(mnist_train):
    mnist_train
    return


@app.cell
def _(mnist_test, mnist_train):
    one_zeros_train = mnist_train[mnist_train.label.isin([0,1])]
    one_zeros_test = mnist_test[mnist_test.label.isin([0,1])]
    return one_zeros_test, one_zeros_train


@app.cell
def _(one_zeros_train):
    one_zeros_train.shape
    return


@app.cell
def _(one_zeros_test):
    one_zeros_test.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We create tensors from the dataframes to do operations with them. We call these tensors "raw" because we will preprocess them later.
    """)
    return


@app.cell
def _(one_zeros_test, one_zeros_train, torch):
    x_train_raw = torch.tensor(one_zeros_train.iloc[:, 1:].values, dtype=torch.float32)
    x_test_raw = torch.tensor(one_zeros_test.iloc[:, 1:].values, dtype=torch.float32)
    return x_test_raw, x_train_raw


@app.cell
def _(x_test_raw, x_train_raw):
    print(f"Shape of x_train: {x_train_raw.shape}")
    print(f"Shape of x_test: {x_test_raw.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A good practice is to normalize the data. We divide the pixel values by 255 to get values between 0 and 1. (Remember: the original pixel values are between 0 and 255.)
    """)
    return


@app.cell
def _(x_test_raw, x_train_raw):
    x_train = x_train_raw / 255 
    x_test = x_test_raw / 255
    return x_test, x_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us now define the target tensors.
    """)
    return


@app.cell
def _(one_zeros_test, one_zeros_train, torch):
    y_train = torch.tensor(one_zeros_train.iloc[:,0].values, dtype=torch.int64).view(-1,1)
    y_test = torch.tensor(one_zeros_test.iloc[:,0].values, dtype=torch.int64).view(-1,1)
    return y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We define some variables that help to read the code.
    """)
    return


@app.cell
def _(x_test, x_train):
    n_training_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    n_features = x_train.shape[1]
    print(f"Number of features: {n_features}")
    print(f"Number of training samples: {n_training_samples}")
    print(f"Number of test samples: {n_test_samples}")
    return n_features, n_test_samples


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Binary logistic regression

    We recall how binary logistic regression works. We have some data $\{(x_i, y_i)\}_{i=0, \ldots, N-1}$ where $x_i \in \mathbb{R}^{1\times M}$ and $y_i \in \{0, 1\}$. The data are thought as realizations of a random sample $(X_0, Y_0), \ldots, (X_{N-1}, Y_{N-1})$ drawn from a certain distribution. We make the *ansatz* that the conditional probability of $Y$ given $X$ is modeled by the logistic function

    $$
    \mathbb{P}(Y=1|X=x) \approx \sigma(x w + b)
    $$

    where $\sigma(t) = 1/(1 + \exp(-t))$ is the logistic function and $w \in \mathbb{R}^{M \times 1}$ and $b \in \mathbb{R}$ are the parameters of the model. We want to find the parameters $w$ and $b$ that minimize the realization of the cross-entropy on the sample (see notes)

    $$
    L(w, b; \{(x_i,y_i)\}_{i=0,\ldots,N-1}) = -\frac{1}{N}\sum_{i=0}^{N-1} \Big( y_i \log(\sigma(x_i w + b)) + (1-y_i) \log(1-\sigma(x_i w + b)) \Big).
    $$
    """)
    return


@app.cell
def _(torch):
    def sigma(t):
      return 1 / (1 + torch.exp(-t))
    return (sigma,)


@app.cell
def _(plt, sigma, torch):
    def _plot():

        t = torch.linspace(-10, 10, 100)
        plt.plot(t.numpy(), sigma(t).numpy())
        plt.title('Logistic function')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\sigma(t)$')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.show()

    _plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us see an example of prediction for a single image done by the model. The prediction is done by applying the logistic function to the dot product of the weights and the image plus the bias.
    """)
    return


@app.cell
def _(n_features, sigma, torch, x_train):
    def _cell():

        x = x_train[0] # pick the first image in the training set
        w = torch.randn(n_features, 1)
        b = torch.randn(1)
        q = sigma(x @ w + b)
        print(q)

    _cell()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Of course, there is no meaning in this prediction because the weights and bias are chosen randomly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We compute the model probability on the whole training set.
    """)
    return


@app.cell
def _(n_features, sigma, torch, x_train):
    def _cell():

        w = torch.randn(n_features, 1)
        b = torch.randn(1)
        q = sigma(x_train @ w + b)
        print(f"Shape of q: {q.shape}")

    _cell()
    return


@app.cell
def _(n_features, sigma, torch, x_train, y_train):
    def _cell():

        w = torch.randn(n_features, 1)
        b = torch.randn(1)
        q = sigma(x_train @ w + b)
        print(-torch.mean(y_train * torch.log(q) + (1 - y_train) * torch.log(1 - q)))

    _cell()
    return


@app.cell
def _(sigma, torch):
    def loss(w,b,x,y):

        y_train_1_bool = (y == 1)
        y_train_0_bool = ~y_train_1_bool
        q = sigma(x @ w + b)
        q_not_0_bool = (q != 0)
        q_not_1_bool = (q != 1)
        return -torch.sum(torch.log(q)[q_not_0_bool*y_train_1_bool]) - torch.sum(torch.log(1 - q)[q_not_1_bool*y_train_0_bool])
    return (loss,)


@app.cell
def _(loss, n_features, torch, x_train, y_train):
    def _cell():

        w = torch.randn((n_features, 1), dtype=torch.float32)
        b = torch.tensor(0,dtype=torch.float32)
        print(loss(w,b,x_train,y_train))

    _cell()
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training

    Finding the minimum of the loss is not feasible explicitly. We need to use an optimization algorithm. In the next lectures we will see how to do this. For now, think of the algorithm as a black box.
    """)
    return


@app.cell
def _(sigma, torch, y_train):
    # ignore this cell, we will understand this later

    def gradient_loss(w,b,x,y):
      q = sigma(x @ w + b)
      # Compute gradients
      gradL_dw = x.t() @ (q-y)
      gradL_db = torch.sum(q - y_train)

      # Norm clipping gradients
      norm = torch.sqrt(torch.sum(gradL_dw**2 + gradL_db**2))
      scaling_factor = max(1, norm)
      gradL_dw = gradL_dw / scaling_factor
      gradL_db = gradL_db / scaling_factor 

      return gradL_dw, gradL_db

    def update(w,b,x,y,learning_rate):
      gradL_dw, gradL_db = gradient_loss(w,b,x,y)
      w = w - learning_rate * gradL_dw
      b = b - learning_rate * gradL_db
      return w, b
    return (update,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the next cell the optimization algorithm is implemented. We will understand what it does in the next lectures. For now, just notice that:
    - The algorithm is iterative. It runs for a certain number of iterations: `n_epochs`.
    - The algorithm starts from an initial guess of the weights $w$ (random) and bias $b$ (0): `w = torch.randn((n_features, 1), dtype=torch.float32)` and `b = torch.tensor(0,dtype=torch.float32)`.
    - In each epoch, we update the values of $w$ and $b$.
    - Every 100 epochs, we keep track of the loss and print it. Later, we will plot it.
    """)
    return


@app.cell
def _(loss, n_features, plt, torch, update, x_train, y_train):
    learning_rate = 0.1 
    n_epochs = 1000
    torch.manual_seed(42)
    w = torch.randn((n_features, 1), dtype=torch.float32)
    b = torch.tensor(0,dtype=torch.float32)
    losses = []
    for epoch in range(n_epochs):
        w, b = update(w,b,x_train,y_train,learning_rate)
        losses.append(loss(w,b,x_train,y_train).item())

    def _plot():
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1 = plt.plot(range(0, n_epochs), losses)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2 = plt.plot(range(0, n_epochs)[1:], losses[1:])
        plt.show()

    _plot()
    return b, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We clearly see that the loss decreases with the number of epochs. This is a good sign. It means that the model is learning and we are getting better values for the weights and bias. After this training phase, we have good values for the weights and bias and we can use them to make predictions on the test set.
    """)
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predictions

    To do a prediction, we decide as follows: If the model probability is greater than $50\%$, we predict that the digit is $0$. Otherwise, we predict that the digit is not $0$.
    """)
    return


@app.cell
def _(b, sigma, w, x_test):
    q = sigma(x_test @ w + b)
    y_pred = (q >= 0.5).float()
    return (y_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us count the percentage of correct predictions.
    """)
    return


@app.cell
def _(n_test_samples, torch, y_pred, y_test):
    p_correct_predictions = (torch.sum(y_test == y_pred)/ n_test_samples).item()
    print(f'Correct predictions: {p_correct_predictions*100:.4f}%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us plot some examples of correct and incorrect predictions.
    """)
    return


@app.cell
def _(x_test, y_pred, y_test):
    correct_predictions = (y_test == y_pred).view(-1)
    wrong_predictions = (y_test != y_pred).view(-1)

    x_test_correct = x_test[correct_predictions, :]
    y_test_correct = y_test[correct_predictions, :]
    y_pred_correct = y_pred[correct_predictions, :]

    x_test_wrong = x_test[wrong_predictions, :]
    y_test_wrong = y_test[wrong_predictions, :]
    y_pred_wrong = y_pred[wrong_predictions, :]
    return (
        x_test_correct,
        x_test_wrong,
        y_pred_correct,
        y_pred_wrong,
        y_test_correct,
        y_test_wrong,
    )


@app.cell
def _(plt, x_test_correct, y_pred_correct, y_test_correct):
    def _test():

        # plot of the correct predictions
        n_example = 20 
        fig = plt.figure(figsize=(n_example, 1))
        axs = []
        offset = 20
        for i in range(n_example):
          axs.append(fig.add_subplot(1, n_example, i+1))
          axs[i].imshow(x_test_correct[offset+i,:].view(28,28), cmap='gray_r')
          axs[i].set_title(f'T{y_test_correct[offset+i,0].item():.0f} P{y_pred_correct[offset+i,0].item():.0f}')
          axs[i].axis('off')

        return fig

    _test()
    return


@app.cell
def _(plt, x_test_wrong, y_pred_wrong, y_test_wrong):
    def _test():

        # plot of the wrong predictions
        n_example = x_test_wrong.shape[0]
        fig = plt.figure(figsize=(n_example, 1))
        axs = []
        for i in range(n_example):
          axs.append(fig.add_subplot(1, n_example, i+1))
          axs[i].imshow(x_test_wrong[i,:].view(28,28), cmap='gray_r')
          axs[i].set_title(f'T{y_test_wrong[i,0].item():.0f} P{y_pred_wrong[i,0].item():.0f}')
          axs[i].axis('off')

        return fig

    _test()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=4)
def _(mo):
    mo.md(r"""
    ## Weights

    We can inspect how weights look like.
    """)
    return


@app.cell
def _(plt, w):
    def _plot():

        fig = plt.figure()
        ax = fig.add_subplot(111)
        heatmap = ax.imshow(w.view((28,28)), cmap='vanimo')
        plt.colorbar(heatmap)
    
        return fig

    _plot()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
