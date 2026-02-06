# /// script
# requires-python = ">=3.14"
# dependencies = [
#    "marimo>=0.19.7",
#    "matplotlib==3.10.8",
#    "pandas==3.0.0",
#    "torch==2.9.1",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(
    width="columns",
    app_title="Multiclass Logistic Regression",
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
    # Logistic regression (multiclass classification) with MNIST

    We use now multiclass logistic regression to recognize the digit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preparing the dataset

    The first part of this notebook is the same as the one to prepare the data in the [previous notebook](03-binary_logistic_regression.py). Look that for more details.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    return pd, plt, torch


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
def _(mnist_train):
    mnist_train
    return


@app.cell
def _(mnist_test):
    mnist_test
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before defining our target, we modify the target tensors in a convenient way. This will be also useful for the next notebook on multi-class classification. The original tensors contain the labels indicating the digit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The labels are just words that we use to identify the digits and they only represent the class where the items in the dataset belong to. The numerical values of the labels are not relevant for the classification task. A way to represent classes numerically is to use *one-hot encoding*.

    One-hot encoding works as follows. Assume you have $K$ classes. We can enumerate the classes $\{0,1,2,\dots, K-1\}$. We represent each class as a vector in a $K$ dimensional space. More precisely, each class is a coordinate direction in the $K$ dimensional space. This is implemented as follows: We define the bijection

    $$
    k \in \{0,1,2,\dots, K-1\} \mapsto \begin{pmatrix} 0 \\ \vdots \\ 0 \\ 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix} \in \mathbb{R}^K,
    $$

    where the $1$ is in the $k$-th position (counting from $0$ to $K-1$).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us implement this in Python. We use a matrix trick to do this.

    The `torch.eye` function creates an identity matrix.
    """)
    return


@app.cell
def _(torch):
    torch.eye(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The one-hot encoding of $k$ is obtained by simply extracting the $k$-th column (or row) of the identity matrix.
    """)
    return


@app.cell
def _(torch):
    torch.eye(10)[2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We reshape the tensor so that it is a column vector. The function `one_hot_encode` also works on many labels at once.
    """)
    return


@app.cell
def _(torch):
    def one_hot_encode(y, num_classes):
        n_samples = y.shape[0]
        return torch.eye(num_classes)[y].view(n_samples, num_classes)
    return (one_hot_encode,)


@app.cell
def _(one_hot_encode, y_test_raw, y_train_raw):
    y_train_one_hot = one_hot_encode(y_train_raw, 10)
    y_test_one_hot = one_hot_encode(y_test_raw, 10)
    return (y_train_one_hot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us see some examples.
    """)
    return


@app.cell
def _(y_train_one_hot, y_train_raw):
    for _i in range(7):
      print(f"Label: {y_train_raw[_i,:].item()} | One-hot encoding: {y_train_one_hot[_i,:]}")
    return


@app.cell
def _(mnist_test, mnist_train, one_hot_encode, torch):
    x_train_raw = torch.tensor(mnist_train.iloc[:, 1:].values, dtype=torch.float32)
    x_test_raw = torch.tensor(mnist_test.iloc[:, 1:].values, dtype=torch.float32)

    x_train = x_train_raw / 255 
    x_test = x_test_raw / 255

    y_train_raw = torch.tensor(mnist_train.iloc[:,0].values, dtype=torch.int64).view(-1,1)
    y_test_raw = torch.tensor(mnist_test.iloc[:,0].values, dtype=torch.int64).view(-1,1)

    y_train = one_hot_encode(y_train_raw, 10)
    y_test = one_hot_encode(y_test_raw, 10)
    return x_test, x_train, y_test, y_test_raw, y_train, y_train_raw


@app.cell
def _(x_test, x_train, y_train):
    n_training_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]
    n_classes, n_features, n_training_samples, n_test_samples
    return n_classes, n_features, n_test_samples


@app.cell
def _(plt, x_train, y_train):
    def _plot():

        # sample plot of the data
        n_sample = 10
        fig = plt.figure(figsize=(n_sample, 3))
        axs = []
        for i in range(n_sample):
          axs.append(fig.add_subplot(1, n_sample, i+1))
          axs[i].imshow(x_train[i,:].view(28,28), cmap='gray_r')
          axs[i].set_title(f'{(y_train[i,:]==1).nonzero().item():.0f} ')
          axs[i].axis('off')
        plt.show()

    _plot()
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiclass logistic regression

    We recall how multiclass logistic regression works. We have some data $\{(x_i, y_i)\}_{i=0, \ldots, N-1}$ where $x_i \in \mathbb{R}^{1\times M}$ and $y_i \in \{0, 1, \ldots, K-1\}$. The data are thought as realizations of a random sample $(X_0, Y_0), \ldots, (X_{N-1}, Y_{N-1})$ drawn from a certain distribution. We make the *ansatz* that the conditional probability of $Y$ given $X$ is modeled by the softmax function

    $$
    \mathbb{P}(Y=k|X=x) \approx \sigma_k(x W_{\cdot k} + b_k)
    $$

    where $\sigma = (\sigma_0, \ldots, \sigma_{K-1}) \colon \mathbb{R}^{1\times K} \to \mathbb{R}^{1 \times K}$ is the softmax function defined as

    $$
    \sigma_k(t) = \frac{e^{t_k}}{\sum_{h=0}^{K-1} e^{t_h}} \quad \text{for } t = (t_0, \ldots, t_{K-1}) \in \mathbb{R}^{1\times K} \, , \quad k = 0, \ldots, K-1 \, .
    $$

    $W_{\cdot k}$ is the $k$-th column of $W \in \mathbb{R}^{M \times K}$, and $b_k$ is the $k$-th element of $b \in \mathbb{R}^{1 \times K}$. We want to find the parameters $W$ and $b$ that minimize the cross-entropy loss function

    $$
    L(W, b; \{(x_i,y_i)\}_{i=0,\ldots,N-1}) = -\sum_{i=0}^{N-1} \sum_{k=0}^{K-1} \tilde y_{ik} \log \sigma_k(x_{i\cdot} W_{\cdot k} + b_k) \, ,
    $$

    where $\tilde y_{ik} = 1$ if $y_i = k$ and $\tilde y_{ik} = 0$ otherwise (one-hot encoding).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Softmax function

    Let us implement the softmax function.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - With `torch.exp` we can compute the exponential of a tensor elementwise.
    - With `torch.sum` we can compute the sum of all the elements of a tensor, or along a specific dimension.
    """)
    return


@app.cell
def _(torch):
    def _test():

        t = torch.tensor([1., 4., 2.])
        print(torch.exp(t))
        print(torch.sum(torch.exp(t)))
        print(torch.exp(t) / torch.sum(torch.exp(t)))

    _test()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we have a tensor with more rows, we want to apply the softmax function to each row.
    """)
    return


@app.cell
def _(torch):
    def _test():

        t = torch.tensor([[1., 4., 2.],
                          [3., 0., 5.]])
        print(torch.exp(t))
        print(torch.sum(torch.exp(t), dim=1, keepdim=True))
        print(torch.exp(t) / torch.sum(torch.exp(t), dim=1, keepdim=True))

    _test()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hence, we implement the softmax function as follows.
    """)
    return


@app.cell
def _(torch):
    def softmax(t):
      return torch.exp(t) / torch.sum(torch.exp(t), dim=1, keepdim=True)
    return (softmax,)


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## The model

    Let us give an example of evaluation of the model.
    """)
    return


@app.cell
def _(softmax, torch):
    def loss(W,b,x,y):

        q = softmax(x @ W + b)

        return - torch.sum(y * torch.log(q)) 
    return (loss,)


@app.cell
def _(loss, n_classes, n_features, softmax, torch, x_train, y_train):
    def _ex():

        W = torch.randn((n_features, n_classes))
        b = torch.randn((1,n_classes))
        q = softmax(x_train @ W + b)

        print(q.shape)

        print(f"Loss: {loss(W,b,x_train,y_train)}")

    _ex()
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training

    Finding the minimum of the loss is not feasible explicitly. We need to use an optimization algorithm. In the next lectures we will see how to do this. For now, think of the algorithm as a black box.
    """)
    return


@app.cell
def _(softmax, torch, y_train):
    def gradient_loss(W,b,x,y):

      q = softmax(x @ W + b)

      # Compute gradients
      gradL_dW = x.t() @ (q - y)
      gradL_db = torch.sum(q - y_train, dim=0, keepdim=True)

      # Norm clipping
      norm = torch.sqrt(torch.sum(gradL_dW**2 + gradL_db**2))
      scaling_factor = max(1, norm)
      gradL_dW = gradL_dW / scaling_factor
      gradL_db = gradL_db / scaling_factor

      return gradL_dW, gradL_db

    def update(W,b,x,y,learning_rate):

      gradL_dW, gradL_db = gradient_loss(W,b,x,y)
      W = W - learning_rate * gradL_dW
      b = b - learning_rate * gradL_db

      return W, b
    return (update,)


@app.cell(hide_code=True)
def _():
    import time
    return


@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="Train")
    train_btn
    return (train_btn,)


@app.cell
def _(
    loss,
    mo,
    n_classes,
    n_features,
    torch,
    train_btn,
    update,
    x_train,
    y_train,
):
    mo.stop(not train_btn.value, mo.md("Press the `Train` button to run the training."))

    learning_rate = 0.1
    epochs = 1_000 # estimated time: 30s | accuracy: 90%
    #epochs = 5_000 # estimated time: 2m 20s | accuracy: 92%
    #epochs = 10_000 # estimated time: 4m 40s | accuracy: 92.14%
    torch.manual_seed(42)
    W = torch.randn((n_features, n_classes))
    b = torch.tensor(0)
    losses = []
    print('Initial loss:', loss(W,b,x_train,y_train).item())

    for epoch in mo.status.progress_bar(range(epochs), title="Training in progress", completion_title="Training completed", completion_subtitle="", subtitle="Please wait...", show_eta=True, show_rate=True):
      if epoch % 100 == 0:
        losses.append(loss(W,b,x_train,y_train).item())
        print('Epoch:', epoch, 'Loss:', losses[-1])
      W, b = update(W, b, x_train, y_train, learning_rate)
    print('Final loss:', loss(W,b,x_train,y_train).item())
    return W, b, epochs, losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us plot how the loss behaves with the number of epochs.
    """)
    return


@app.cell
def _(epochs, losses, plt):
    def _plot():

        # plot of the loss
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1 = plt.plot(range(0, epochs, 100), losses)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2 = plt.plot(range(0, epochs, 100)[2:], losses[2:])
        plt.show()

    _plot()
    return


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inference

    We can now use the model to make predictions. We can compute the accuracy of the model on the test set.

    We have produced `q`, which gives the probability of each digit for each image in the test set. For example, `q[0]` gives the probability of each digit for the first image in the test set.

    We want to make a prediction by selecting the digit with the highest probability. We can use the `torch.argmax` function to extract the index of the maximum value in a tensor.
    """)
    return


@app.cell
def _(W, b, softmax, torch, x_test, y_test_raw):
    q = softmax(x_test @ W + b)
    print(q[0])

    print("Predicted digit:", torch.argmax(q[0]).item())
    print("True digit:", y_test_raw[0].item())
    return (q,)


@app.cell
def _(n_test_samples, q, torch, y_test):
    label_pred = torch.argmax(q, dim=1)
    label_true = torch.argmax(y_test, dim=1)
    p_correct_predictions = ( torch.sum(label_pred == label_true) / n_test_samples ).item()
    print(f'Correct predictions: {p_correct_predictions*100:.4f}%')
    return label_pred, label_true


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us show some wrong predictions.
    """)
    return


@app.cell
def _(label_pred, label_true, x_test, y_test):
    mask = label_pred != label_true
    wrong_images = x_test[mask]
    wrong_predictions = y_test[mask]
    wrong_labels = label_pred[mask]
    true_labels = label_true[mask]
    return true_labels, wrong_images, wrong_labels


@app.cell
def _(mo, true_labels, wrong_labels):
    wrong_predictions_options = { f"Predicted: {wrong_labels[i].item()} | True: {true_labels[i].item()}": i for i in range(len(wrong_labels))}
    wrong_predictions_dropdown = mo.ui.dropdown(label="Choose wrong prediction", options=wrong_predictions_options, value=list(wrong_predictions_options.keys())[0])
    return (wrong_predictions_dropdown,)


@app.cell
def _(wrong_predictions_dropdown):
    wrong_predictions_dropdown
    return


@app.cell
def _(plt, wrong_images, wrong_predictions_dropdown):
    def plot_wrong_prediction():

        i = wrong_predictions_dropdown.value
        title = wrong_predictions_dropdown.selected_key

        fig = plt.figure(figsize=(2,2))
        ax = fig.add_subplot(111)
        ax.imshow(wrong_images[i,:].view(28,28), cmap='gray_r')
        ax.set_title(title)
        ax.axis('off')
        ax.set_aspect('equal')

        return fig

    plot_wrong_prediction()
    return


@app.cell(column=5, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Weights

    We can inspect the aspect of weights.
    """)
    return


@app.cell
def _(mo):
    class_dropdown = mo.ui.dropdown(options=[i for i in range(10)], label="Choose class", value=0)
    return (class_dropdown,)


@app.cell
def _(W, class_dropdown, mo, plt):
    def plot_weights():

        fig = plt.figure(figsize=(3,3))
        i = class_dropdown.value
        ax = fig.add_subplot(111)
        heatmap = ax.imshow(W[:,i].view(28,28), cmap='seismic')
        ax.set_title(f'Class {i}')
        ax.axis('off')
        ax.set_aspect('equal')
        plt.colorbar(heatmap)

        return fig 

    mo.vstack([
        class_dropdown,
        plot_weights()
    ], justify='center', align='center')
    return


if __name__ == "__main__":
    app.run()
