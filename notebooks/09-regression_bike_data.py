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
    app_title="Regression Bike Data",
    css_file="./style.css",
    auto_download=["html"],
)


@app.cell(column=0, hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Seoul Bike Sharing Demand

    ## Importing the dataset

    In this notebook we explore the Seoul Bike Sharing Demand dataset, which is available on the UCI Machine Learning Repository. The dataset is available at the following link: https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand.
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

    data_path = BASE_DIR / 'datasets' / 'SeoulBikeData.csv'
    raw_data = pd.read_csv(data_path)
    return (raw_data,)


@app.cell
def _(raw_data):
    raw_data
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocessing the dataset

    We notice that the data type of some data is not a numerical value, so we need to preprocess the data before we can use it for training a machine learning model.
    """)
    return


@app.cell
def _(raw_data):
    raw_data.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Date

    First of all, we convert the date in the correct data type.
    """)
    return


@app.cell
def _(pd):
    def convert_to_datetime(df):
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        return df
    return (convert_to_datetime,)


@app.cell
def _(convert_to_datetime, raw_data):
    convert_to_datetime(raw_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We separate the date into year, month, day, and weekday.

    It is convenient to preprocess these columns to include the cyclical nature of time. For example, the month of December is closer to January than to July. We can encode this by using the sine and cosine functions to transform the month and day columns into two dimensions. The same can be done for the hour of the day.
    """)
    return


@app.cell
def _(torch):
    def to_sin(column, period):
        return torch.sin(2 * torch.pi / period * torch.tensor(column))

    def to_cos(column, period):
        return torch.cos(2 * torch.pi / period * torch.tensor(column))
    return to_cos, to_sin


@app.cell
def _(to_cos, to_sin):
    def preprocess_datetime(df):

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.dayofweek

        df['Month-sin'] = to_sin(df['Month'],12)
        df['Month-cos'] = to_cos(df['Month'],12)

        df['Day-sin'] = to_sin(df['Day'],31)
        df['Day-cos'] = to_cos(df['Day'],31)

        df['Weekday-sin'] = to_sin(df['Weekday'],7)
        df['Weekday-cos'] = to_cos(df['Weekday'],7)

        df['Hour-sin'] = to_sin(df['Hour'],24)
        df['Hour-cos'] = to_cos(df['Hour'],24)

        df = df.drop(columns=['Date', 'Hour', 'Month', 'Day', 'Weekday'])

        return df
    return (preprocess_datetime,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Seasons

    We do a similar preprocessing to seasons.
    """)
    return


@app.cell
def _(raw_data):
    raw_data['Seasons'].unique()
    return


@app.cell
def _(to_cos, to_sin):
    def season_to_int(season):
        if season == 'Winter':
            return 0
        if season == 'Spring':
            return 1
        if season == 'Summer':
            return 2
        if season == 'Autumn':
            return 3

    def preprocess_seasons(df):
        df['Seasons'].apply(season_to_int)
        df['Seasons-sin'] = to_sin(df['Seasons'].index, 4)
        df['Seasons-cos'] = to_cos(df['Seasons'].index, 4)
        df = df.drop(columns=['Seasons'])
        return df
    return (preprocess_seasons,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Holidays

    We do one-hot encoding to categorical data.
    """)
    return


@app.cell
def _(raw_data):
    raw_data['Holiday'].unique()
    return


@app.cell
def _():
    def holiday_one_hot(entry):
        if entry == 'No Holiday':
            return 0
        if entry == 'Holiday':
            return 1

    def preprocess_holidays(df):
        df['Holiday'] = df['Holiday'].apply(holiday_one_hot)
        return df
    return (preprocess_holidays,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Functioning Days
    """)
    return


@app.cell
def _(raw_data):
    raw_data['Functioning Day'].unique()
    return


@app.cell
def _():
    def functioning_day_one_hot(entry):
        if entry == 'No':
            return 0
        if entry == 'Yes':
            return 1

    def preprocess_functioning_day(df):
        df['Functioning Day'] = df['Functioning Day'].apply(functioning_day_one_hot)
        return df
    return (preprocess_functioning_day,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Converting to `float32`
    """)
    return


@app.function
def convert_to_float32(df):
    df = df.astype('float32')
    return df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Splitting train and test set
    """)
    return


@app.cell
def _(raw_data, torch):
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(len(raw_data))
    train_indices = shuffled_indices[:int(0.8 * len(raw_data))]
    test_indices = shuffled_indices[int(0.8 * len(raw_data)):]
    return test_indices, train_indices


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scaling

    Before training the model, it is a good practice to scale the data.

    **Note**: We do this on the training set and apply the same transformation to the testing set (because we should not assume that we know the mean and standard deviation of the testing set). We do not scale the columns that we have transformed using the sine and cosine functions (they are already between -1 and 1).
    """)
    return


@app.cell
def _(test_indices, train_indices):
    def scale_features(df):

        unscaled_features = ['Rented Bike Count', 'Temperature(C)', 'Humidity(%)',
               'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(C)',
               'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Holiday',
               'Functioning Day', 'Year']

        for feature in unscaled_features:
            mean = df.loc[train_indices, [feature]].mean()
            std = df.loc[train_indices, [feature]].std()
            df.loc[train_indices,[feature]] = (df.loc[train_indices, [feature]] - mean) / std
            df.loc[test_indices, [feature]] = (df.loc[test_indices, [feature]] - mean) /std

        return df
    return (scale_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Applying the preprocessing pipeline
    """)
    return


@app.cell
def _(
    convert_to_datetime,
    preprocess_datetime,
    preprocess_functioning_day,
    preprocess_holidays,
    preprocess_seasons,
    raw_data,
    scale_features,
):
    data = ( 
    raw_data
        .pipe(convert_to_datetime)
        .pipe(preprocess_datetime)
        .pipe(preprocess_seasons)
        .pipe(preprocess_holidays)
        .pipe(preprocess_functioning_day)
        .pipe(convert_to_float32)
        .pipe(scale_features)
    )
    return (data,)


@app.cell
def _(data):
    data.info()
    return


@app.cell
def _(data):
    data
    return


@app.cell
def _(data, test_indices, train_indices):
    data_train = data.loc[train_indices]
    data_test = data.loc[test_indices]
    return data_test, data_train


@app.cell
def _(data_train):
    data_train.describe()
    return


@app.cell
def _(data_test):
    data_test.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are ready to define the tensors for training and test purposes.
    """)
    return


@app.cell
def _(data_test, data_train, torch):
    x_train = torch.tensor( data_train.drop('Rented Bike Count',axis=1).values, dtype=torch.float32)
    x_test = torch.tensor( data_test.drop('Rented Bike Count',axis=1).values, dtype=torch.float32)
    y_train = torch.tensor( data_train['Rented Bike Count'].values, dtype=torch.float32).view(-1,1)
    y_test = torch.tensor( data_test['Rented Bike Count'].values, dtype=torch.float32).view(-1,1)
    return x_test, x_train, y_test, y_train


@app.cell
def _(x_train, y_train):
    print(x_train.shape)
    print(y_train.shape)
    return


@app.cell
def _(x_test, x_train):
    n_train_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    n_features = x_train.shape[1]
    print(f"Number of training samples: {n_train_samples}")
    print(f"Number of test samples: {n_test_samples}")
    print(f"Number of features: {n_features}")
    return n_features, n_train_samples


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear regression

    ### Normal equations

    Let us implement a linear regression model as we did in the first lecture, by solving the normal equations.
    """)
    return


@app.cell
def _(mo, n_train_samples, torch, x_test, x_train, y_test, y_train):
    def linear_regression():

        x_tilde = torch.cat([x_train, torch.ones(n_train_samples, 1)], axis=1)
        A = ( x_tilde.t() @ x_tilde )/x_tilde.shape[0]
        c = x_tilde.t() @ y_train / x_tilde.shape[0]
        solution = torch.linalg.solve(A, c)
        w = solution[:-1]
        b = solution[-1]

        R2 = 1 - torch.sum((x_test @ w + b - y_test)**2)/torch.sum((y_test - y_test.mean())**2)

        return mo.md(fr"$R^2$ on the test set: {R2}")

    linear_regression()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Linear neural network

    Let us implement a linear regression model using our `mydl` package.
    """)
    return


@app.cell
def _():
    from mydl.architecture import Sequential
    from mydl.layers import Linear 
    from mydl.losses import MSE
    from mydl.optimizers import GD
    return GD, Linear, MSE, Sequential


@app.cell
def _(mo):
    learning_rate_slider = mo.ui.slider(start=0.01, stop=0.5, step=0.01, value=0.5, label="Learning Rate", show_value=True)
    n_epochs_slider = mo.ui.slider(start=20, stop=10000, step=100, value=100, label="Number of Epochs",show_value=True)
    return learning_rate_slider, n_epochs_slider


@app.cell
def _(mo):
    train_linear_btn = mo.ui.run_button(label="Train Linear NN")
    return (train_linear_btn,)


@app.cell
def _(learning_rate_slider, n_epochs_slider):
    learning_rate = learning_rate_slider.value
    n_epochs = n_epochs_slider.value
    return learning_rate, n_epochs


@app.cell
def _(
    GD,
    Linear,
    MSE,
    Sequential,
    learning_rate,
    n_epochs,
    n_features,
    torch,
    x_test,
    x_train,
    y_test,
    y_train,
):
    def train_linear_nn():
        model = Sequential([
            Linear(n_features, 1)
        ])
        loss = MSE()
        optimizer = GD(model, learning_rate=learning_rate)    
        losses_history = model.train(x_train, y_train, loss, optimizer, n_epochs=n_epochs)
        # test set evaluation
        y = model.forward(x_test)
        R2 = 1 - torch.sum((y - y_test)**2)/torch.sum((y_test - y_test.mean())**2)
        return R2, losses_history
    return (train_linear_nn,)


@app.cell
def _(learning_rate_slider, mo, n_epochs_slider, train_linear_btn):
    mo.vstack([
        learning_rate_slider,
        n_epochs_slider,
        train_linear_btn
    ])
    return


@app.cell
def _(mo, train_linear_btn, train_linear_nn):
    mo.stop(not train_linear_btn.value, mo.md("Press the button above to run the training."))
    R2_linear_nn, losses_history_linear_nn = train_linear_nn()
    return R2_linear_nn, losses_history_linear_nn


@app.cell
def _(mo, n_epochs_slider):
    first_epoch_slider = mo.ui.slider(start=0, stop=n_epochs_slider.value-1, step=100, value=0, label="First Epoch to Display", show_value=True)
    return (first_epoch_slider,)


@app.cell
def _(R2_linear_nn, first_epoch_slider, losses_history_linear_nn, mo, plt):
    def plot_losses_linear_nn():

        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        first_epoch = first_epoch_slider.value
        ax.plot(range(len(losses_history_linear_nn[first_epoch:])), losses_history_linear_nn[first_epoch:], label='Training Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Training Loss History')
        ax.legend()

        return fig 

    mo.vstack([
        plot_losses_linear_nn(),
        first_epoch_slider,
        mo.md(fr"$R^2$ on the test set: {R2_linear_nn}")
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The performance is comparable with the one obtained with the normal equations.
    """)
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nonlinear neural network

    Let us implement a more expressive model using nonlinearities.
    """)
    return


@app.cell
def _():
    from mydl.layers import Sigmoid
    return (Sigmoid,)


@app.cell
def _(mo):
    train_nonlinear_btn = mo.ui.run_button(label="Train Nonlinear NN")
    return (train_nonlinear_btn,)


@app.cell
def _(
    GD,
    Linear,
    MSE,
    Sequential,
    Sigmoid,
    learning_rate,
    n_epochs,
    n_features,
    torch,
    x_test,
    x_train,
    y_test,
    y_train,
):
    def train_nonlinear_nn():

        #model = Sequential([
        #   Linear(n_features, 10),
        #   Sigmoid(),
        #   Linear(10, 1)
        #])
        #print(model)

        model = Sequential([
             Linear(n_features, 64),
             Sigmoid(),
             Linear(64, 32),
             Sigmoid(),
             Linear(32,16),
             Sigmoid(),
             Linear(16,8),
             Sigmoid(),
             Linear(8,4),
             Sigmoid(),
             Linear(4,2),
             Sigmoid(),
             Linear(2,1)
        ])
        print(model)

        loss = MSE()
        optimizer = GD(model, learning_rate=learning_rate)    
        losses_history = model.train(x_train, y_train, loss, optimizer, n_epochs=n_epochs)
        # test set evaluation
        y = model.forward(x_test)
        R2 = 1 - torch.sum((y - y_test)**2)/torch.sum((y_test - y_test.mean())**2)
        return R2, losses_history
    return (train_nonlinear_nn,)


@app.cell
def _(learning_rate_slider, mo, n_epochs_slider, train_nonlinear_btn):
    mo.vstack([
        learning_rate_slider,
        n_epochs_slider,
        train_nonlinear_btn
    ])
    return


@app.cell
def _(mo, train_nonlinear_btn, train_nonlinear_nn):
    mo.stop(not train_nonlinear_btn.value, mo.md("Press the button above to run the training."))
    R2_nonlinear_nn, losses_history_nonlinear_nn = train_nonlinear_nn()
    return R2_nonlinear_nn, losses_history_nonlinear_nn


@app.cell
def _(
    R2_nonlinear_nn,
    first_epoch_slider,
    losses_history_nonlinear_nn,
    mo,
    plt,
):
    def plot_losses_nonlinear_nn():

        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        first_epoch = first_epoch_slider.value
        ax.plot(range(len(losses_history_nonlinear_nn[first_epoch:])), losses_history_nonlinear_nn[first_epoch:], label='Training Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Training Loss History')
        ax.legend()

        return fig 

    mo.vstack([
        plot_losses_nonlinear_nn(),
        first_epoch_slider,
        mo.md(fr"$R^2$ on the test set: {R2_nonlinear_nn}")
    ])
    return


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using SGD

    We train the network using mini-batches and stochastic gradient descent (SGD).
    """)
    return


@app.cell
def _():
    from mydl.optimizers import SGD
    return (SGD,)


@app.cell
def _(mo):
    train_with_SGD_btn = mo.ui.run_button(label="Train Nonlinear NN with SGD")
    batch_size_slider = mo.ui.slider(start=8, stop=256, step=8, value=32, label="Batch Size", show_value=True)
    return batch_size_slider, train_with_SGD_btn


@app.cell
def _(batch_size_slider):
    batch_size = batch_size_slider.value
    return (batch_size,)


@app.cell
def _(
    Linear,
    MSE,
    SGD,
    Sequential,
    Sigmoid,
    batch_size,
    learning_rate,
    n_epochs,
    n_features,
    torch,
    x_test,
    x_train,
    y_test,
    y_train,
):
    def train_with_SGD():

        model = Sequential([
            Linear(n_features, 64),
            Sigmoid(),
            Linear(64, 32),
            Sigmoid(),
            Linear(32,16),
            Sigmoid(),
            Linear(16,8),
            Sigmoid(),
            Linear(8,4),
            Sigmoid(),
            Linear(4,2),
            Sigmoid(),
            Linear(2,1)
        ])
        print(model)

        loss = MSE()
        optimizer = SGD(model, learning_rate=learning_rate, batch_size=batch_size)    
        losses_history = model.train(x_train, y_train, loss, optimizer, n_epochs=n_epochs)
        # test set evaluation
        y = model.forward(x_test)
        R2 = 1 - torch.sum((y - y_test)**2)/torch.sum((y_test - y_test.mean())**2)
        return R2, losses_history
    return (train_with_SGD,)


@app.cell
def _(
    batch_size_slider,
    learning_rate_slider,
    mo,
    n_epochs_slider,
    train_with_SGD_btn,
):
    mo.vstack([
        learning_rate_slider,
        n_epochs_slider,
        batch_size_slider,
        train_with_SGD_btn
    ])
    return


@app.cell
def _(mo, train_with_SGD, train_with_SGD_btn):
    mo.stop(not train_with_SGD_btn.value, mo.md("Press the button above to run the training."))
    R2_SGD, losses_history_SGD = train_with_SGD()
    return R2_SGD, losses_history_SGD


@app.cell
def _(R2_SGD, first_epoch_slider, losses_history_SGD, mo, plt):
    def plot_losses_SGD():

        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        first_epoch = first_epoch_slider.value
        ax.plot(range(len(losses_history_SGD[first_epoch:])), losses_history_SGD[first_epoch:], label='Training Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Training Loss History')
        ax.legend()

        return fig 

    mo.vstack([
        plot_losses_SGD(),
        first_epoch_slider,
        mo.md(fr"$R^2$ on the test set: {R2_SGD}")
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
