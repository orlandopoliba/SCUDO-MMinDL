# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.1",
#     "pandas==3.0.0",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/risk.slides.json",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return np, pd, plt


@app.cell
def _(mo):
    btn = mo.ui.run_button(label=r"Sample")
    return (btn,)


@app.cell
def _(mo):
    mo.vstack([
    mo.md(r"""
    # Empirical risk

    **Ideal objective**: Find a function $y = f(x)$.


    """)
    ], justify="center", align="start", gap=1)
    return


@app.cell
def _():
    N = 100
    return (N,)


@app.cell
def _(N, btn, np):
    X = 5*np.random.randn(N)
    if btn.value:
        X = 5*np.random.randn(N)
    return (X,)


@app.cell
def _(N, X, np):
    err = 0.1*np.random.randn(N)
    Y = 0.5*np.sin(0.5*X) + err
    return (Y,)


@app.cell
def _(mo, np):
    A_slider = mo.ui.slider(label=r"$A$", start=0.0, stop=2.0, step=0.01, value=1.0)
    omega_slider = mo.ui.slider(label=r"$\omega$", start=0.0, stop=2.0, step=0.01, value=1.0)
    phi_slider = mo.ui.slider(label=r"$\phi$", start=-np.pi, stop=np.pi, step=0.01, value=0.0)
    return A_slider, omega_slider, phi_slider


@app.cell
def _(np):
    def f(x,A,omega,phi):
        return A * np.sin(omega * x + phi)
    return (f,)


@app.cell
def _(A_slider, X, Y, f, omega_slider, phi_slider):
    A = A_slider.value
    omega = omega_slider.value
    phi = phi_slider.value
    y_pred = f(X, A, omega, phi)
    loss0 = (y_pred[0] - Y[0])**2
    return A, loss0, omega, phi, y_pred


@app.cell
def _(X, Y, np, pd, y_pred):
    df = pd.DataFrame({"x": np.round(X,3), "y": np.round(Y,3), "y_pred": np.round(y_pred,3)})
    return (df,)


@app.cell
def _(A_slider, X, Y, btn, loss0, mo, omega_slider, phi_slider):
    mo.vstack([
        mo.md("## Input and output as random variables"),
        mo.md(r"The input $X$ and the output $Y$ are treated as random variables."),
        btn, 
        mo.md(r"$X$" + r" $=$ " + fr"${X[0]:.3f}$"),
        mo.md(r"$Y$" + r" $=$ " + fr"${Y[0]:.3f}$"),
        mo.md(r"**Model**: $f(x;A,\omega,\phi) = A\sin(\omega x + \phi)$"),
        A_slider,
        omega_slider,
        phi_slider,
        mo.md(r"**Loss**: $\ell( y_{\mathrm{pred}}, y_{\mathrm{true}} ) = (y_{\mathrm{pred}} - y_{\mathrm{true}})^2$"),
        mo.md(r"The loss computed on the random variables $X$ and $Y$ is a random variable: $\ell( f(X;A,\omega,\phi), Y )$" + fr" $= {loss0:.4f}$")
    ], justify="center", align="start", gap=1)
    return


@app.cell
def _(A, X, Y, f, np, omega, phi, plt):
    def plot_guess():

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(X, Y, label="Data", color="blue", alpha=0.5)

        t = np.linspace(-15,15,100)
        t_pred = f(t, A, omega, phi)
        ax.plot(t, t_pred, label="Model", color="red")
        ax.set_ylim(-1,1)
        ax.set_xlim(-15,15)
        ax.legend()

        return fig
    return (plot_guess,)


@app.cell
def _(Y, plt, y_pred):
    def plot_l_dist():

        fig = plt.figure()
        ax = fig.add_subplot(111)

        losses = (y_pred - Y)**2
        ax.hist(losses, bins=20, color="green", alpha=0.7, density=True)

        ax.set_xlim(0,4)
        ax.set_ylim(0,3)
        ax.set_xlabel("Loss value")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Losses over the Sample")

        return fig
    return (plot_l_dist,)


@app.cell
def _(
    A_slider,
    Y,
    btn,
    df,
    mo,
    np,
    omega_slider,
    phi_slider,
    plot_l_dist,
    y_pred,
):
    mo.hstack([
        mo.vstack([
            mo.md(r"**Risk**: $\mathbb{E}[ \ell( f(X;A,\omega,\phi), Y ) ]$"),
            mo.md(r"**Sample**: $(X_0,Y_0), (X_1, Y_1), \ldots, (X_{N-1}, Y_{N-1})$"),
            btn,
            mo.md(r"Dataset (observation of the sample): $(x_0,y_0), (x_1, y_1), \ldots, (x_{N-1}, y_{N-1})$"),
            mo.ui.table(df, page_size=5),
            mo.md(r"**Empirical risk**: $\displaystyle \frac{1}{N} \sum_{i=0}^{N-1} \ell( f(X_i;A,\omega,\phi), Y_i )$" + fr" $= {np.mean((y_pred - Y)**2):.6f}$")
        ], justify="center", align="start", gap=1),
        mo.vstack([
            plot_l_dist(),
            A_slider,
            omega_slider,
            phi_slider
        ], justify='center', align="center", gap=1)
    ], gap=0, justify="center", align="start")
    return


@app.cell
def _(
    A_slider,
    Y,
    btn,
    df,
    mo,
    np,
    omega_slider,
    phi_slider,
    plot_guess,
    y_pred,
):
    mo.hstack([
        mo.vstack([
            mo.md(r"**Risk**: $\mathbb{E}[ \ell( f(X;A,\omega,\phi), Y ) ]$"),
            mo.md(r"**Sample**: $(X_0,Y_0), (X_1, Y_1), \ldots, (X_{N-1}, Y_{N-1})$"),
            btn,
            mo.md(r"Dataset (observation of the sample): $(x_0,y_0), (x_1, y_1), \ldots, (x_{N-1}, y_{N-1})$"),
            mo.ui.table(df, page_size=5),
            mo.md(r"**Empirical risk**: $\displaystyle \frac{1}{N} \sum_{i=0}^{N-1} \ell( f(X_i;A,\omega,\phi), Y_i )$" + fr" $= {np.mean((y_pred - Y)**2):.6f}$")
        ], justify="center", align="start", gap=1),
        mo.vstack([
            plot_guess(),
            A_slider,
            omega_slider,
            phi_slider
        ], justify='center', align="center", gap=1)
    ], gap=0, justify="center", align="start")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
