# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.19.9",
#     "numpy==2.4.2",
#     "plotly==6.5.2",
#     "wigglystuff==0.2.24",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(
    width="medium",
    app_title="Universal Approximation Theorem",
    css_file="style.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    return go, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Universal Approximation Theorem

    In this notebook we show some fleshed out examples of the Universal Approximation Theorem.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Approximation with step functions

    First, we show that step functions can approximate any continuous function.
    """)
    return


@app.cell
def _(np):
    class Step:
        def __init__(self, step_points, step_heights):
            self.step_points = step_points
            self.step_heights = step_heights

        def __call__(self, x):
            x = np.array(x)
            return (
                np.sum(
                    [
                        (x >= self.step_points[i])
                        * (x < self.step_points[i + 1])
                        * self.step_heights[i]
                        for i in range(len(self.step_points) - 1)
                    ],
                    axis=0,
                )
                + (x >= self.step_points[-1]) * self.step_heights[-1]
            )

        def interpolate(self, f):
            self.step_heights = [f(p) for p in self.step_points]
    return (Step,)


@app.cell
def _(mo):
    n_steps_slider = mo.ui.slider(
        label="Number of steps", start=1, stop=100, value=1, step=1
    )
    return (n_steps_slider,)


@app.cell
def _(Step, go, n_steps_slider, np):
    def f(x):
        return 4 * x**2 + np.cos(2 * 2 * np.pi * x)

    def plot_approximation():
        x = np.linspace(0, 1, 100)
        y = f(x)

        n_steps = n_steps_slider.value or 1
        step = Step(step_points=np.linspace(0, 1, n_steps + 1), step_heights=None)
        step.interpolate(f)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="f(x)"))
        fig.add_trace(
            go.Scatter(
                x=step.step_points,
                y=step.step_heights,
                mode="lines",
                name="Step approximation",
                line_shape="hv",
            )
        )
        fig.update_layout(
            title="Approximation of function f with step functions",
            xaxis_title="x",
            yaxis_title="f(x)",
            xaxis=dict(
                showgrid=False,
            ),
            yaxis=dict(
                showgrid=False,
            ),
            showlegend=True,
        )
        return fig
    return (plot_approximation,)


@app.cell
def _(mo, n_steps_slider, plot_approximation):
    mo.vstack([
        plot_approximation(),
        n_steps_slider
    ], justify="center", align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sigmoid is a smooth version of a smooth function
    """)
    return


@app.cell
def _(mo):
    w_slider = mo.ui.slider(label=r"$w$", start=0.1, stop=20, value=1, step=0.1)
    b_slider = mo.ui.slider(label=r"$b$", start=-20, stop=20, value=0, step=0.1)
    return b_slider, w_slider


@app.cell
def _(Step, b_slider, go, np, w_slider):
    def sigmoid(t):
        return 1 / (1 + np.exp(-t))

    def plot_scaled_sigmoid():
        step_points = np.array([-3.,-1.,3.])
        step_heigths = np.array([0., 1., 1.])
        heaviside = Step(step_points=step_points, step_heights=step_heigths)

        x = np.linspace(-3, 3, 100)

        w = w_slider.value
        b = b_slider.value
        y_sigmoid = sigmoid(w * x + b)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=y_sigmoid, 
                mode="lines")
        )
        fig.add_trace(
            go.Scatter(
                x=heaviside.step_points, 
                y=heaviside.step_heights, 
                mode="lines", 
                line_shape="hv"
            )
        )
        fig.update_layout(
            title="Scaled sigmoid approximating a step function",
            xaxis_title="x",
            yaxis_title="f(x)",
            xaxis=dict(
                showgrid=False,
            ),
            yaxis=dict(
                showgrid=False,
            ),
            showlegend=False,
        )
        return fig
    return (plot_scaled_sigmoid,)


@app.cell
def _(b_slider, mo, plot_scaled_sigmoid, w_slider):
    mo.vstack([
        plot_scaled_sigmoid(),
        w_slider,
        b_slider
    ], justify="center", align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using a neural network to approximate a function
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
