# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.19.6",
#     "numpy==2.4.2",
#     "plotly==6.5.2",
#     "wigglystuff==0.2.16",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(
    width="columns",
    app_title="Gradient Descent",
    css_file="style.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Gradient descent
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    return go, np


@app.cell
def _(mo):
    mo.md(r"""
    ## Gradient of a function

    The gradient $\nabla L(w)$ of a function $L$ at a point $w$ indicates the direction of steepest ascent.
    """)
    return


@app.cell
def _(np):
    # L, grad_L

    def L(w0,w1):
        return 2*w0**2 + w1**2

    def grad_L(w0,w1):
        return np.array([4*w0, 2*w1])
    return L, grad_L


@app.cell
def _(mo):
    # w sliders

    w0_slider = mo.ui.slider(start=-5, stop=5, step=0.1,label=r'$w^0_1$',value=1)
    w1_slider = mo.ui.slider(start=-5, stop=5, step=0.1,label=r'$w^0_2$',value=0)
    return w0_slider, w1_slider


@app.cell
def _(mo):
    mo.md(r"""
    **Instructions**: Move the sliders to change the point where the gradient is computed.
    """)
    return


@app.cell
def _(L, go, grad_L, mo, np, w0_slider, w1_slider):
    # plots

    def plot_L_contour():

        w0 = np.zeros(2)
        w0[0] = w0_slider.value
        w0[1] = w1_slider.value
        z0 = L(w0[0], w0[1])

        tau = 0.15
        w1 = np.zeros_like(w0)
        w1 = w0 + tau * grad_L(w0[0], w0[1])
        z1 = L(w1[0], w1[1])

        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = L(X,Y)

        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=Z,
                colorscale="Viridis",
                contours=dict(coloring="heatmap", showlines=False),
                showscale=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[w0[0]],
                y=[w0[1]],
                mode="markers",
                marker=dict(color="red", size=10),
                showlegend=False,
            )
        )
        # add gradient vector
        fig.add_trace(
            go.Scatter(
                x=[w0[0], w1[0]],
                y=[w0[1], w1[1]],
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(color="blue", size=5),
                showlegend=False,
            )
        )

        fig.update_layout(
            width=450,
            height=450,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(scaleanchor="y"),
        )
        return fig

    def plot_L_3D():

        w0 = np.zeros_like(grad_L(0,0))
        w0[0] = w0_slider.value
        w0[1] = w1_slider.value
        z0 = L(w0[0], w0[1])

        # surface grid
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = L(X, Y)

        # gradient at (x0, y0)
        norm = np.linalg.norm(grad_L(w0[0], w0[1]))
        nu = grad_L(w0[0], w0[1]) / norm

        # line parameter
        t = np.linspace(-5, 5, 100)
        w0_line = w0[0] + t * nu[0]
        w1_line = w0[1] + t * nu[1]
        z_line = L(w0_line, w1_line)

        fig = go.Figure()

        # surface
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=Z,
                colorscale="Viridis",
                showscale=False,
                opacity=0.75,
            )
        )

        # point
        fig.add_trace(
            go.Scatter3d(
                x=[w0[0]],
                y=[w0[1]],
                z=[z0],
                mode="markers",
                marker=dict(color="red", size=5),
                showlegend=False,
            )
        )

        # trace along gradient direction
        fig.add_trace(
            go.Scatter3d(
                x=w0_line,
                y=w1_line,
                z=z_line,
                mode="lines",
                line=dict(color="blue", width=5),
                showlegend=False,
            )
        )

        fig.update_layout(
            width=450,
            height=450,
            margin=dict(l=10, r=10, t=10, b=10),
            scene=dict(
                aspectmode="cube",
                dragmode="turntable"
            ),
        )

        return fig


    mo.hstack([mo.vstack([plot_L_contour(), w0_slider, w1_slider], justify='center', align='center'), plot_L_3D()], justify='center', align='center')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Gradient descent algorithm

    The gradient descent algorithm starts from an initial guess. At each iteration, it computes the gradient of the function at the current point and updates the point by moving in the opposite direction of the gradient, scaled by a _learning rate_.
    """)
    return


@app.cell
def _(mo):
    # steps and learning rate sliders

    steps_slider = mo.ui.slider(start=1, stop=20, step=1, label=r'Number of steps', value=1)
    learning_rate_slider = mo.ui.slider(start=0.001, stop=1.0, step=0.001, label=r'Learning rate', value=0.2)
    return learning_rate_slider, steps_slider


@app.cell
def _(mo):
    mo.md(r"""
    In 2 dimensions, if the objective is to minimize $L(w)$, the algorithm is:

    - Choose an initial guess $w^0$
    - Choose a learning rate $\tau$
    - For each step $k \geq 0$ do:
        - Compute the gradient $\nabla L(w^k)$
        - Update the point with the formula: $w^{k+1} = w^k - \tau \nabla L (w^k)$
    """)
    return


@app.function
# algorithm
def gradient_descent(L,grad_L,w0,learning_rate,steps):

    w = w0
    list_of_points = [w]
    for _ in range(steps):
        w = w - learning_rate * grad_L(w[0],w[1])
        list_of_points.append(w)
    return list_of_points


@app.cell
def _(mo):
    mo.md(r"""
    **Instructions**: Choose the coordinates of the initial point, the learning rate, and the number of steps with the sliders.
    """)
    return


@app.cell
def _(
    L,
    go,
    grad_L,
    learning_rate_slider,
    mo,
    np,
    steps_slider,
    w0_slider,
    w1_slider,
):
    # plots for gradient descent

    def plot_gradient_descent_contour(L,grad_L):

        fig = go.Figure()

        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = L(X,Y)

        fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=Z,
                colorscale="Viridis",
                contours=dict(coloring="heatmap", showlines=False),
                showscale=False,
            )
        )

        fig.update_layout(
            width=350,
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(scaleanchor="y"),
        )

        list_of_points = gradient_descent(L=L,grad_L=grad_L,
            w0=np.array([w0_slider.value, w1_slider.value]),
            learning_rate=learning_rate_slider.value,
            steps=steps_slider.value,
        )

        x_points = [p[0] for p in list_of_points]
        y_points = [p[1] for p in list_of_points]

        fig.add_trace(
            go.Scatter(
                x=x_points,
                y=y_points,
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(color="red", size=5),
                showlegend=False,
            )
        )

        return fig

    camera_state = {"eye": dict(x=1.25, y=1.25, z=1.25)}

    def plot_gradient_descent_3D(L,grad_L):

        global camera_state


        fig = go.Figure()

        # surface grid
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = L(X, Y)

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=Z,
                colorscale="Viridis",
                showscale=False,
                opacity=0.75,
            )
        )

        fig.update_layout(
            width=350,
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            scene=dict(
                aspectmode="cube",
                dragmode="turntable"
            ),
        )

        list_of_points = gradient_descent(L=L,grad_L=grad_L,
            w0=np.array([w0_slider.value, w1_slider.value]),
            learning_rate=learning_rate_slider.value,
            steps=steps_slider.value,
        )

        x_points = [p[0] for p in list_of_points]
        y_points = [p[1] for p in list_of_points]
        z_points = [L(p[0], p[1]) for p in list_of_points]

        fig.add_trace(
            go.Scatter3d(
                x=x_points,
                y=y_points,
                z=z_points,
                mode="lines+markers",
                line=dict(color="red", width=5),
                marker=dict(color="red", size=5),
                showlegend=False,
            )
        )

        return fig

    def plot_cost_during_training(L,grad_L):

        list_of_points = gradient_descent(L=L,grad_L=grad_L,
            w0=np.array([w0_slider.value, w1_slider.value]),
            learning_rate=learning_rate_slider.value,
            steps=steps_slider.value,
        )

        costs = [L(p[0], p[1]) for p in list_of_points]
        steps = list(range(len(costs)))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=costs,
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(color="red", size=5),
                showlegend=False,
            )
        )

        fig.update_layout(
            width=350,
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Step",
            yaxis_title="Cost L(w)",
        )

        return fig

    mo.hstack([mo.vstack([plot_gradient_descent_contour(L,grad_L), w0_slider, w1_slider, learning_rate_slider, steps_slider], justify='center', align='center'), plot_gradient_descent_3D(L,grad_L), plot_cost_during_training(L,grad_L)], justify='center', align='center')
    return (
        plot_cost_during_training,
        plot_gradient_descent_3D,
        plot_gradient_descent_contour,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Quadratic losses

    Let us study quadratic losses of the form

    $$
    L(w) = \frac{1}{2} w^T A w
    $$

    where $A$ is a symmetric positive definite matrix.
    """)
    return


@app.cell
def _(mo, np):
    # theta, eig_1, eig_2 sliders
    theta_slider = mo.ui.slider(start=0, stop=2*np.pi, step=2*np.pi/100, label=r'$\theta$', value=0)
    eig_1_slider = mo.ui.slider(start=0.1, stop=5.0, step=0.1, label=r'$\lambda_1$', value=1.0)
    eig_2_slider = mo.ui.slider(start=0.1, stop=5.0, step=0.1, label=r'$\lambda_2$', value=1.0)
    return eig_1_slider, eig_2_slider, theta_slider


@app.cell
def _(np, theta_slider):
    # Q
    theta = theta_slider.value
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return (Q,)


@app.cell
def _(eig_1_slider, eig_2_slider, np):
    # D
    eig_1 = eig_1_slider.value
    eig_2 = eig_2_slider.value
    D = np.array([[eig_1, 0],
                  [0, eig_2]])
    return D, eig_1, eig_2


@app.cell
def _(D, Q):
    # A
    A = Q @ D @ Q.T
    return (A,)


@app.cell
def _(A):
    # L_quadratic

    def L_quadratic(w0,w1):
        return 0.5 * ( w0 * (A[0,0] * w0 + A[0,1] * w1) + w1 * (A[1,0] * w0 + A[1,1] * w1) )
    return (L_quadratic,)


@app.cell
def _(L_quadratic, go, np):
    # plot L_quadratic

    def plot_L_quadratic_contour():

        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = L_quadratic(X,Y)

        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=Z,
                colorscale="Viridis",
                contours=dict(coloring="heatmap", showlines=False),
                showscale=False,
            )
        )

        fig.update_layout(
            width=450,
            height=450,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(scaleanchor="y"),
        )
        return fig


    def plot_L_quadratic_3D():

        # surface grid
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = L_quadratic(X, Y)

        fig = go.Figure()

        # surface
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=Z,
                colorscale="Viridis",
                showscale=False,
                opacity=0.75,
            )
        )

        fig.update_layout(
            width=450,
            height=450,
            margin=dict(l=10, r=10, t=10, b=10),
            scene=dict(
                aspectmode="cube",
                dragmode="turntable",
                zaxis=dict(range=[0, 400])
            ),
        )

        return fig
    return plot_L_quadratic_3D, plot_L_quadratic_contour


@app.cell
def _(
    A,
    D,
    Q,
    eig_1_slider,
    eig_2_slider,
    mo,
    plot_L_quadratic_3D,
    plot_L_quadratic_contour,
    theta_slider,
):
    mo.vstack([
        theta_slider,
        eig_1_slider,
        eig_2_slider,
        mo.md(fr"""

        $$
        Q = 
        \begin{{pmatrix}}
        \cos(\theta) & -\sin(\theta) \\
        \sin(\theta) & \cos(\theta)
        \end{{pmatrix}}
        = \begin{{pmatrix}}
        {Q[0,0]:.2f} & {Q[0,1]:.2f} \\
        {Q[1,0]:.2f} & {Q[1,1]:.2f}
        \end{{pmatrix}}
        $$

        """
        ),

        mo.md(fr"""

        $$
        D =
        \begin{{pmatrix}}
        \lambda_1 & 0 \\
        0 & \lambda_2
        \end{{pmatrix}}
        = \begin{{pmatrix}}
        {D[0,0]:.2f} & 0 \\
        0 & {D[1,1]:.2f}
        \end{{pmatrix}}
        $$
        """),

        mo.md(fr"""

        $$
        A = Q D Q^T =
        \begin{{pmatrix}}
        {Q[0,0]:.2f} & {Q[0,1]:.2f} \\
        {Q[1,0]:.2f} & {Q[1,1]:.2f}
        \end{{pmatrix}} 
        \begin{{pmatrix}}
        {D[0,0]:.2f} & 0 \\
        0 & {D[1,1]:.2f}
        \end{{pmatrix}}
        \begin{{pmatrix}}
        {Q[0,0]:.2f} & {Q[1,0]:.2f} \\
        {Q[0,1]:.2f} & {Q[1,1]:.2f}
        \end{{pmatrix}}
        = 
        \begin{{pmatrix}}
        {A[0,0]:.2f} & {A[0,1]:.2f} \\
        {A[1,0]:.2f} & {A[1,1]:.2f}
        \end{{pmatrix}}
        $$

        """),

        mo.hstack([
            plot_L_quadratic_contour(),
            plot_L_quadratic_3D()
        ], justify='center', align='center')

    ], justify='center', align='start')
    return


@app.cell
def _(mo):
    mo.md(r"""
    We can find the optimal learning rate that minimizes the convergence rate.
    """)
    return


@app.cell
def _(eig_1, eig_1_slider, eig_2, eig_2_slider, go, mo, np):
    def plot_rate():

        tau = np.linspace(0, 1, 200)

        abs1 = np.abs(1 - tau * eig_1)
        abs2 = np.abs(1 - tau * eig_2)

        rate = np.maximum(np.abs(1 - tau * eig_1), np.abs(1 - tau * eig_2))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=tau,
                y=abs1,
                mode="lines",
                line=dict(color="red", width=1, dash="dash"),
                name=r'$|1 - \tau \lambda_1|$',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=tau,
                y=abs2,
                mode="lines",
                line=dict(color="green", width=1, dash="dash"),
                name=r'$|1 - \tau \lambda_2|$',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=tau,
                y=rate,
                mode="lines",
                line=dict(color="blue", width=2),
                showlegend=False,
            )
        )
        fig.update_layout(
            width=650,
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Learning rate $\\tau$",
            yaxis_title="Convergence rate",
            yaxis=dict(range=[0, 2])
        )
        return fig

    def plot_optimal_rate():

        condition_numbers = np.linspace(1, 50, 200)
        optimal_rates = (condition_numbers - 1) / (condition_numbers + 1)

        tau_optimal = 2 / (eig_1 + eig_2)
        rate_optimal = (eig_2 - eig_1) / (eig_1 + eig_2)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=condition_numbers,
                y=optimal_rates,
                mode="lines",
                line=dict(color="blue", width=2),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[eig_2 / eig_1],
                y=[rate_optimal],
                mode="markers",
                marker=dict(color="red", size=10),
            )
        )
        fig.update_layout(
            width=650,
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Condition number $\\kappa = \\frac{\\lambda_1}{\\lambda_2}$",
            yaxis_title="Optimal convergence rate",
            xaxis=dict(range=[1, 50]),
            yaxis=dict(range=[0, 1])
        )
        return fig

    mo.vstack([
        eig_1_slider,
        eig_2_slider,
        mo.hstack([
            plot_rate(),
            plot_optimal_rate()
        ], justify='center', align='center')
    ], justify='center', align='center')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Issues
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Issue 1: suitable choice of the learning rate

    The learning rate is a hyperparameter that controls how much we are adjusting the weights of our network with respect the loss gradient. If the learning rate is too large, the algorithm can get stuck or, even worse, diverge. If the learning rate is too small, the algorithm will take too long to converge.
    """)
    return


@app.cell
def _(
    L,
    grad_L,
    learning_rate_slider,
    mo,
    plot_cost_during_training,
    plot_gradient_descent_3D,
    plot_gradient_descent_contour,
    steps_slider,
    w0_slider,
    w1_slider,
):
    mo.hstack([mo.vstack([plot_gradient_descent_contour(L,grad_L), w0_slider, w1_slider, learning_rate_slider, steps_slider], justify='center', align='center'), plot_gradient_descent_3D(L,grad_L), plot_cost_during_training(L,grad_L)], justify='center', align='center')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Issue 2: local minima

    Even when the learning rate is well chosen, the gradient descent algorithm can get stuck around a local minimum.

    Try the following example with a non-convex function.
    """)
    return


@app.cell
def _(np):
    # non-convex L and grad_L

    def L_non_convex(w0,w1):
        return 80 + w0**2 - 40*np.cos(2*np.pi*w0/5) + w1**2 - 40*np.cos(2*np.pi*w1/5)

    def grad_L_non_convex(w0,w1):
        dL_dx = 2*w0 + 20*np.pi*np.sin(2*np.pi*w0)
        dL_dy = 2*w1 + 20*np.pi*np.sin(2*np.pi*w1)
        return np.array([dL_dx, dL_dy])
    return L_non_convex, grad_L_non_convex


@app.cell
def _(
    L_non_convex,
    grad_L_non_convex,
    learning_rate_slider,
    mo,
    plot_cost_during_training,
    plot_gradient_descent_3D,
    plot_gradient_descent_contour,
    steps_slider,
    w0_slider,
    w1_slider,
):
    # plots for gradient descent on non-convex loss

    mo.hstack([mo.vstack([plot_gradient_descent_contour(L_non_convex,grad_L_non_convex), w0_slider, w1_slider, learning_rate_slider, steps_slider], justify='center', align='center'), plot_gradient_descent_3D(L_non_convex,grad_L_non_convex), plot_cost_during_training(L_non_convex,grad_L_non_convex)], justify='center', align='center')
    return


if __name__ == "__main__":
    app.run()
