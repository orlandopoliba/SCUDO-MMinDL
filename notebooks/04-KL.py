# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.6",
#     "matplotlib==3.10.8",
#     "numpy==2.4.1",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(
    width="columns",
    app_title="Kullback-Leibler Divergence",
    css_file="style.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Kullback-Leibler divergence

    Guess the underlying probability distribution $P$ by inspecting their _Kullback-Leibler divergence_.
    """)
    return


@app.cell
def _(np):
    range = np.arange(0,5)
    return (range,)


@app.cell
def _(mo):
    q0_slider = mo.ui.slider(start=0, stop=1, step=0.01, label=r"$Q(0)$", orientation='vertical',value=1/5)
    q1_slider = mo.ui.slider(start=0, stop=1, step=0.01, label=r"$Q(1)$", orientation='vertical',value=1/5)
    q2_slider = mo.ui.slider(start=0, stop=1, step=0.01, label=r"$Q(2)$", orientation='vertical',value=1/5)
    q3_slider = mo.ui.slider(start=0, stop=1, step=0.01, label=r"$Q(3)$", orientation='vertical',value=1/5)
    q4_slider = mo.ui.slider(start=0, stop=1, step=0.01, label=r"$Q(4)$", orientation='vertical',value=1/5)
    return q0_slider, q1_slider, q2_slider, q3_slider, q4_slider


@app.cell
def _(np, q0_slider, q1_slider, q2_slider, q3_slider, q4_slider):
    Q = np.array([q0_slider.value, q1_slider.value, q2_slider.value, q3_slider.value, q4_slider.value])
    Q = Q / Q.sum()
    return (Q,)


@app.cell
def _(mo):
    seed_input = mo.ui.number(label=r'Random seed for $P$:', value=0)
    return (seed_input,)


@app.cell
def _(np, range, seed_input):
    seed = np.random.seed(seed_input.value)
    P = np.random.randn(range.shape[0])
    P = np.exp(P) / np.exp(P).sum()
    return (P,)


@app.cell
def _(mo):
    visible_switch = mo.ui.switch(label=r'Show $P$')
    return (visible_switch,)


@app.cell
def _(P, Q, np):
    KL_div = - (P * np.log(Q/P)).sum()
    return (KL_div,)


@app.cell
def _(
    KL_div,
    P,
    Q,
    mo,
    plt,
    q0_slider,
    q1_slider,
    q2_slider,
    q3_slider,
    q4_slider,
    range,
    seed_input,
    visible_switch,
):
    def plot():

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(range,Q, color='blue', alpha=0.8, label=r'$Q$')
        visible = visible_switch.value
        if visible:
            opacity = 0.2
        else:
            opacity = 0.0
        ax.bar(range,P, color='red', alpha=opacity, label=r'$P$')
        ax.set_xticks(range)
        ax.set_ylim(0,1)
        ax.legend()

        return fig 

    mo.hstack([

        mo.vstack([

            mo.hstack([seed_input, visible_switch], justify='center', align='center'), 

            plot(), 

            mo.hstack([

                mo.md("distribution $Q:$"), 
                q0_slider,
                q1_slider,
                q2_slider,
                q3_slider,
                q4_slider

                ], 
                justify='center', align='center'
            )
            ],
            justify='center', align='center'
        ),

        mo.md(r"$D_{KL}(P||Q) =$" + f" ${KL_div:.3f}$")

        ],
        justify='center', align='center'
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
