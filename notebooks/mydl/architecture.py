# /// script
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "torch==2.9.1",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    return


@app.class_definition
class Sequential: 

    def __init__(self, layers):
        self.layers = layers    

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
