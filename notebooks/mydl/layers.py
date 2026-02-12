# /// script
# dependencies = [
#     "marimo",
#     "torch==2.9.1",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    import torch


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return


@app.class_definition
class Layer: 

    def __init__(self): 
        self.parameters = {}

    def forward(self,x):
        raise NotImplementedError("Forward method not implemented")


@app.class_definition
class Linear(Layer): 

    def __init__(self, fan_in, fan_out):
        super().__init__()
        self.parameters['W'] = torch.randn((fan_in, fan_out), dtype=torch.float32, requires_grad=False)
        self.parameters['b'] = torch.zeros((1,fan_out), dtype=torch.float32, requires_grad=False)
        self.n_parameters = fan_in * fan_out + fan_out

    def forward(self,x):
        return x @ self.parameters['W'] + self.parameters['b']


@app.class_definition
class Sigmoid(Layer):

    def __init__(self):
        super().__init__()
        self.n_parameters = 0

    def forward(self,x):
        return 1 / (1 + torch.exp(-x))


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
