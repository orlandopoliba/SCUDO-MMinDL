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
        self.gradL_d = {}

    def forward(self,x):
        raise NotImplementedError("Forward method not implemented")

    def backward(self,dL_dy):
        raise NotImplementedError("Backward method not implemented")


@app.class_definition
class Linear(Layer): 

    def __init__(self, fan_in, fan_out):
        super().__init__()
        self.parameters['W'] = torch.randn((fan_in, fan_out), dtype=torch.float32, requires_grad=False)
        self.parameters['b'] = torch.zeros((1,fan_out), dtype=torch.float32, requires_grad=False)
        self.n_parameters = fan_in * fan_out + fan_out

    def forward(self,x):
        self.x = x
        return x @ self.parameters['W'] + self.parameters['b']

    def backward(self,dL_dy):
        self.gradL_d['W'] = (dL_dy @ self.x).t()
        self.gradL_d['b'] = torch.sum( dL_dy, dim=1, keepdim=True ).t()
        dL_dx = self.parameters['W'] @ dL_dy
        return dL_dx


@app.class_definition
class Sigmoid(Layer):

    def __init__(self):
        super().__init__()
        self.n_parameters = 0

    def forward(self,x):
        self.y = 1 / (1 + torch.exp(-x))
        return self.y

    def backward(self,dL_dy):
        dL_dx = dL_dy * self.y * (1 - self.y).t()
        return dL_dx


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
