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

__generated_with = "0.19.4"
app = marimo.App(
    width="medium",
    app_title="Regression Bike Data",
    css_file="style.css",
    auto_download=["html"],
)
