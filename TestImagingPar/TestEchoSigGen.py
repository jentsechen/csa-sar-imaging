import csv
import numpy as np
import json
import subprocess
import matplotlib.pyplot as plt
from TestConstructor import read_csv
import plotly.graph_objs as go
import plotly.offline as pof
from plotly.subplots import make_subplots
import scipy.io


if __name__ == "__main__":
    point_target_echo_signal = scipy.io.loadmat("point_target_echo_signal.mat")[
        "point_target_echo_signal"
    ]
    print(point_target_echo_signal.shape)

    plt.figure()
    plt.imshow(point_target_echo_signal[:, 1920:3200].real)
    plt.colorbar(label="Amplitude")
    plt.title("Matrix Visualization")
    plt.xlabel("X Index")
    plt.ylabel("Y Index")
    plt.show()

    # figure = make_subplots(rows=1, cols=1)
    # figure.add_trace(go.Scatter(y=col_vec.real), row=1, col=1)
    # # figure.update_layout(
    # #     xaxis=dict(title="sample"),
    # #     yaxis=dict(title="amplitude (real part)"),
    # #     xaxis2=dict(title="sample"),
    # #     yaxis2=dict(title="amplitude (imaginary part)"),
    # #     font=dict(size=20),
    # # )
    # figure.write_html("data.html")
