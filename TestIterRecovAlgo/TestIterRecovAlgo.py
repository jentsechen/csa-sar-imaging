import csv
import numpy as np
import json
import subprocess
import scipy.io
import plotly.graph_objs as go
import plotly.offline as pof
from plotly.subplots import make_subplots
from enum import Enum, auto
import time

if __name__ == "__main__":
    start_time = time.time()
    proc = subprocess.run(
        "../build/bin/Debug/TestIterRecovAlgo.exe",
        capture_output=True,  # Capture stdout and stderr
        text=True,  # Decode output as text (string)
        check=True,  # Raise an exception for non-zero exit codes (errors)
    )
    print("\n--- C++ Program Output (STDOUT) ---")
    print(proc.stdout)
    exe_time_sec = time.time() - start_time
    print("Execution time: {} seconds".format(exe_time_sec))

    # echo_signal = np.load("./echo_signal.npy")
    csa_out_iter_0 = np.load("./result/csa_out_iter_0.npy")
    csa_out_iter_1 = np.load("./result/csa_out_iter_1.npy")
    # print(csa_out_iter_0.shape, csa_out_iter_1.shape)
    figure = make_subplots(rows=1, cols=1)
    figure.add_trace(go.Scatter(y=csa_out_iter_0[5555, :].real), row=1, col=1)
    figure.add_trace(go.Scatter(y=csa_out_iter_1[5555, :].real), row=1, col=1)
    pof.iplot(figure)
