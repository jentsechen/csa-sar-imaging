import csv
import numpy as np
import json
import subprocess
import scipy.io
import plotly.graph_objs as go
import plotly.offline as pof
from plotly.subplots import make_subplots
from enum import Enum, auto


class CheckDataType(Enum):
    ColVec, Mat = auto(), auto()


def col2row(input):
    output = []
    for i in input:
        output.append(i[0])
    return np.array(output)


def check(par_name, data_type):
    if data_type == CheckDataType.ColVec:
        golden = col2row(
            scipy.io.loadmat("./golden/{}.mat".format(par_name))["{}".format(par_name)]
        )
    else:
        golden = scipy.io.loadmat("./golden/{}.mat".format(par_name))[
            "{}".format(par_name)
        ]
    result = np.load("./result/{}.npy".format(par_name))
    print(golden.shape, result.shape)
    if data_type == CheckDataType.ColVec:
        print("error of {}: {}".format(par_name, sum(abs(result - golden))))
    elif data_type == CheckDataType.Mat:
        print("error of {}: {}".format(par_name, sum(sum(abs(result - golden)))))
    else:
        print("data type is not supported")
    return golden, result


if __name__ == "__main__":
    proc = subprocess.run(
        "../build/bin/Debug/TestChirpScalingAlgo.exe",
        capture_output=True,  # Capture stdout and stderr
        text=True,  # Decode output as text (string)
        check=True,  # Raise an exception for non-zero exit codes (errors)
    )
    print("\n--- C++ Program Output (STDOUT) ---")
    print(proc.stdout)

    check("migr_par", CheckDataType.ColVec)
    check("modified_range_fm_rate_hz_s", CheckDataType.ColVec)
    check("chirp_scaling", CheckDataType.Mat)

    # par_name = "chirp_scaling"
    # result = np.load("./result/{}.npy".format(par_name))
    # golden = scipy.io.loadmat("./golden/{}.mat".format(par_name))["{}".format(par_name)]

    # figure = make_subplots(rows=1, cols=1)
    # figure.add_trace(go.Scatter(y=golden[(0+3000), :].real), row=1, col=1)
    # figure.add_trace(go.Scatter(y=result[(5555+3000), :].real), row=1, col=1)
    # # figure.write_html("data.html")
    # pof.iplot(figure)

