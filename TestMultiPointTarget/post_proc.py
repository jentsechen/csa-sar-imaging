import csv
import numpy as np
import json
import subprocess
import scipy.io
import plotly.graph_objects as go

def run_cpp(cmd):
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    print("\n--- C++ Program Output (STDOUT) ---")
    print(proc.stdout)

if __name__ == "__main__":
    run_cpp(cmd=["../build/linear_to_db_scale", "./focused_image/single_point_target"])
    image = np.load("./focused_image/single_point_target_mag_db.npy")
    print(image.shape)
    print("DONE")