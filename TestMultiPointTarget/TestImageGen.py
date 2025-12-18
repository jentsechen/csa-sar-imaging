import csv
import numpy as np
import json
import subprocess
import scipy.io
import plotly.graph_objects as go

def run_cpp(args=[]):
    proc = subprocess.run(
        "../build/bin/Debug/TestMultiPointTarget.exe {} {}".format(args[0], args[1]),
        capture_output=True,  # Capture stdout and stderr
        text=True,  # Decode output as text (string)
        check=True,  # Raise an exception for non-zero exit codes (errors)
    )
    print("\n--- C++ Program Output (STDOUT) ---")
    print(proc.stdout)

if __name__ == "__main__":
    run_cpp(args=["test_image", "dont_care"])
    # run_cpp(args=["test", "dont_care"])
    print("DONE")
