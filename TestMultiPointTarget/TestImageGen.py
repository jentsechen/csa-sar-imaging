import csv
import numpy as np
import json
import subprocess
import scipy.io
import plotly.graph_objects as go

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.load("./iter_result_multi_point_image/csa_out_iter_9_mag_db.npy")
    # data = np.load("./focused_image/multi_point_target_image_mag_db.npy")
    print(data.shape)
    plt.imshow(data, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.savefig("coast_iter_recov.png", dpi=300, bbox_inches="tight")
    print("DONE")
