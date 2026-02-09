import csv
import numpy as np
import json
import subprocess
import scipy.io
import plotly.graph_objects as go
from TestMultiPointTarget import Scene

import matplotlib.pyplot as plt

def plot_single_point_target():
    data = np.load("./focused_image/single_point_target_mag_db.npy")[(5555 - 400) : (5555 + 400), (2560 - 160) : (2560 + 160)]
    print(data.shape)
    plt.imshow(data, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.savefig("single_point_target.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__": 
    # scene = Scene.Coast
    scene = Scene.Island
    if scene == Scene.Coast:
        image_size = 556
        image_name = "coast"
    elif scene == Scene.Island:
        image_size = 292
        image_name = "island"
    else:
        print("The scene is not supported!")
    data = np.load("./focused_image/multi_point_target_image_mag_db.npy")[int(image_size*3/2):int(image_size*5/2), int(image_size*1):int(image_size*3)]
    print(data.shape)
    # plt.imshow(data, origin='lower', cmap='viridis', aspect=2.0)
    # max_val = np.percentile(data, 99.8)
    max_val = np.max(data)
    clipped_data = np.clip(data, max_val-30, max_val)
    plt.imshow(clipped_data, origin='lower', cmap='viridis', aspect=2.0)
    plt.colorbar()
    plt.savefig(f"{image_name}_csa.png", dpi=300, bbox_inches="tight")
    plt.clf()

    for i in range(5):
        data = np.load("./iter_result_multi_point_image/csa_out_iter_{}_mag_db.npy".format(i))[int(image_size*3/2):int(image_size*5/2), int(image_size*1):int(image_size*3)]
        # plt.imshow(data, origin='lower', cmap='viridis', aspect=2.0)
        max_val = np.max(data)
        print(max_val)
        clipped_data = np.clip(data, max_val-30, max_val)
        plt.imshow(clipped_data, origin='lower', cmap='viridis', aspect=2.0)
        plt.colorbar()
        plt.savefig("coast_iter_{}.png".format(i), dpi=300, bbox_inches="tight")
        plt.clf()
    
    print("DONE")
