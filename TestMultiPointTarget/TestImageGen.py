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

    for i in range(1):
        data = np.load("./iter_result_multi_point_image/csa_out_iter_{}_mag_db.npy".format(i))[int(image_size*3/2):int(image_size*5/2), int(image_size*1):int(image_size*3)]
        # plt.imshow(data, origin='lower', cmap='viridis', aspect=2.0)
        max_val = np.max(data)
        print(max_val)
        clipped_data = np.clip(data, max_val-30, max_val)
        plt.imshow(clipped_data, origin='lower', cmap='viridis', aspect=2.0)
        plt.colorbar()
        plt.savefig("coast_iter_{}.png".format(i), dpi=300, bbox_inches="tight")
        plt.clf()

    x = np.arange(0, 220, 20)
    y1 = [10.6795, 10.6244, 10.5947, 10.5726, 10.5518, 10.5306, 10.5085, 10.4849, 10.4592, 10.431, 10.3998]
    y2 = [1364224, 84790, 60312, 51432, 49504, 48001, 46943, 45941, 44998, 44162, 43255]

    plt.plot(x, y1)
    plt.xlabel("threshold")
    plt.ylabel("entropy")
    plt.grid()
    plt.savefig("entropy.png")
    plt.clf()
    plt.plot(x, y2)
    plt.xlabel("threshold")
    plt.ylabel("number of non-zero pixels")
    plt.grid()
    plt.savefig("n_non_zero_pixel.png")
    plt.clf()
    
    print("DONE")
