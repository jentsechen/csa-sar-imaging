import csv
import numpy as np
import json
import subprocess
import scipy.io
import plotly.graph_objects as go

import matplotlib.pyplot as plt

def plot_single_point_target():
    data = np.load("./focused_image/single_point_target_mag_db.npy")[(5555 - 400) : (5555 + 400), (2560 - 160) : (2560 + 160)]
    print(data.shape)
    plt.imshow(data, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.savefig("single_point_target.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":    
    data = np.load("./focused_image/multi_point_target_image_mag_db.npy")[int(556/2):int(556*7/2), int(556/2):int(556*7/2)]
    # data = abs(np.load("./focused_image/multi_point_target_image.npy")[int(556/2):int(556*7/2), int(556/2):int(556*7/2)])
    print(data.shape)
    # plt.imshow(data, origin='lower', cmap='viridis', aspect=2.0)
    clipped_data = np.clip(data, np.max(data)-50, np.max(data))
    plt.imshow(clipped_data, origin='lower', cmap='viridis', aspect=2.0)
    plt.colorbar()
    plt.savefig("coast_csa.png", dpi=300, bbox_inches="tight")
    # for i in range(5):
    #     data = np.load("./iter_result_multi_point_image/csa_out_iter_{}_mag_db.npy".format(i))[:, int(556/2):int(556*7/2)]
    #     plt.imshow(data, origin='lower', cmap='viridis', aspect=2.0)
    #     plt.colorbar()
    #     plt.savefig("coast_iter_{}.png".format(i), dpi=300, bbox_inches="tight")
    #     plt.clf()
    
    print("DONE")
