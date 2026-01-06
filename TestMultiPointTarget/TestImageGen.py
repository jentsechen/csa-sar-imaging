import csv
import numpy as np
import json
import subprocess
import scipy.io
import plotly.graph_objects as go

import matplotlib.pyplot as plt

if __name__ == "__main__":    
    data = np.load("./focused_image/multi_point_target_image_mag_db.npy")[int(556/2):int(556*7/2), int(556/2):int(556*7/2)]
    print(data.shape)
    # plt.imshow(data, origin='lower', cmap='viridis')
    plt.imshow(data, origin='lower', cmap='viridis', aspect=2.0)
    plt.colorbar()
    plt.savefig("coast_csa.png", dpi=300, bbox_inches="tight")
    # for i in range(5):
    #     data = np.load("./iter_result_multi_point_image/csa_out_iter_{}_mag_db.npy".format(i))[:, int(556/2):int(556*7/2)]
    #     plt.imshow(data, origin='lower', cmap='viridis', aspect=2.0)
    #     plt.colorbar()
    #     plt.savefig("coast_iter_{}.png".format(i), dpi=300, bbox_inches="tight")
    #     plt.clf()
    print("DONE")
