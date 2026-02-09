import csv
import numpy as np
import json
import subprocess
import scipy.io
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal
import json

def run_cpp(cmd):
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    print("\n--- C++ Program Output (STDOUT) ---")
    print(proc.stdout)

def plot_mf_out(image_line, center, bnd, xlabel, file_name):
    up_sample_factor = 100
    low_bnd = center-bnd
    up_bnd = center+bnd
    slice = image_line[low_bnd:up_bnd]
    slice_upsampling = signal.resample(slice, len(slice)*up_sample_factor)
    slice_db = 10*np.log10(abs(slice_upsampling)**2)
    plt.plot(np.linspace(low_bnd, up_bnd, len(slice)*up_sample_factor), slice_db - np.max(slice_db))
    plt.axhline(y=-13.26, color='r', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel('magnitude (dB)')
    plt.grid(True)
    plt.savefig(file_name)
    plt.clf()

if __name__ == "__main__":
    # run_cpp(cmd=["../build/linear_to_db_scale", "./focused_image/single_point_target"])
    # image = np.load("./focused_image/single_point_target_mag_db.npy")
    image = np.load("./focused_image/single_point_target.npy")
    print(image.shape)
    azimuth_center, range_center = int(image.shape[0]/2), int(image.shape[1]/2)
    row, col = np.unravel_index(np.argmax(abs(image)), image.shape)
    print(row, col)

    plot_mf_out(image_line=image[azimuth_center-40, :],
                center=range_center+13,
                bnd=20,
                xlabel="range",
                file_name="range_slice.png")
    plot_mf_out(image_line=image[:, range_center+13],
                center=azimuth_center-40,
                bnd=40,
                xlabel="azimuth",
                file_name="azimuth_slice.png")
    show_azi_bnd, show_rng_bnd = 1300, 640
    # image_db = 10*np.log10(abs(image)**2)
    image_db = 10*np.log10(abs(image[(azimuth_center-show_azi_bnd):(azimuth_center+show_azi_bnd), (range_center-show_rng_bnd):(range_center+show_rng_bnd)])**2)
    # max_val = np.max(image_db)
    # clipped_data = np.clip(image_db, max_val-50, max_val)
    # plt.imshow(clipped_data - np.max(clipped_data), 
    #            extent=[range_center-show_rng_bnd, range_center+show_rng_bnd, azimuth_center-show_azi_bnd, azimuth_center+show_azi_bnd],
    #            origin='lower', 
    #            cmap='viridis')
    plt.imshow(image_db - np.max(image_db), 
               extent=[range_center-show_rng_bnd, range_center+show_rng_bnd, azimuth_center-show_azi_bnd, azimuth_center+show_azi_bnd],
               origin='lower', 
               cmap='viridis')
    plt.xlabel('range')
    plt.ylabel('azimuth')
    plt.colorbar()
    plt.savefig("single_point_target.png", dpi=300, bbox_inches="tight")
    plt.clf()

    with open('./point_target_location.json', 'r') as file:
        point_target_location = json.load(file)
    point_target_location = np.array(point_target_location)
    point_target_location = np.flipud(point_target_location)
    print(point_target_location.shape)
    plt.imshow(point_target_location,
               origin='lower', 
               cmap='viridis')
    plt.xlabel('range')
    plt.ylabel('azimuth')
    plt.colorbar()
    plt.savefig("point_target_location.png", dpi=300, bbox_inches="tight")
    plt.clf()

    print("DONE")