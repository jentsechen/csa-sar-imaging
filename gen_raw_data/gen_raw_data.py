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
import cv2

def plot_binary_image():
    with open('../TestMultiPointTarget/point_target_location.json', 'r') as file:
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

def gen_point_target_location(file_name):
    img = cv2.imread(f"./{file_name}.png")
    ret, output1 = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
    binary_pixel_matrix = (output1 / 255).astype(np.uint8)
    binary_matrix = []
    for i in range(len(binary_pixel_matrix)):
        binary_matrix.append([])
        for j in range(len(binary_pixel_matrix[i])):
            binary_matrix[i].append(int(binary_pixel_matrix[i][j][0]))
        # binary_matrix[i].append(0)

    output_json_file = "../TestMultiPointTarget/point_target_location.json"
    with open(output_json_file, "w") as f:
        json.dump(binary_matrix, f)
    with open("../TestMultiPointTarget/point_target_location.json", "r", encoding="UTF-8") as f:
        data = json.load(f)
    print(len(data), len(data[0]))

if __name__ == "__main__":
    # gen_point_target_location(file_name="image_coast")
    # gen_point_target_location(file_name="image_island")
    plot_binary_image()
    print("DONE")