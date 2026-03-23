import boto3
from botocore import UNSIGNED
from botocore.config import Config
import json
import os
from tqdm import tqdm
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt

def download_tif(uuid):
    bucket_name = "umbra-open-data-catalog"
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    print(f"Searching for .tif files associated with: {uuid}...")
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix='sar-data/tasks/ship_detection_testdata/')

    found_files = 0
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            
            if uuid in key and key.lower().endswith(('.tif', '.tiff')):
                local_path = os.path.basename(key)
                file_size = obj['Size'] # Get total bytes for the progress bar
                
                # Create the progress bar
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=local_path) as pbar:
                    s3.download_file(
                        bucket_name, 
                        key, 
                        local_path,
                        Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
                    )
                
                found_files += 1

    if found_files == 0:
        print("No matching .tif files found.")
    else:
        print(f"\nDone! Downloaded {found_files} file(s).")

def read_tif(file_name, scene_name, row_start, row_len, col_start, col_len, threshold, is_bin):
    with rasterio.open(file_name) as dataset:
        # 1. Extract the actual scattering intensity/complex data
        # (SAR data is often 32-bit, which Pillow/OpenCV might truncate)
        sar_data = dataset.read(1) 
        
        # 2. Extract Geometry: The Affine transform
        # This gives you the spacing between pixels in meters or degrees
        transform = dataset.transform
        pixel_spacing_x = transform[0]
        pixel_spacing_y = -transform[4] # Vertical spacing
        
        # 3. Get the Coordinate Reference System (CRS)
        # Essential for calculating the distance between scattering points
        crs = dataset.crs

    print(f"Data Shape: {sar_data.shape}")
    print(f"Pixel Resolution: {pixel_spacing_x}m x {pixel_spacing_y}m")

    sar_data_slice = sar_data[row_start:(row_start+row_len), col_start:(col_start+col_len)]
    if is_bin ==True:
        _, sar_data_bin = cv2.threshold(sar_data_slice, threshold, 255, cv2.THRESH_BINARY)
        binary_pixel_matrix = (sar_data_bin / 255).astype(np.uint8)
        point_target_loc = []
        for i in range(len(binary_pixel_matrix)):
            point_target_loc.append([])
            for j in range(len(binary_pixel_matrix[i])):
                point_target_loc[i].append(int(binary_pixel_matrix[i][j]))
    else:
        point_target_loc = sar_data_slice.tolist()

    calc_num_zero(scene_name=scene_name, point_target_loc=point_target_loc)
    with open(scene_name+'.json', "w") as f:
        json.dump(point_target_loc, f)
    with open(scene_name+'.json', "r", encoding="UTF-8") as f:
        data = json.load(f)
    print(np.array(data).shape)

def plot_binary_image(scene_name):
    with open(scene_name+'.json', 'r') as file:
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
    plt.savefig(scene_name+".png", dpi=300, bbox_inches="tight")
    plt.clf()

def calc_num_zero(scene_name, point_target_loc):
    num_zero = sum(sublist.count(0) for sublist in point_target_loc)
    percentage = (num_zero / (len(point_target_loc)*len(point_target_loc[0]))) * 100
    print(f"Number of zeros: {num_zero} ({percentage}%)")

    flat_data = np.array(point_target_loc).flatten()
    counts, _ = np.histogram(flat_data, bins=256, range=(0, 256))
    plt.plot(counts)
    plt.xlabel('value')
    plt.ylabel('count')
    plt.grid()
    plt.savefig(scene_name + "_hist.png")
    plt.clf()

if __name__ == "__main__":
    # download_tif(uuid='c59dab96-3b16-456b-80fe-866f30a3cabe')
    # read_tif(file_name='2023-04-12-13-00-32_UMBRA-04_GEC.tif', scene_name='tsoying_naval_base', 
    #          row_start=3200, row_len=2000, col_start=3500, col_len=2000, threshold=40)
    read_tif(file_name='2023-04-12-13-00-32_UMBRA-04_GEC.tif', scene_name='tsoying_naval_base', 
             row_start=4050, row_len=640, col_start=4180, col_len=720, threshold=40, is_bin=False)
    plot_binary_image(scene_name='tsoying_naval_base')

    # download_tif(uuid='8cd3eeb0-22e2-42d7-969e-030826a3a0c6')
    # read_tif(file_name='2023-06-30-12-56-34_UMBRA-05_GEC.tif', scene_name='port_of_kaohsiung', 
    #          row_start=1500, row_len=10000, col_start=0, col_len=10000, threshold=40)
    # plot_binary_image(scene_name='port_of_kaohsiung')

    # with open('/home/phd/jtc/csa-sar-imaging/TestMultiPointTarget/point_target_location.json', 'r') as file:
    #     point_target_location = json.load(file)
    # print(np.array(point_target_location).shape)