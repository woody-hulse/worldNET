"""

The dataset contains 62,058 high quality Google Street View images. The images cover the
downtown and neighboring areas of Pittsburgh, PA; Orlando, FL and partially Manhattan, NY.
Accurate GPS coordinates of the images and their compass direction are provided as well.
For each Street View placemark (i.e. each spot on one street), the 360° spherical view is broken
down into 4 side views and 1 upward view. There is one additional image per placemark which
shows some overlaid markers, such as the address, name of streets, etc.

Naming format:
The name of the images has the following format: XXXXXX_Y.jpg
XXXXXX is the identifier of the placemark. There are total number of 10343 placemarks in this
dataset, so XXXXXX ranges from 000001 to 10343.
Y is the identifier of the view. 1, 2, 3 and 4 are the side views and 5 is the upward view. 0 is the
view with markers overlaid (explained above). Thus, there are total number of 6 images per
placemark.

GPS Coordinates & Compass Direction:
The Matlab file 'GPS_Long_Lat_Compass.mat' includes the GPS coordinates and compass
direction of each placemark. The row number XXXXXX corresponds to the placemark number
XXXXXX. The 1st and 2nd columns are the latitude and longitude values. The 3rd column is the
compass direction (in degrees from North towards West) of the view number 4. The rest of the
side views are exactly 90° apart from the view number 4.
The file 'Cartesian_Location_Coordinates.mat' contains the location coordinates in a metric
Cartesian system (unlike longitude and latitude). The Euclidean distance between such XYZ
coordinates of two points is the actual distance (in meters) between them.

Image Geo-localization based on Multiple Nearest Neighbor Feature Matching using
Generalized Graphs. Amir Roshan Zamir and Mubarak Shah. IEEE Transactions on
Pattern Analysis and Machine Intelligence (TPAMI), 2014.

"""


"""

Instructions:

-   create a data/ directory outside of worldNET directory

-   from http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/:
-       get Cartesian_Location_Coordinates.mat
-       get all parts from zipped_images/ and add image contents to an images/ subdirectory
-       run get_coordinates() to retrieve all XYZ coordinates, indexed in order
-       run load_images() to load in all images
"""


import os
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

import matplotlib
matplotlib.use("tkagg")
from matplotlib import pyplot as plt


DATA_PATH = "../data/"
NUM_SAMPLES = 500

def get_coordinates():
    """
    retrieves the coordinates of all images
    """
    filename = "Cartesian_Location_Coordinates.mat"
    print("loading", NUM_SAMPLES, "coordinates from", DATA_PATH + filename)
    mat = loadmat(DATA_PATH + filename)
    return mat["XYZ_Cartesian"][:NUM_SAMPLES]


def load_image(filename, target_shape):
    """
    load a single image
    """
    image = Image.open(filename).resize(target_shape)
    return np.asarray(image) / 255

def load_images(image_dir, target_shape=(224, 224), image_arr_filename="image_arr", save=True, load=True):
    """
    load and save all available images
    """

    if load:
        if os.path.exists(DATA_PATH + image_arr_filename + ".npy"):
            return np.load(DATA_PATH + image_arr_filename + ".npy")

    num_angles = 4
    files = os.listdir(DATA_PATH + image_dir)
    files.sort()
    if ".DS_Store" in files: files.remove('.DS_Store')
    files = files[:NUM_SAMPLES * 6]

    print("loading", NUM_SAMPLES, "images from", DATA_PATH + image_dir)
    images = np.empty((NUM_SAMPLES, num_angles, target_shape[0], target_shape[1], 3))

    indices, angles = [], []
    filepaths = []
    for file in files:
        index, angle = map(int, file.split('.')[0].split('_'))
        index, angle = index - 1, angle - 1
        if angle == -1 or angle == 4: continue
        indices.append(index)
        angles.append(angle)
        filepaths.append(DATA_PATH + image_dir + file)

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(load_image)(filepath, target_shape) for filepath in tqdm(filepaths))

    for index, angle, result in zip(indices, angles, results):
        images[index, angle] = result

    if save:
        np.save(DATA_PATH + image_arr_filename, images)
        print("image array saved at", DATA_PATH + image_arr_filename)
    
    return images


def main():
    images = load_images("images/")
    coords = get_coordinates()

    print(coords.shape, images.shape)

if __name__ == "__main__":
    main()

