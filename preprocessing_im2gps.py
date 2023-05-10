import os
import tensorflow as tf
import numpy as np
from struct import unpack
from scipy.io import loadmat
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

import matplotlib
matplotlib.use("tkagg")
from matplotlib import pyplot as plt

"""

*********** [UNUSED] ***********

"""

DATA_PATH = "../data/"
NUM_SAMPLES = 500
IMAGE_PATH = DATA_PATH + "gps_query_imgs/"


def get_image_coordinates():
    files = os.listdir(IMAGE_PATH)
    if ".DS_Store" in files: files.remove('.DS_Store')
    files.sort()
    files = files[:NUM_SAMPLES]

    for file in files:
        img = Image.open(IMAGE_PATH + file)
        exif = { ExifTags.TAGS[k]: v for k, v in img.getexif().items() if k in ExifTags.TAGS }
        print(exif)


def load_image(filepath, target_shape):
    image = Image.open(filepath).resize(target_shape)
    return np.asarray(image) / 255

def load_images(target_shape=(224, 224), save=True, load=True, arr_filename="image_arr"):

    if load:
        try:
            images = np.load(DATA_PATH + arr_filename + ".npy")
            print("loaded", images.shape[0], "images from", DATA_PATH + arr_filename + ".npy")
            return images
        except:
            pass

    files = os.listdir(IMAGE_PATH)
    if ".DS_Store" in files: files.remove('.DS_Store')
    files.sort()
    files = files[:NUM_SAMPLES]
    filepaths = [IMAGE_PATH + file for file in files]
    
    print("loading", len(files), "images from", IMAGE_PATH)

    images = np.empty((len(files), target_shape[0], target_shape[1], 3))

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(load_image)(filepath, target_shape) for filepath in tqdm(filepaths))

    for i, result in enumerate(results):
        images[i] = result

    if save:
        np.save(DATA_PATH + arr_filename, images)
        print("images saved to", DATA_PATH + arr_filename + ".npy")

    return images


def main():
    # load_images()

    get_image_coordinates()

if __name__ == "__main__":
    main()