import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


"""

data from   : https://www.kaggle.com/datasets/amaralibey/gsv-cities
paper       : https://arxiv.org/pdf/2210.10239.pdf

cite        : 

@article{ali2022gsv,
title={GSV-Cities: Toward appropriate supervised visual place recognition},
author={Ali-bey, Amar and Chaib-draa, Brahim and Gigu{\`e}re, Philippe},
journal={Neurocomputing},
volume={513},
pages={194--203},
year={2022},
publisher={Elsevier}
}

"""


DATA_PATH = "../data/archive/"


def pass_through_VGG(data):
    vgg = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=data.shape[1:],
        pooling=None,
    )
    
    return vgg(data)


def train_test_split(data, prop=16/23):
    """
    splits training and testing data
    """

    train_samples = int(len(data) * prop)
    return data[:train_samples], data[train_samples:]


def load_random_data(image_path="Images/", num_per_city=100):
    """
    loads random data
    """

    print("compiling randomized image paths ...")

    paths = []
    cities = os.listdir(DATA_PATH + image_path)
    random.shuffle(cities)
    if ".DS_Store" in cities:
        cities.remove(".DS_Store")
    for city in cities:
        image_files = os.listdir(DATA_PATH + image_path + city)
        random.shuffle(image_files)
        for image_file in image_files[:num_per_city]:
            paths.append(DATA_PATH + image_path + city + "/" + image_file)

    return load_data_from_paths(paths)


def load_data_from_paths(paths):
    """
    loads data from specified paths
    """

    print("loading images from paths ...")

    images = []
    labels = []
    cities = []
    for path in tqdm(paths):
        info = list(path.split('/')[-1].split('_'))[:8]
        city, _, year, month, _, lat, lon, _ = info
        with Image.open(path) as image:
            images.append(np.array(image.resize((300, 400))))
            labels.append(np.array([float(lat), float(lon)]))
            cities.append(city)
    images = np.stack(images)
    labels = np.stack(labels)
    cities = np.array(cities)
    
    return images, labels, cities


def load_images(image_path):
    """
    loads all images (not recommended)
    """

    print("loading all images ...")

    cities = os.listdir(DATA_PATH + image_path)
    if ".DS_Store" in cities:
        cities.remove(".DS_Store")

    total = 0
    for city in cities:
        total += len(os.listdir(DATA_PATH + image_path + city))

    data = []

    with tqdm(total=total) as pbar:
        for city in cities:
            image_files = os.listdir(DATA_PATH + image_path + city)
            for image_file in image_files:
                info = list(image_file.split('_'))[:8]
                city, _, year, month, _, lat, lon, _ = info
                with Image.open(DATA_PATH + image_path + city + "/" + image_file) as image:
                    data.append([np.array(image), lat, lon, city])
                pbar.update(1)
    
    return data



def load_data(data_path):
    """
    loads saved data
    """

    print("loading data from", data_path, "...")

    images = []
    labels = []
    cities = []
    files = os.listdir(data_path)
    for file in tqdm(files):
        info = list(file.split('.jpg')[0].split('_'))[:3]
        city, lat, lon = info
        with Image.open(data_path + file) as image:
            images.append(np.array(image.resize((300, 400))))
            labels.append(np.array([float(lat), float(lon)]))
            cities.append(city)
    images = np.stack(images)
    labels = np.stack(labels)
    cities = np.array(cities)
    
    return images, labels, cities

def save_data(images, labels, cities, data_path):
    """
    saves data
    """

    print("saving data to", data_path, "...")
    
    for image, label, city in tqdm(zip(images, labels, cities)):
        lat, lon = label
        image = Image.fromarray(image)
        filepath = data_path + city + "_" + str(lat) + "_" + str(lon) + ".jpg"
        image.save(fp=filepath)
    images = np.stack(images)
    labels = np.stack(labels)
    cities = np.array(cities)
    
    return images, labels, cities



def main():
    paths = load_random_data("Images/")
    images, labels, cities = load_data_from_paths(paths)

if __name__ == "__main__":
    main()