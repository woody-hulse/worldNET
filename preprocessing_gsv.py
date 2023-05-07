# preprocessing_gsv.py
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import glob

import matplotlib
from matplotlib import pyplot as plt

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
IMAGE_SHAPE = (300, 400)


def group_feature_labels(labels, num_features=512):
    """
    groups every 512 feature labels
    """

    return labels.reshape(-1, num_features, labels.shape[1])


def shuffle_data(images, labels, cities):
    """
    randomly shuffles data
    """
    
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    return images[indices], labels[indices], cities[indices]


def remove_files(filepath):
    """
    removes all files from directory
    """

    files = glob.glob(filepath)
    for f in files:
        os.remove(f)


def plot_features(features, num):
    """
    plots some number of features
    """


    indices = np.arange(len(features))
    np.random.shuffle(indices)

    for i in indices[:num]:
        plt.imshow(features[i].reshape(9, 12))
        plt.show()


def normalize_labels(labels):
    """
    normalize degree angle of labels
    """

    nomarlized_labels = (labels + np.array([[90, 180]])) / np.array([[180, 360]])

    return nomarlized_labels


def unnormalize_labels(labels):
    """
    unnormalize degree angles of labels
    """

    unnormalized_labels = labels * np.array([[180, 360]]) - np.array([[90, 180]])

    return unnormalized_labels

def get_layer_model(vgg, layer):
  layer = vgg.get_layer(layer)
  return tf.keras.models.Model(
    inputs=[vgg.input],
    outputs=[layer.output]
  )


def pass_through_VGG(images):
    """
    passes input through vgg
    """

    print("\nloading vgg ...")

    images = tf.image.resize(images, [images.shape[1] // 2, images.shape[2] // 2])

    vgg = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=images.shape[1:],
        pooling=None,
    )

    vgg_layer = get_layer_model(vgg, "block4_conv2")

    print("\npassing image data through vgg ...")

    features = vgg_layer.predict(images)

    print(features.shape)

    return features


def reshape_features(image_features, labels, cities):
    """
    reformats feature data to link individual features to labels
    """
    feature_vectors = []
    feature_labels = []
    feature_cities = []

    print("\nreshaping features ...")
    image_features = np.transpose(image_features, axes=[0, 3, 1, 2])
    with tqdm(total=image_features.shape[0] * image_features.shape[1]) as pbar:
        for features, label, city in zip(image_features, labels, cities):
            for feature in features:
                feature_vectors.append(feature.flatten())
                feature_labels.append(label)
                feature_cities.append(city)
                pbar.update(1)

    feature_vectors = np.stack(feature_vectors)
    feature_labels = np.stack(feature_labels)
    feature_cities = np.array(feature_cities)

    return feature_vectors, feature_labels, feature_cities


def reshape_grouped_features(image_features, labels, cities):
    """
    transpose features
    """
    image_features = np.transpose(image_features, axes=[0, 3, 1, 2])
    image_features = image_features.reshape((image_features.shape[0], image_features.shape[1], -1))
    return image_features, labels, cities

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
            images.append(np.array(image.resize(IMAGE_SHAPE)))
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
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    for file in tqdm(files):
        info = list(file.split('.jpg')[0].split('_'))[:3]
        city, lat, lon = info
        with Image.open(data_path + file) as image:
            images.append(np.array(image))
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
    
    with tqdm(total=len(images) + 1) as pbar:
        for image, label, city in zip(images, labels, cities):
            lat, lon = label
            image = Image.fromarray(image)
            filepath = data_path + city + "_" + str(lat) + "_" + str(lon) + ".jpg"
            image.save(fp=filepath)
            pbar.update(1)

    images = np.stack(images)
    labels = np.stack(labels)
    cities = np.array(cities)
    
    return images, labels, cities



def main():
    paths = load_random_data("Images/")
    images, labels, cities = load_data_from_paths(paths)

if __name__ == "__main__":
    main()