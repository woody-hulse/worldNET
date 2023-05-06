import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import scipy
import glob

import matplotlib
matplotlib.use("tkagg")
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


def plot_points(points_list, image_path, colors=['r'], density_map=False, normalize_points=False):
    """
    plots points over an image
    """

    print("plotting points ...")

    if normalize_points:
        normalized_points_list = []
        for points in points_list:
            normalized_points_list.append(normalize_labels(points))
        points_list = normalized_points_list

    with Image.open(image_path) as image:
        plt.imshow(image, origin="upper")
        image = np.array(image)
        width, height, _ = image.shape
        for points, color in zip(points_list, colors):
            x, y = points[:, 1] * width, height - points[:, 0] * height
            if density_map and len(points_list) == 1:
                plot_density_map(np.array([x, y]).T, image, r=50, grad=10)
            else:
                plt.scatter(x, y, c=color)
        plt.grid(False)
        plt.axis('off')
        plt.xlim(0, image.shape[1])
        plt.ylim(image.shape[0], 0)
        plt.show()


def plot_density_map(points, image, r, grad=5):
    x_min, y_min = 0, 0
    x_max, y_max = image.shape[0], image.shape[1]
    n_bins = 500
    x_bins = np.linspace(x_min, x_max, n_bins)
    y_bins = np.linspace(y_min, y_max, n_bins)
    xx, yy = np.meshgrid(x_bins, y_bins)

    density = np.zeros((n_bins, n_bins))
    kdtree = scipy.spatial.cKDTree(points)
    for i in tqdm(range(n_bins)):
        for j in range(n_bins):
            for g in range(1, grad):
                indices = kdtree.query_ball_point([xx[i, j], yy[i, j]], r * (g / grad))
                density[i, j] += len(indices)

    density = density / np.max(density)
    heatmap = np.empty((n_bins, n_bins, 4))
    heatmap[:] = np.array([1, 0, 0, 0])
    heatmap[:, :, 3] = density

    plt.imshow(heatmap, extent=[x_min, x_max, y_min, y_max], cmap="hot", origin="lower", alpha=0.8)
    plt.colorbar()


def uniform_geographic_distribution(images, labels, cities, radius=10, maximum=100):
    """
    remove image/label/cities from dense areas
    """

    print("\nfiltering", len(images), "images for uniform distribution ...")

    print_city_distribution(cities, cities)

    num_images = len(images)
    kdtree = scipy.spatial.cKDTree(labels)
    remove_indices = set()
    remove_labels = [[0, 0]]
    remove_kdtree = scipy.spatial.cKDTree(remove_labels)
    with tqdm(total=len(images)) as pbar:
        for i, label in enumerate(labels):
            indices = kdtree.query_ball_point(label, radius)
            indices_ = remove_kdtree.query_ball_point(label, radius)
            if len(indices) - len(indices_) > maximum:
                remove_indices.add(i)
                remove_labels.append(label)
                remove_kdtree = scipy.spatial.cKDTree(remove_labels)
            pbar.update(1)

    new_images, new_labels, new_cities = [], [], []
    for i in tqdm(range(num_images)):
        if i not in remove_indices:
            new_images.append(images[i])
            new_labels.append(labels[i])
            new_cities.append(cities[i])
    new_images = np.stack(new_images)
    new_labels = np.stack(new_labels)
    new_cities = np.stack(new_cities)

    print(num_images, "->", len(new_images))
    print_city_distribution(cities, new_cities)

    return new_images, new_labels, new_cities


def print_city_distribution(index_cities, cities):
    """
    prints table of images in each city
    """
    index_cities = list(set(index_cities))
    cities = list(cities)
    counts = [cities.count(index_city) for index_city in index_cities]

    df = pd.DataFrame(counts, index=index_cities, columns=["count"])
    print()
    print(df)
    print()


def group_feature_labels(labels, num_features=512):
    """
    groups every 512 feature labels
    """

    return labels.reshape(-1, num_features, labels.shape[1])


def expand_and_group_feature_labels(labels, num_features=512):
    """
    expands and groups feature labels
    """

    grouped_labels = []
    for label in labels:
        grouped_labels.append([label] * num_features)

    return np.array(grouped_labels)


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


def pass_through_VGG(images):
    """
    passes input through vgg
    """

    print("\nloading vgg ...")

    vgg = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=images.shape[1:],
        pooling=None,
    )

    print("\npassing image data through vgg ...")

    features = vgg.predict(images)

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