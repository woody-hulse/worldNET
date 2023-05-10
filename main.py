import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial.distance import cdist

LAND_POLYGONS = gpd.read_file("./GSHHS_f_L3.shp")
LAND_MASK = LAND_POLYGONS.geometry.unary_union

# import matplotlib
# matplotlib.use("tkagg")
from matplotlib import pyplot as plt

import preprocessing_gsv as preprocessing
import models


def map_to_land(y_true, y_pred):
    """
    maps all predictions to nearest land coordinate using known geometries
    y_true      : true image coordinates
    y_pred      : predicted image coordinates
    """

    for i in range(y_pred.shape[0]):
        point = Point(y_pred[i, 1], y_pred[i, 0])
        if not point.within(LAND_MASK):
            for polygon in LAND_MASK.geoms:
                if polygon.intersects(point):
                    land_points = np.array(list(polygon.exterior.coords))
                    dists = cdist([point.coords[0]], land_points)
                    nearest_land_point = land_points[np.argmin(dists)]
                    y_pred[i] = np.array([nearest_land_point[1], nearest_land_point[0]])

    return y_true, y_pred


def print_results(models, test_data, test_labels, metrics):
    """
    prints the results of each model after training
    models      : list of models
    test_data   : model-specific test data
    test_labels : model-specific test labels
    metrics     : table evaluation metrics
    """

    table = []
    
    for model, data in zip(models, test_data):
        table.append([])
        for metric in metrics:
            y_true = test_labels[:data.shape[0]]
            y_pred = model.call(data)

            table[-1].append(metric(y_true, y_pred).numpy())
    
    table_df = pd.DataFrame(
        data=table, 
        index=[model.name for model in models], 
        columns=[metric.name for metric in metrics])
    
    print()
    print(table_df)
    print()


def train_model(model, train_data, train_labels, test_data=[], test_labels=[], epochs=10, batch_size=16, summary=False, compile=True):
    """
    train a model
    model       : model inheriting from tf.keras.Model
    train_data  :
    train_labels:
    test_data   :
    test_labels :
    epochs      : number of training epochs
    batch_size  : batch size for training
    summary     : print description of model
    compile     : False for pre-compiled models
    """

    print("\ntraining", model.name, "...")

    if compile:
        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=[])
        model.build(train_data.shape)
    if summary:
        model.summary()
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_data, test_labels))


def train_distribution_model(model, train_data, train_labels, test_data=[], test_labels=[], epochs=10, batch_size=16, summary=False, downsampling=4, verbose=1, compile=True):
    """
    [UNUSED] train a distribution model with mu, sigma outputs
    model       : model inheriting from tf.keras.Model
    train_data  :
    train_labels:
    test_data   :
    test_labels :
    epochs      : number of training epochs
    batch_size  : batch size for training
    summary     : print description of model
    downsampling: factor for reducing training samples
    verbose     : print characteristic
    compile     : False for pre-compiled models
    """

    print("\ntraining", model.name, "...")

    if compile:
        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=[])
        model.build(train_data.shape)
    if summary:
        model.summary()

    train_length = train_data.shape[0]
    test_length = test_data.shape[0]
    indices = np.arange(train_length)

    for epoch in range(epochs):
        np.random.shuffle(indices)
        train_loss = 0
        if verbose >= 1: print("epoch", epoch + 1, "/", epochs)

        for i in tqdm(range(0, train_length, batch_size * downsampling)):
            batch_indices = indices[i:i + batch_size]
            batch_data = train_data[batch_indices]
            batch_labels = train_labels[batch_indices]

            with tf.GradientTape() as tape:
                mu = model(batch_data)
                loss = model.loss.call(batch_labels, mu)
            
            train_loss += loss.numpy() / (train_length // batch_size)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if not len(test_data) == 0:
            if verbose >= 1:
                test_indices = np.arange(test_length)
                np.random.shuffle(test_indices)
                test_batch_indices = test_indices[:batch_size * 8]
                test_batch_data = test_data[test_batch_indices]
                test_batch_labels = test_labels[test_batch_indices]
                test_mu = model(test_batch_data)
                test_loss = model.loss.call(test_batch_labels, test_mu).numpy()
                # train_acc = models.DistanceAccuracy().call(train_labels[:batch_size], model(train_data[:batch_size])[0])
                test_acc = models.DistanceAccuracy().call(test_batch_labels, test_mu).numpy()
                print("training loss :", train_loss, "testing loss :", test_loss, "testing accuracy :", test_acc)

                if verbose >= 2:
                    test_mu_mu = tf.math.reduce_mean(test_mu, axis=0)
                    test_mu_std = tf.math.reduce_std(test_mu, axis=0)
                    test_labels_std = tf.math.reduce_std(test_batch_labels, axis=0)
                    print("testing mean :", test_mu_mu.numpy(), "testing std :", test_mu_std.numpy(), "labels std :", test_labels_std.numpy())
        else:
            if verbose >= 1: print("training loss :", train_loss)
    
    if verbose == 0:
        if not len(test_data) == 0:
            test_indices = np.arange(test_length)
            np.random.shuffle(test_indices)
            test_batch_indices = test_indices[:batch_size * 8]
            test_batch_data = test_data[test_batch_indices]
            test_batch_labels = test_labels[test_batch_indices]
            test_mu, test_sigma = model(test_batch_data)

            test_loss = model.loss.call(test_batch_labels, test_mu, test_sigma).numpy()
            print("training loss :", train_loss, "testing loss :", test_loss)
        else:
            print("training loss :", train_loss)


def main(save=True, load=False, train=True, load_model=False, save_model=True):

    """
    main function for training
    save        : save loaded images/features to local directory
    load        : load images/features to local directory
    load_model  : load models from weights.h5 file
    save_model  : save models to weights.h5 file
    """

    data_path = "data/"
    features_path = "features/"
    weights_path = "weights/"
    
    if load:
        images, labels, cities = preprocessing.load_data(data_path)
        images, labels, cities = preprocessing.shuffle_data(images, labels, cities)
        cities, city_labels = preprocessing.ohe_cities_labels(cities, np.copy(labels))
        print("\nloading features from", features_path, "...")
        features = np.load(features_path + "features.npy")
    else:
        images, labels, cities = preprocessing.load_random_data(num_per_city=400)
        preprocessing.plot_points([labels], "world_image.jpeg", density_map=True, normalize_points=True)
        images, labels, cities = preprocessing.shuffle_data(images, labels, cities)
        images, labels, cities = preprocessing.uniform_geographic_distribution(images, labels, cities, radius=40, maximum=400)
        preprocessing.plot_points([labels], "world_image.jpeg", density_map=True, normalize_points=True)
        features = preprocessing.pass_through_VGG(images)
    if save:
        print("\nsaving data to", data_path, "...")
        preprocessing.remove_files(data_path + "*")
        preprocessing.save_data(images, labels, cities, data_path)

        print("\nsaving features to", features_path, "...")
        np.save(features_path + "features", features)

    # data preprocessing
    labels = preprocessing.normalize_labels(labels)

    grouped_features, grouped_feature_labels, grouped_feature_cities = preprocessing.reshape_grouped_features(features, labels, cities)
    features, feature_labels, feature_cities = preprocessing.reshape_features(features, labels, cities)

    train_images, test_images = preprocessing.train_test_split(images)
    train_labels, test_labels = preprocessing.train_test_split(labels)
    train_cities, test_cities = preprocessing.train_test_split(cities)

    train_features, test_features = preprocessing.train_test_split(features)
    train_feature_labels, test_feature_labels = preprocessing.train_test_split(feature_labels)
    train_feature_cities, test_feature_cities = preprocessing.train_test_split(feature_cities)

    train_grouped_features, test_grouped_features = preprocessing.train_test_split(grouped_features)
    grouped_feature_labels = preprocessing.expand_and_group_feature_labels(labels)
    grouped_feature_cities = preprocessing.expand_and_group_feature_labels(cities)
    train_grouped_labels, test_grouped_labels = preprocessing.train_test_split(grouped_feature_labels)
    train_grouped_cities, test_grouped_cities = preprocessing.train_test_split(grouped_feature_cities)

    # initial model classifier
    city_model = models.VGGCityModel(input_shape=images.shape[1:], output_units=cities.shape[1], dropout=0.5)
    city_model.compile(optimizer=city_model.optimizer, loss=city_model.loss, metrics=["accuracy"])
    city_model.build(train_images.shape)
    if load_model: city_model.load_weights("city_model_weights.h5")
    city_model.summary()
    city_model.fit(train_images, train_cities, batch_size=32, epochs=10, validation_data=(test_images, test_cities))
    if save_model: city_model.save_weights("city_model_weights.h5")

    # worldNET interpolation head from model classifier
    worldNET_city = models.worldNETCity(city_model=city_model, city_labels=city_labels, output_units=cities.shape[1], units=64, layers=2)
    worldNET_city.compile(optimizer=worldNET_city.optimizer, loss=worldNET_city.loss, metrics=[])
    worldNET_city.build(train_images.shape)
    if load_model: worldNET_city.load_weights("worldNET_weights.h5")
    worldNET_city.summary()
    worldNET_city.fit(train_images, train_labels, batch_size=32, epochs=10, validation_data=(test_images, test_labels))
    if save_model: worldNET_city.save_weights("worldNET_weights.h5")

    # view example worldNET predictions
    y_pred = worldNET_city(test_images[:16])[0]
    y_true = test_labels[:16]
    y_true, y_pred = map_to_land(y_true, y_pred)
    preprocessing.plot_points([y_pred, y_true], "world_image.jpeg", colors=['b', 'r'])

    # control models
    naive_vgg_model = models.NaiveVGG(units=512, output_units=2, layers=2)
    train_model(naive_vgg_model, train_grouped_features, train_labels, test_grouped_features, test_labels, epochs=1, batch_size=16)
    mean_model = models.MeanModel(train_labels=train_labels, loss_fn=models.MeanHaversineDistanceLoss())
    guess_model = models.GuessModel(train_labels=train_labels, loss_fn=models.MeanHaversineDistanceLoss())
    randomized_guess_model = models.RandomizedGuessModel()

    # create table for final results
    print_results([mean_model, guess_model, randomized_guess_model, naive_vgg_model, worldNET_city], 
                  [test_images, test_images, test_images, test_images[:64], test_images[:64]], 
                  test_labels, metrics=[tf.keras.losses.MeanSquaredError(), models.MeanHaversineDistanceLoss(), models.DistanceAccuracy()])


if __name__ == "__main__":
    os.system("clear")
    main(save=False, load=True, train=True, load_model=True, save_model=True)