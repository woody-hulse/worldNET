import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("tkagg")
from matplotlib import pyplot as plt

import preprocessing_gsv as preprocessing
import models


def print_results(models, test_data, test_labels, metrics):
    """
    prints the results of each model after training
    """

    table = []
    
    for model, data in zip(models, test_data):
        table.append([])
        for metric in metrics:
            table[-1].append(metric(test_labels[:data.shape[0]], model.call(data)).numpy())
    
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
    train a distribution model with mu, sigma outputs
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
                mu, sigma = model(batch_data)
                loss = model.loss.call(batch_labels, mu, sigma)
            
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
                test_mu, test_sigma = model(test_batch_data)
                test_loss = model.loss.call(test_batch_labels, test_mu, test_sigma).numpy()
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

    data_path = "data/"
    features_path = "features/"
    weights_path = "weights/"
    
    if load:
        images, labels, cities = preprocessing.load_data(data_path)
        # print("\nloading features from", features_path, "...")
        # features = np.load(features_path + "features.npy")
    else:
        images, labels, cities = preprocessing.load_random_data(num_per_city=200)
        images, labels, cities = preprocessing.shuffle_data(images, labels, cities)
        images, labels, cities = preprocessing.uniform_geographic_distribution(images, labels, cities, radius=10, maximum=200)
        # features = preprocessing.pass_through_VGG(images)
        # features = np.load(features_path + "features.npy")
    if save:
        print("\nsaving data to", data_path, "...")
        preprocessing.remove_files(data_path + "*")
        preprocessing.save_data(images, labels, cities, data_path)

        # print("\nsaving features to", features_path, "...")
        # np.save(features_path + "features", features)

    # images, labels, cities = preprocessing.shuffle_data(images, labels, cities)
    # features, feature_labels, feature_cities = preprocessing.shuffle_data(features, labels, cities)

    # grouped_features, grouped_feature_labels, grouped_feature_cities = preprocessing.reshape_grouped_features(features, feature_labels, feature_cities)
    # features, feature_labels, feature_cities = preprocessing.reshape_features(features, feature_labels, feature_cities)

    # feature_labels = preprocessing.normalize_labels(feature_labels)

    # grouped_feature_labels = preprocessing.group_feature_labels(feature_labels)
    # features, feature_labels, feature_cities = preprocessing.shuffle_data(features, feature_labels, feature_cities)

    # preprocessing.plot_features(features, 6)

    labels = preprocessing.normalize_labels(labels)

    train_images, test_images = preprocessing.train_test_split(images)
    train_labels, test_labels = preprocessing.train_test_split(labels)
    train_cities, test_cities = preprocessing.train_test_split(cities)

    # train_features, test_features = preprocessing.train_test_split(features)
    # train_feature_labels, test_feature_labels = preprocessing.train_test_split(feature_labels)
    # train_feature_cities, test_feature_cities = preprocessing.train_test_split(feature_cities)

    # train_grouped_features, test_grouped_features = preprocessing.train_test_split(grouped_features)
    grouped_feature_labels = preprocessing.expand_and_group_feature_labels(labels)
    train_grouped_labels, test_grouped_labels = preprocessing.train_test_split(grouped_feature_labels)
    # train_grouped_cities, test_grouped_cities = preprocessing.train_test_split(grouped_feature_cities)

    """
    plt.scatter(train_labels[:, 0], train_labels[:, 1])
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    for i, txt in enumerate(train_cities):
        plt.annotate(txt, (train_labels[i, 0], train_labels[i, 1]))

    plt.show()
    """

    # nearest_neighbors_model = models.FeatureNearestNeighbors(input_shape=images.shape[1:])
    # lat, long, std = nearest_neighbors_model.nearest_neighbor_classify(train_features[:10000], train_feature_labels[:10000], test_features[:100])
    # print(models.MeanHaversineDistanceLoss()(test_feature_labels[:100], np.stack([lat, long]).T))

    """
    test_model = models.createInceptionModel(input_shape=images.shape[1:])
    test_model_weights_path = weights_path + "test.h5"
    test_model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=models.MeanHaversineDistanceLoss(), metrics=[])
    test_model.build(train_images.shape)
    if load_model:
        print("\nloading test model from", test_model_weights_path, "...")
        test_model.load_weights(test_model_weights_path)
    if train:
        train_model(test_model, train_images, train_labels, test_images, test_labels, epochs=1, batch_size=64, summary=True, compile=False)
    if save_model:
        print("\nsaving test model to", test_model_weights_path, "...")
        test_model.save_weights(test_model_weights_path)
    """

    worldNET = models.worldNET(hidden_size=64, num_layers=3, input_shape=images.shape[1:])
    worldNET_weights_path = weights_path + "worldNET.h5"
    worldNET.feature_distribution_model.compile(optimizer=worldNET.feature_distribution_model.optimizer, loss=worldNET.feature_distribution_model.loss, metrics=[])
    worldNET.feature_distribution_model.build(train_images.shape)
    if load_model:
        print("\nloading worldNET model from", worldNET_weights_path, "...")
        worldNET.feature_distribution_model.load_weights(worldNET_weights_path)
    if train:
        train_model(worldNET.feature_distribution_model, train_images, train_grouped_labels, test_images, test_grouped_labels, epochs=1, batch_size=64, summary=True, compile=False)
    if save_model:
        print("\nsaving worldNET model to", worldNET_weights_path, "...")
        worldNET.feature_distribution_model.save_weights(worldNET_weights_path)

    vgg_feature_model = models.VGGFeatureDistributionModel(input_shape=images.shape[1:], hidden_size=16, num_layers=3)
    train_model(vgg_feature_model, train_images[:1000], train_grouped_labels[:1000], test_images, test_grouped_labels, epochs=4, batch_size=256, summary=True)

    # feature_model = models.FeatureDistributionModel(hidden_size=64, num_layers=4)
    # train_distribution_model(feature_model.feature_distribution_nn, train_features, train_feature_labels, test_features, test_feature_labels, epochs=4, batch_size=128, downsampling=32, verbose=2, summary=True)


    """
    pred_mu, pred_sigma = feature_model.feature_distribution_nn(train_features[:64])
    true = train_feature_labels[:64]

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    for i in range(64):
        plt.plot([pred_mu[i, 1], true[i, 1]], [pred_mu[i, 0], true[i, 0]], c='#000000', linewidth='0.5')
    plt.scatter(pred_mu[:, 1], pred_mu[:, 0], c='b')
    plt.scatter(true[:, 1], true[:, 0], c='r')
    plt.show()
    """

    naive_vgg_model = models.NaiveVGG(units=8, output_units=2, layers=1, input_shape=images.shape[1:])
    train_model(naive_vgg_model, train_images, train_labels, test_images, test_labels, epochs=1, batch_size=32)

    simple_nn_model = models.SimpleNN(output_units=2, name="simple_nn")
    train_model(simple_nn_model, train_images, train_labels, test_images, test_labels, epochs=4, batch_size=32)

    # nearest_neighbors_model = models.FeatureNearestNeighbors(input_shape=images.shape[1:])
    # train_model(nearest_neighbors_model, train_images, train_labels, batch_size=16, epochs=4, validation_data=(test_images, test_labels))

    mean_model = models.MeanModel(train_labels=train_labels, loss_fn=models.MeanHaversineDistanceLoss())
    guess_model = models.GuessModel(train_labels=train_labels, loss_fn=models.MeanHaversineDistanceLoss())
    randomized_guess_model = models.RandomizedGuessModel()

    print_results([simple_nn_model, mean_model, guess_model, randomized_guess_model, naive_vgg_model, worldNET], 
                  [test_images, test_images, test_images, test_images, test_images, test_images[:64]], 
                  test_labels, metrics=[tf.keras.losses.MeanSquaredError(), models.MeanHaversineDistanceLoss(), models.DistanceAccuracy()])


if __name__ == "__main__":
    os.system("clear")
    main(save=False, load=False, train=False, load_model=False, save_model=True)