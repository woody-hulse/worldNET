import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

import preprocessing_gsv as preprocessing
import models


def print_results(models, test_data, test_labels, metrics):
    """
    prints the results of each model after training
    """

    table = []
    
    for model in models:
        table.append([])
        for metric in metrics:
            table[-1].append(metric(test_labels, model.call(test_data)).numpy())
    
    table_df = pd.DataFrame(
        data=table, 
        index=[model.name for model in models], 
        columns=[metric.name for metric in metrics])
    
    print()
    print(table_df)
    print()


def train_model(model, train_data, train_labels, test_data=[], test_labels=[], epochs=10, batch_size=16, summary=False):
    """
    train a model
    """

    print("\ntraining", model.name, "...")

    model.compile(optimizer=model.optimizer, loss=model.loss, metrics=[])
    model.build(train_data.shape)
    if summary:
        model.summary()
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_data, test_labels))


def train_distribution_model(model, train_data, train_labels, test_data=[], test_labels=[], epochs=10, batch_size=16, summary=False, downsampling=4, verbose=1):
    """
    train a distribution model with mu, sigma outputs
    """

    print("\ntraining", model.name, "...")

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
        if verbose >= 1: print("epoch", epoch + 1)

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
                print("training loss :", train_loss, "testing loss :", test_loss)

                if verbose >= 2:
                    test_mu_mu = tf.math.reduce_mean(test_mu, axis=0)
                    test_mu_var = tf.math.reduce_variance(test_mu, axis=0)
                    print("testing mean :", test_mu_mu.numpy(), "testing variance :", test_mu_var.numpy())
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



def main(save=True, load=False, use_features=True):

    data_path = "data/"
    features_path = "features/"
    
    if load:
        images, labels, cities = preprocessing.load_data(data_path)
    else:
        images, labels, cities = preprocessing.load_random_data()
    if use_features:
        if load:
            print("\nloading features from", features_path, "...")
            features = np.load(features_path + "features.npy")
        else:
            features = preprocessing.pass_through_VGG(images)
            if save:
                print("\nsaving features to", features_path, "...")
                np.save(features_path + "features", features)
    if save:
        preprocessing.save_data(images, labels, cities, data_path)
    if use_features:
        features, feature_labels, feature_cities = preprocessing.reshape_features(features, labels, cities)

    labels = preprocessing.normalize_labels(labels)

    train_images, test_images = preprocessing.train_test_split(images)
    train_labels, test_labels = preprocessing.train_test_split(labels)
    train_cities, test_cities = preprocessing.train_test_split(cities)

    if use_features:
        train_features, test_features = preprocessing.train_test_split(features)
        train_feature_labels, test_feature_labels = preprocessing.train_test_split(feature_labels)
        train_feature_cities, test_feature_cities = preprocessing.train_test_split(feature_cities)

        feature_model = models.FeatureDistributionModel(hidden_size=32)
        train_distribution_model(feature_model.feature_distribution_nn, train_features, train_feature_labels, test_features, test_feature_labels, epochs=24, batch_size=128, downsampling=16, verbose=2, summary=True)

    simple_nn_model = models.SimpleNN(output_units=2, name="simple_nn")
    train_model(simple_nn_model, train_images, train_labels, test_images, test_labels, epochs=4, batch_size=16)

    print("\nnaive vgg ...")

    naive_vgg_model = models.NaiveVGG(input_shape=images.shape[1:], units=8, output_units=2, layers=1)
    # train_model(naive_vgg_model, train_images, train_labels, test_images, test_labels, epochs=4, batch_size=16)

    # nearest_neighbors_model = models.FeatureNearestNeighbors(input_shape=images.shape[1:])
    # train_model(nearest_neighbors_model, train_images, train_labels, batch_size=16, epochs=4, validation_data=(test_images, test_labels))

    mean_model = models.MeanModel(train_labels=train_labels, loss_fn=models.MeanHaversineDistanceLoss())
    guess_model = models.GuessModel(train_labels=train_labels, loss_fn=models.MeanHaversineDistanceLoss())

    print_results([simple_nn_model, mean_model, guess_model], test_images, test_labels, 
                  metrics=[tf.keras.losses.MeanSquaredError(), models.MeanHaversineDistanceLoss()])


if __name__ == "__main__":
    os.system("clear")
    main(save=False, load=True, use_features=True)