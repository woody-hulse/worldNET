import os
import tensorflow as tf
import numpy as np
import pandas as pd

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



def main():
    
    images, labels, cities = preprocessing.load_random_data()

    train_images, test_images = preprocessing.train_test_split(images)
    train_labels, test_labels = preprocessing.train_test_split(labels)
    train_cities, test_cities = preprocessing.train_test_split(cities)

    print("\ntraining naive models ...")

    print("\nsimple nn (mse) ...")

    simple_nn_model_mse = models.SimpleNN(output_units=2, name="simple_nn_mse")
    simple_nn_model_mse.compile(optimizer=simple_nn_model_mse.optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=[])
    simple_nn_model_mse.build(images.shape)
    simple_nn_model_mse.fit(train_images, train_labels, batch_size=16, epochs=4, validation_data=(test_images, test_labels))

    print("\nsimple nn (haversine) ...")

    simple_nn_model_haversine = models.SimpleNN(output_units=2, name="simple_nn_haversine")
    simple_nn_model_haversine.compile(optimizer=simple_nn_model_haversine.optimizer, loss=models.MeanHaversineDistanceLoss(), metrics=[])
    simple_nn_model_haversine.build(images.shape)
    simple_nn_model_haversine.fit(train_images, train_labels, batch_size=16, epochs=4, validation_data=(test_images, test_labels))

    print("\nnaive vgg ...")
    
    naive_vgg_model = models.NaiveVGG(input_shape=images.shape[1:], units=32, output_units=2)
    naive_vgg_model.compile(
        optimizer=naive_vgg_model.optimizer,
        loss=naive_vgg_model.loss,
        metrics=[],
    )
    naive_vgg_model.build(images.shape)
    naive_vgg_model.summary()
    # model.fit(train_images, train_labels, batch_size=64, epochs=4, validation_data=(test_images, test_labels))

    mean_model = models.MeanModel(train_labels=train_labels, loss_fn=naive_vgg_model.loss)
    guess_model = models.GuessModel(train_labels=train_labels, loss_fn=naive_vgg_model.loss)

    print_results([simple_nn_model_mse, simple_nn_model_haversine, mean_model, guess_model], test_images, test_labels, 
                  metrics=[tf.keras.losses.MeanSquaredError(), models.MeanHaversineDistanceLoss()])


if __name__ == "__main__":
    os.system("clear")
    main()