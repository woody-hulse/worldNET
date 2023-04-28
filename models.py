import tensorflow as tf
import numpy as np
import random

def radians(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180


class MeanHaversineDistanceLoss(tf.keras.losses.Loss):
    def __init__(self, name="mean_haversine_distance_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        earth_radius = 6371000
        lat1, lon1 = tf.unstack(y_true, axis=-1)
        lat2, lon2 = tf.unstack(y_pred, axis=-1)

        lat1_rad = tf.cast(lat1 * np.pi / 180, tf.float32)
        lon1_rad = tf.cast(lon1 * np.pi / 180, tf.float32)
        lat2_rad = tf.cast(lat2 * np.pi / 180, tf.float32)
        lon2_rad = tf.cast(lon2 * np.pi / 180, tf.float32)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = tf.square(tf.sin(dlat / 2)) + tf.cos(lat1_rad) * tf.cos(lat2_rad) * tf.square(tf.sin(dlon / 2))
        c = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a))
        distance = earth_radius * c

        mean_distance = tf.reduce_mean(distance)

        return mean_distance


class NaiveVGG(tf.keras.Model):
    def __init__(self, 
            units		= 512, 				# number of units in each dense layer
            layers		= 2, 				# number of dense layers
            dropout		= 0.2, 				# dropout proportion per dense layer
            input_shape	= (224, 224, 3), 	# input shape
            output_units= 2, 				# number of units in output dense layer
            freeze_vgg	= True, 			# freeze vgg weights
            name		= "naive_vgg"		# name of model
        ):		
        
        super().__init__()

        self.vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )
        self.flatten_layer = tf.keras.layers.Flatten(name=f"{name}_flatten")
        self.head = [
            (tf.keras.layers.Dense(units, activation="relu", name=f"{name}_dense_{i}"), \
            tf.keras.layers.Dropout(dropout, name = f"{name}_dropout_{i}")) for i in range(layers)
        ]
        self.output_layer = tf.keras.layers.Dense(output_units, name = f"{name}_output_dense")

        if freeze_vgg:
            self.vgg.trainable = False

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.01)

    def call(self, input):
        x = self.vgg(input)
        x = self.flatten_layer(x)
        for dense_layer, dropout_layer in self.head:
            x = dense_layer(x)
            x = dropout_layer(x)
        x = self.output_layer(x)

        return x
    

class GuessModel():

    """
    predicts by guessing from training data
    """

    def __init__(self, train_labels, loss_fn, name="guess_model"):
        self.outputs = train_labels.shape[1]
        self.train_labels = train_labels
        self.loss_fn = loss_fn
        self.name = name

    def call(self, x):
        pred_labels = np.empty((x.shape[0], self.outputs))
        for i in range(x.shape[0]):
            pred_labels[i] = self.train_labels[random.randint(0, self.outputs - 1)]
        
        return pred_labels
    

class MeanModel():

    """
    predicts with mean of training data
    """

    def __init__(self, train_labels, loss_fn, name="mean_model"):
        self.outputs = train_labels.shape[1]
        self.train_mean = np.mean(train_labels, axis=0)
        self.loss_fn = loss_fn
        self.name = name

    def call(self, x):
        pred_labels = np.empty((x.shape[0], self.outputs))
        for i in range(x.shape[0]):
            pred_labels[i] = self.train_mean
        
        return pred_labels
    
    
class SimpleNN(tf.keras.Model):

    """
    dumb neural network
    """

    def __init__(self, output_units, name="simple_nn"):

        super().__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_units, activation='softmax')

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.01)
    
    def call(self, data):
        return self.dense(self.flatten(data))
