import tensorflow as tf
import numpy as np
from math import erf
from scipy.stats import multivariate_normal
import scipy
import skimage
import sklearn
import random
from tqdm import tqdm
from sklearn.cluster import KMeans, MeanShift

# import matplotlib
# matplotlib.use("tkagg")
from matplotlib import pyplot as plt

import preprocessing_gsv as preprocessing

def radians(deg):
    """
    converts degrees to radians
    """
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180


class DistanceAccuracy():
    """
    computes accuracy for threshold distance
    """

    def __init__(self, thresh=1000000, name="distance_accuracy"):
        self.thresh = thresh
        self.name = name
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = preprocessing.unnormalize_labels(y_true)
        y_pred = preprocessing.unnormalize_labels(y_pred)

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

        y_pred = tf.reduce_mean(tf.cast(distance < self.thresh, tf.float32))

        super().update_state(y_true, y_pred, sample_weight)

    def call(self, y_true, y_pred):
        """
        computes accuracy within threshold
        """

        y_true = preprocessing.unnormalize_labels(y_true)
        y_pred = preprocessing.unnormalize_labels(y_pred)

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

        return tf.reduce_mean(tf.cast(distance < self.thresh, tf.float32))
    
    def __call__(self, y_true, y_pred):
        return self.call(y_true, y_pred)


class MeanHaversineDistanceLoss(tf.keras.losses.Loss):
    """
    computes haversine distance loss
    """
    def __init__(self, name="mean_haversine_distance_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):

        y_true = preprocessing.unnormalize_labels(y_true)
        y_pred = preprocessing.unnormalize_labels(y_pred)

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

        distance = tf.reduce_mean(c * earth_radius)

        return distance
    

class SpreadMeanHaversineDistanceLoss(tf.keras.losses.Loss):
    """
    *********** [UNUSED] ***********
    computes haversine distance loss with distribution requirement 
    """
    def __init__(self, name="mean_haversine_distance_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):

        y_pred_norm = y_pred
        y_true = preprocessing.unnormalize_labels(y_true)
        y_pred = preprocessing.unnormalize_labels(y_pred)

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

        lmbda1 = 3
        lmbda2 = 0.002
        epsilon = 1e-5
        c = tf.pow(2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a)), 4)
        c = tf.reduce_mean(c)
        # center_penalty = tf.reduce_mean(tf.square(y_pred_norm - tf.constant(np.full(y_pred_norm.shape, 0.5), dtype=tf.float32)))
        var_penalty = 1 / (tf.reduce_mean(tf.math.reduce_variance(y_pred_norm, axis=0)) + epsilon)

        loss = c # + var_penalty * lmbda2 # + center_penalty * lmbda1

        return loss
    

class MeanNormalHaversineDistanceLoss(tf.keras.losses.Loss):
    """
    *********** [UNUSED] ***********
    computes haversine distance loss for a distribution
    """
    def __init__(self, name="mean_normal_haversine_distance_loss"):
        super().__init__(name=name)

    def call(self, y_true, mu, sigma):

        y_true = preprocessing.unnormalize_labels(y_true)
        mu = preprocessing.unnormalize_labels(mu)

        earth_radius = 6371000
        lat1, lon1 = tf.unstack(y_true, axis=-1)
        lat2, lon2 = tf.unstack(mu, axis=-1)

        lat1_rad = tf.cast(lat1 * np.pi / 180, tf.float32)
        lon1_rad = tf.cast(lon1 * np.pi / 180, tf.float32)
        lat2_rad = tf.cast(lat2 * np.pi / 180, tf.float32)
        lon2_rad = tf.cast(lon2 * np.pi / 180, tf.float32)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = tf.square(tf.sin(dlat / 2)) + tf.cos(lat1_rad) * tf.cos(lat2_rad) * tf.square(tf.sin(dlon / 2))
        c = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a))

        mean_c = tf.reduce_mean(c)
        std_c = tf.math.reduce_std(c)
        std_mu = tf.reduce_sum(tf.math.reduce_std(mu, axis=0))

        mean_sigma = tf.reduce_mean(sigma)
        
        epsilon = (1e-9)

        return mean_c + 1 / (std_c + epsilon) + (1 / mean_sigma + mean_sigma) / 100




class NaiveVGG(tf.keras.Model):
    """
    Naive VGG implementation, structure of models
    """
    def __init__(self, 
            units		= 512, 				# number of units in each dense layer
            input_shape = (300, 400, 3),    # image input shape
            layers		= 2, 				# number of dense layers
            dropout		= 0.2, 				# dropout proportion per dense layer
            output_units= 2, 				# number of units in output dense layer
            name		= "naive_vgg"		# name of model
        ):		
        
        super().__init__(name=name)

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

        self.loss = MeanHaversineDistanceLoss()
        self.optimizer = tf.keras.optimizers.Adam(0.01)

    def call(self, x):
        x = self.vgg(x)
        x = self.flatten_layer(x)
        for dense_layer, dropout_layer in self.head:
            x = dense_layer(x)
            x = dropout_layer(x)
        x = self.output_layer(x)

        return x


class worldNETCity(tf.keras.Model):
    """
    FINAL WORLDNET IMPLEMENTATION

    utilizes city_model classifier to predict continuous location
    """
    def __init__(self, 
            city_model,
            city_labels,
            units       = 512,
            layers		  = 2,
            dropout		  = 0.2,
            output_units= 2,
            name		    = "worldnet_city"
        ):

      super().__init__(name=name)

      self.city_model	= city_model
      self.city_labels = tf.constant([city_labels], dtype=tf.float32)
      self.city_model.trainable = False

      self.head = [
          (tf.keras.layers.Dense(units, activation="relu", name=f"{name}_dense_{i}"), \
          tf.keras.layers.Dropout(dropout, name = f"{name}_dropout_{i}")
          ) for i in range(layers)
      ]

      self.output_layer = tf.keras.layers.Dense(output_units, name=f"{name}_output_dense", activation="softmax")

      self.loss = MeanHaversineDistanceLoss()
      self.optimizer = tf.keras.optimizers.Adam(0.001)


    def call(self, x):
      x = self.city_model(x)
      for dense_layer, dropout_layer in self.head:
          x = dense_layer(x)
          x = dropout_layer(x)
      x = self.output_layer(x)

      return x @ self.city_labels




class VGGCityModel(tf.keras.Model):
    """
    City classifier model
    """
    def __init__(self, 
            units		= 512,
            input_shape = (300, 400, 3),
            layers		= 2,
            dropout		= 0.2,
            output_units= 2,
            name		= "vgg_city"
        ):		
        
        super().__init__(name=name)

        self.vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )

        # Freeze majority of VGG layers
        for i in range(18):
          self.vgg.layers[i].trainable = False

        self.flatten_layer = tf.keras.layers.Flatten(name=f"{name}_flatten")
        self.head = [
            (tf.keras.layers.Dense(units, activation="relu", name=f"{name}_dense_{i}"), \
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout, name = f"{name}_dropout_{i}")
            ) for i in range(layers)
        ]
        self.output_layer = tf.keras.layers.Dense(output_units, name = f"{name}_output_dense", activation="softmax")

        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(0.001)

    def call(self, x):
        x = self.vgg(x)
        x = self.flatten_layer(x)
        for dense_layer, norm_layer, dropout_layer in self.head:
            x = dense_layer(x)
            x = norm_layer(x)
            x = dropout_layer(x)
        x = self.output_layer(x)

        return x


class FeatureNearestNeighbors():

    """

    nearest neighbors classifier

    general idea:

        store all or some of the training feature vectors and their corresponding latitude/longitude label
        for each example in a testing dataset:
            find the k nearest neighbors (with either euclidian or cosine or some other similarity metric)
            compute the weighted mean of their locations
            compute the weighted standard deviation of their locations
            return this mean and standard deviation
    
    """

    def __init__(self, input_shape=(224, 224, 3), vector_shape=(12, 9), name="feature_nearest_neighbors"):
        self.name = name

        self.vector_shape = vector_shape
        self.vectors = []
        self.labels = []

        # self.vgg = tf.keras.models.load_model("weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
        # self.vgg.summary()
        
        '''
        self.vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )
        
        self.vgg.trainable = False
        '''

        self.loss = MeanHaversineDistanceLoss()
        self.optimizer = tf.keras.optimizers.Adam(0.01)

    def train(self, images, labels):
        
        num_images = len(images)
        
        #vgg_features = self.vgg(images)

        print("\n" + self.name, "training on", num_images, "images ...")
        print(np.shape(images))
        with tqdm(total=num_images) as pbar:
            for i in range(np.shape(images)[0]):
                self.vectors.append(images[i,:,:].flatten())
                self.labels.append(labels[i,:,:].flatten()[:2])
                pbar.update(1)


    def build_vocabulary(self, image_paths, vocab_size):
        ppc = 16
        cpb = 2

        num_imgs = len(image_paths)
        total_fds = np.empty((0, cpb*cpb*9))

        for i in range(num_imgs):
            img = skimage.io.imread(image_paths[i], as_gray=True)
            fd_mat = skimage.feature.hog(img, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb), feature_vector=True)
            hog_fds = fd_mat.reshape(-1, cpb*cpb*9)
            total_fds = np.append(total_fds, hog_fds, axis=0)

        kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=vocab_size, max_iter=100).fit(total_fds)

        return kmeans.cluster_centers_

    def get_bags_of_words(self, image_paths, vocab):
        ppc = 16
        cpb = 2

        hist_size = vocab.shape[0]
        img_hists = np.empty((0, hist_size))

        for i in range(len(image_paths)):
            img_hist = np.zeros((1, hist_size)) 
            img = skimage.io.imread(image_paths[i], as_gray=True)
            fd_mat = skimage.feature.hog(img, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb), feature_vector=True)
            hog_fds = fd_mat.reshape(-1, cpb*cpb*9)

            dists = scipy.spatial.distance.cdist(hog_fds, vocab, "euclidean")
            inds = np.argsort(dists, axis=1)[:, 0]
            for j in range(len(inds)):
                img_hist[0][inds[j]] += 1
            img_hist[0] = img_hist[0] / np.linalg.norm(img_hist[0])
            img_hists = np.append(img_hists, img_hist, axis=0)
        
        return img_hists

    def calc_mean(self, labels, dists):
        # labels shape: (x, 2)
        # dists shape: (x, k)
        # x = number of test images
        plt.scatter(labels[:, 0], labels[:, 1])
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.show()
        weighted_x = np.sum(labels[:, 0] * dists, axis=1) / np.sum(dists, axis=1)
        weighted_y = np.sum(labels[:, 1] * dists, axis=1) / np.sum(dists, axis=1)
        return weighted_x, weighted_y
    
    def calc_sd(self, dists, k):
        # means shape: (x, 1)
        # dists shape: (x, k)
        # x = number of test images
        weighted_sd = np.sqrt(np.sum((dists ** 2), axis=1) / k)
        return weighted_sd

    def nearest_neighbor_classify(self, train_image_feats, train_labels, test_image_feats):
        
        k = 100
        # finding the distances btwn each of the feature vectors
        print("finding distances...")
        dists = scipy.spatial.distance.cdist(test_image_feats, train_image_feats, 'euclidean')
        # finding k nearest neighbors of vectors
        print("sorting and finding k neighbors...")
        k_inds = np.argsort(dists, axis=1)[:, :k]
        print(np.shape(k_inds))
        # extracting the k nearest neighbors x and y labels 
        k_labels_x = np.take(train_labels[:,0:1], k_inds)
        k_labels_y = np.take(train_labels[:,1:2], k_inds)
        # k_labels shape: (x, k, 2) -> x training image features, k NN, 2 coordinates
        k_labels = np.stack((k_labels_x, k_labels_y), axis=2)

        # k nearest distances to the given training image
        k_dists = np.take(dists, k_inds)

        # calculate the weighted means and standard deviations
        weighted_m = self.calc_mean(k_labels, k_dists)
        weighted_m_x = weighted_m[:, 0]
        weighted_m_y = weighted_m[:, 1]
        weighted_sd = self.calc_sd(k_dists, k)

        return weighted_m_x, weighted_m_y, weighted_sd
    
    def call(self, images):
        # pass
        test_feats = []
        for i in range(np.shape(images)[0]):
          test_feats.append(images[i,:,:].flatten())
        print(np.shape(self.vectors))
        print(np.shape(test_feats))
        print(np.shape(self.labels))
        return self.nearest_neighbor_classify(train_image_feats=np.array(self.vectors), train_labels=np.array(self.labels), test_image_feats=np.array(test_feats))
    
    
class GuessModel():

    """
    predicts by guessing from training data [control]
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
    predicts with mean of training data [control]
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
    

class RandomizedGuessModel():

    """
    predicts random location [control]
    """

    def __init__(self, name="randomized_guess_model"):
        self.name = name
    
    def call(self, x):
        return np.random.random(2)
    
    
class SimpleNN(tf.keras.Model):

    """
    dumb neural network [control]
    """

    def __init__(self, output_units, name="simple_nn"):

        super().__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_units, activation='sigmoid')

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.01)
    
    def call(self, data):
        return self.dense(self.flatten(data))
    


    


class VGGCityFeaturesModel(tf.keras.Model):
  """
  *********** [UNUSED] ***********
  classifies features based on city
  """
  def __init__(self, 
            units		= 512,
            input_shape = (300, 400, 3),
            layers		= 1,
            dropout		= 0.4,
            output_units= 2,
            name		= "vgg_feature_city"
        ):		
        
        super().__init__(name=name)

        self.vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )

        self.feature_distribution_nn = FeatureDistributionNN(hidden_size=64, num_layers=4, output_units=output_units, output_activation="softmax")

        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(0.01)

  def call(self, x):
      x = self.vgg(x)
      x = tf.transpose(x, perm=[0, 3, 1, 2])
      x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
      x = tf.keras.layers.TimeDistributed(self.feature_distribution_nn)(x)
      return x




class FeatureDistributionNN(tf.keras.Model):

    """
    *********** [UNUSED] ***********
    predicts the mean and standard deviation of location of features
    """

    def __init__(self, hidden_size=8, num_layers=4, output_units=2, output_activation="sigmoid", name="feature_distribution_nn"):

        super().__init__(name=name)

        self.dense_layers = []

        for layer in range(num_layers):
            self.dense_layers.append(tf.keras.layers.Dense(hidden_size, activation="leaky_relu", kernel_initializer=tf.keras.initializers.GlorotUniform()))
            self.dense_layers.append(tf.keras.layers.Dropout(0.7))
        self.mu_layer = tf.keras.layers.Dense(output_units, activation=output_activation, kernel_initializer=tf.keras.initializers.GlorotUniform())

        self.loss = SpreadMeanHaversineDistanceLoss()
        self.optimizer = tf.keras.optimizers.Adam(0.01)
    
    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        mu = self.mu_layer(x)
        return mu
    


class FeatureDistributionModel():

    """
    *********** [UNUSED] ***********
    predicts location based on feature locations
    """

    def __init__(self, hidden_size=8, num_layers=2, name="feature_distribution_model"):
        
        self.name = name
        self.num_clusters = 5

        self.feature_distribution_nn = FeatureDistributionNN(hidden_size=hidden_size, num_layers=num_layers)

        self.loss = MeanNormalHaversineDistanceLoss()
        self.optimizer = tf.keras.optimizers.Adam(0.01)


    def call(self, x):
        mean_preds = self.feature_distribution_nn.call(x)
        mean_preds = mean_preds.numpy()

        plt.scatter(mean_preds[0, :, 1], mean_preds[0, :, 0])
        plt.show()

        centers = []
        for mean in mean_preds:
            kmeans = KMeans(n_clusters=self.num_clusters, n_init='auto')
            kmeans.fit(mean)

            cluster_totals = np.sum(np.eye(self.num_clusters)[kmeans.labels_], axis=1)

            centers.append(kmeans.cluster_centers_[np.argmax(cluster_totals)])
        
        return np.array(centers)



class VGGFeatureDistributionModel(tf.keras.Model):

    """
    *********** [UNUSED] ***********
    predicts based on VGG and feature predictor
    """

    def __init__(self, input_shape=(300, 400), hidden_size=8, num_layers=2, name="vgg_feature_distribution_model"):
        
        super().__init__(name=name)

        self.vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )

        # self.vgg.trainable = False

        self.feature_distribution_nn = FeatureDistributionNN(hidden_size=hidden_size, num_layers=num_layers)

        self.loss = SpreadMeanHaversineDistanceLoss()
        self.optimizer = tf.keras.optimizers.Adam(0.01)


    def call(self, x):
        x = self.vgg(x)
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        mu = tf.keras.layers.TimeDistributed(self.feature_distribution_nn)(x)
        return mu
        


class VGGFullFeatureDistributionModel(tf.keras.Model):

    """
    *********** [UNUSED] ***********
    predicts based on VGG and feature predictor
    """

    def __init__(self, input_shape=(300, 400), hidden_size=8, num_layers=2, name="vgg_full_feature_distribution_model"):
        
        super().__init__(name=name)

        self.vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )

        self.feature_distribution_nn = FeatureDistributionNN(hidden_size=hidden_size, num_layers=num_layers)

        self.head = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ]

        self.loss = SpreadMeanHaversineDistanceLoss()
        self.optimizer = tf.keras.optimizers.Adam(0.01)


    def call(self, x):
        x = self.vgg(x)
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x, sigma = tf.keras.layers.TimeDistributed(self.feature_distribution_nn)(x)
        for layer in self.head:
            x = layer(x)
        return x

    

class worldNET_test():

    """
    *********** [UNUSED] ***********
    predicts based on VGG and feature predictor, returns dominant cluster centroid
    """

    def __init__(self, input_shape=(300, 400), hidden_size=8, num_layers=2, num_clusters=5, name="worldnet"):

        self.name = name
        self.num_clusters = num_clusters

        self.feature_distribution_model = VGGFeatureDistributionModel(input_shape, hidden_size, num_layers)

        self.loss = SpreadMeanHaversineDistanceLoss()
    
    def call(self, x):

        print("\ncomputing predicted image centers ...")

        mean_preds = self.feature_distribution_model(x).numpy()

        centers = []
        for means in tqdm(mean_preds):

            """
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.scatter(means[:, 1], means[:, 0], c='b')
            plt.show()
            """

            kmeans = KMeans(n_clusters=self.num_clusters, n_init='auto')
            kmeans.fit(means)

            cluster_totals = np.sum(np.eye(self.num_clusters)[kmeans.labels_], axis=1)

            centers.append(kmeans.cluster_centers_[np.argmax(cluster_totals)])
        
        return np.array(centers)
        

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    """
    adapted from google
    creates inception module
    """

    kernel_init = tf.keras.initializers.glorot_uniform()
    bias_init = tf.keras.initializers.Constant(value=0.2)
    
    conv_1x1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = tf.keras.layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = tf.concat([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

    
def createInceptionModel(input_shape, output_units=32):

    """
    adapted from google
    creates inception network
    """
        
    kernel_init = tf.keras.initializers.glorot_uniform()
    bias_init = tf.keras.initializers.Constant(value=0.2)

    input_layer = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = tf.keras.layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=64,
                        filters_3x3_reduce=96,
                        filters_3x3=128,
                        filters_5x5_reduce=16,
                        filters_5x5=32,
                        filters_pool_proj=32,
                        name='inception_3a')

    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=192,
                        filters_5x5_reduce=32,
                        filters_5x5=96,
                        filters_pool_proj=64,
                        name='inception_3b')

    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=192,
                        filters_3x3_reduce=96,
                        filters_3x3=208,
                        filters_5x5_reduce=16,
                        filters_5x5=48,
                        filters_pool_proj=64,
                        name='inception_4a')


    x1 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x1 = tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.7)(x1)
    x1 = tf.keras.layers.Dense(output_units, activation='sigmoid', name='auxilliary_output_1')(x1)

    x = inception_module(x,
                        filters_1x1=160,
                        filters_3x3_reduce=112,
                        filters_3x3=224,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4b')

    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=256,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4c')

    x = inception_module(x,
                        filters_1x1=112,
                        filters_3x3_reduce=144,
                        filters_3x3=288,
                        filters_5x5_reduce=32,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4d')


    x2 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x2 = tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Dense(1024, activation='relu')(x2)
    x2 = tf.keras.layers.Dropout(0.7)(x2)
    x2 = tf.keras.layers.Dense(output_units, activation='sigmoid', name='auxilliary_output_2')(x2)

    x = inception_module(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_4e')

    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_5a')

    x = inception_module(x,
                        filters_1x1=384,
                        filters_3x3_reduce=192,
                        filters_3x3=384,
                        filters_5x5_reduce=48,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_5b')

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(output_units, activation="sigmoid", name='output_dense')

    return tf.keras.Model(input_layer, [x, x1, x2], name='inception')
