import tensorflow as tf
from math import erf
from scipy.stats import multivariate_normal

class DLModel(tf.keras.models.Model):
	def __init__(self,
	      units = 512,
	      dropout = 0.2,
	      num_layers = 2,
	      input_shape = (16, ),
	      layer_activation = "relu",
	      name = "deep_learning_model"
		  ):
		
		super().__init__(name = name)

		self.training = True
		self.input_shape = input_shape
		self.units = units
		self.dropout = dropout

		self.dense_layers = [
			tf.keras.layers.Dense(self.units, activation = layer_activation, name = f"{name}_dense_{i}") \
				for i in range(num_layers)]
		
		self.dropout_layers = [
			tf.keras.layers.Dropout(self.dropout, name = f"{name}_dropout_{i}") \
				for i in range(num_layers)]
		
		self.mean_prediction = tf.keras.layers.Dense(2, activation = "linear")

		# Can either be activated with ReLU or clip_by_value.
		# Problem with clip_by_value is it affects gradient
		# - there is an alternative of tfp.clip_by_value_preserve_gradient
		self.sigma_prediction = tf.keras.layers.Dense(1, activation = "relu")
	
	def call(self, x):
		
		for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
			x = dense_layer(x)
			x = dropout_layer(x)

		mean_pred = self.mean_prediction(x)
		sigma_pred = tf.math.log1p(self.sigma_prediction(x))

		return mean_pred, sigma_pred
	
	def predict_sample(self, x):
		"""
		Predict the most likely coordinate given samples of features.

		Logic:
		Find the maximum point of the sum of probability distribution functions (PDF) of all normal distributions.
		Since there may be a lot of local max and min (and wrt computational resources), I am not sure if gradient
		optimization is the best idea. I have two ideas of finding or estimating the maximum point.

		Ideas:
		1. Estimation from uniform samples
		- uniformly sample points across entire space and find point with maximum probability
		2. Sum of directions
		- sum all the scaled directions between every mean and the center
		- this is more like a "walk" over the space
		- this raises the question of whether the center (0, 0) is the best starting point?

		Args:
		- self properties
			- call function used to predict mean predictions and sigma predictions of all features
		- x, features to predict from

		Returns:
		- coordinate (x, y) of highest probability
		- scaled probability density output at that coordinate
		"""

		# Get mean and sigma predictions
		mean_preds, sigma_preds = self.call(x)

		# Create 2d normal distributions with every pair of mean and sigma predictions
		distributions = [
			multivariate_normal(mu, sigma) for mu, sigma in zip(mean_preds, sigma_preds)
		]

		# Idea 1: Estimation from uniform samples
		# - find the "box" that contains all means
		# - uniformly sample points over the box
		# - calculate the summed PDF at every point
		# - find coordinate of maximum value

		# Bottom left corner
		bottom_left_coordinate = tf.reduce_min(mean_preds, axis = 0)

		# Top right corner
		top_right_coordinate = tf.reduce_max(mean_preds, axis = 0)


		# Sample grid coordinates of the box 
		nx, ny = (10, 10)

		x_space = tf.linspace(bottom_left_coordinate[0], top_right_coordinate[0], nx)
		y_space = tf.linspace(bottom_left_coordinate[1], top_right_coordinate[1], ny)

		X, Y = tf.meshgrid(x_space, y_space)

		mesh_coords = tf.reshape(tf.concat([X[..., tf.newaxis], Y[..., tf.newaxis]], axis = -1), (-1, 2))

		# Sum probability density of each coordinate for every distribution based on each feature prediction
		pdf_scores = tf.zeros((tf.shape(mesh_coords)[0], ), dtype = tf.float32)

		for dist in distributions:
			pdf_scores += dist.pdf(mesh_coords)
		
		return mesh_coords[tf.argmax(pdf_scores, axis = 0)], (tf.reduce_max(pdf_scores) / tf.reduce_sum(pdf_scores))


		# Idea 2: Sum of directions
		# - calculate all vector directions between every mean and the center
		# - scale each direction based on the std of the distribution that is pointed TOWARDS (simulating "pull" of distribution)
		# - sum all directions

	def normal_cdf(x, mu, sigma):
		return 0.5 * (1. + erf((x - mu) / (sigma * (2 ** 0.5))))

		
