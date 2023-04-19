import tensorflow as tf
import numpy as np

import preprocessing
import preprocessing_im2gps as preprocessing2
import random

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
		x = tf.keras.applications.vgg19.preprocess_input(input)
		x = self.vgg(x)
		x = self.flatten_layer(x)
		for dense_layer, dropout_layer in self.head:
			x = dense_layer(x)
			x = dropout_layer(x)
		x = self.output_layer(x)

		return x
	

def guess_loss(train_labels, test_labels, loss_fn):
	pred_labels = np.empty(test_labels.shape)
	for i in range(len(test_labels)):
		pred_labels[i] = train_labels[random.randint(0, len(train_labels) - 1)]
	
	return loss_fn(pred_labels, test_labels)
	
	


def main():

	train_prop = 0.8
	val_prop = 0.1
	num_samples = 500
	preprocessing.NUM_SAMPLES = num_samples
	num_training_samples, num_testing_samples = int(train_prop * num_samples), int((1 - train_prop - val_prop) * num_samples)
	images = preprocessing.load_images("images/", save=True, load=True, one_angle=True)
	labels = preprocessing.get_coordinates()
	train_images, val_images = images[:num_training_samples], images[num_training_samples:-num_testing_samples]
	train_labels, val_labels = labels[:num_training_samples], labels[num_training_samples:-num_testing_samples]

	model = NaiveVGG(units=32, output_units=3)
	model.compile(
		optimizer=model.optimizer,
		loss=model.loss,
		metrics=[],
	)
	model.build((None, 224, 224, 3))
	model.summary()
	model.fit(train_images, train_labels, batch_size=64, epochs=4, validation_data=(val_images, val_labels))

	print("loss to beat:", guess_loss(train_labels, val_labels, model.loss))


if __name__ == "__main__":
	main()