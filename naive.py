import tensorflow as tf

def create_naive_model(
		units = 512, # number of units in each dense layer
		layers = 2, # number of dense layers
		output_units = 2, # number of units in output dense layer
		dropout = 0.2, # dropout percentage per dense layer
		input_shape = (224, 224, 3), # input shape
		freeze_vgg = True, # whether to freeze vgg or not
		name="naive_vgg" # name of model
		):
	"""
	Creates a naive model that uses frozen pretrained VGG19 weights
	and a trainable densely connected network head to make predictions.

	Args (with defaults):
		units = 512, # number of units in each dense layer
		layers = 2, # number of dense layers
		output_units = 2, # number of units in output dense layer
		dropout = 0.2, # dropout percentage per dense layer
		input_shape = (224, 224, 3), # input shape
		freeze_vgg = True, # whether to freeze vgg or not
		name="naive_vgg" # name of model

	Output:
		Tensorflow Keras Model instance:
		- With default parameters, there are 13 million trainable parameters (head)
		and 20 million frozen parameters (vgg).
	"""
	
	vgg = tf.keras.applications.VGG19(
			include_top=False,
			weights="imagenet",
			input_tensor=None,
			input_shape=input_shape,
			pooling=None,
		)
	
	if freeze_vgg:
		vgg.trainable = False
	
	input = tf.keras.layers.Input(shape = input_shape, name=f"{name}_input")

	vgg_output = vgg(input)

	x = tf.keras.layers.Flatten(name=f"{name}_flatten")(vgg_output)

	for i in range(layers):
		x = tf.keras.layers.Dense(units, activation = "relu", name=f"{name}_dense_{i}")(x)
		if dropout > 0:
			x = tf.keras.layers.Dropout(dropout, name = f"{name}_dropout_{i}")(x)

	output_layer = tf.keras.layers.Dense(output_units, activation = "linear", name = f"{name}_output_dense")(x)

	return tf.keras.models.Model(inputs = [input], outputs = [output_layer], name = name)

if __name__ == "__main__":
	m = create_naive_model()
	print(m.summary())
	