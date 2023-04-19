import tensorflow as tf

def create_naive_model(
		units = 512,
		layers = 2,
		output_units = 2,
		dropout = 0.2,
		input_shape = (224, 224, 3),
		freeze_vgg = True,
		name="naive_vgg"):
	
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
	