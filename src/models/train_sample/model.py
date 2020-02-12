import tensorflow as tf


def make_inputs(data, labels, epochs, batch_size):
    inputs = (data, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def create_model(input_dimensions):
    model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=input_dimensions)
    model.compile(optimize='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
