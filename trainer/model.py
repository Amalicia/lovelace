from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import logging

log = logging.getLogger(__name__)


def make_inputs(data, labels, batch_size, generator):
	dataset = tf.data.Dataset.from_generator(
		lambda: generator.flow(data, labels, batch_size=batch_size),
		output_types=(tf.float32, tf.uint8),
		output_shapes=([None, 224, 224, 3], [None, 6])
	)
	return dataset


# https://www.kaggle.com/arsenyinfo/f-beta-score-for-keras
def fbeta(y_pred, y_true, beta=2):
	y_pred = tf.keras.backend.clip(y_pred, 0, 1)

	true_pos = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)), axis=1)
	false_pos = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred - y_true, 0, 1)), axis=1)
	false_neg = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true - y_pred, 0, 1)), axis=1)

	precision = true_pos / (true_pos + false_pos + tf.keras.backend.epsilon())
	recall = true_pos / (true_pos + false_neg + tf.keras.backend.epsilon())

	beta_squared = beta ** 2
	fbeta_score = tf.keras.backend.mean(
		(1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + tf.keras.backend.epsilon()))
	return fbeta_score


def create_baseline_model(input_dimensions, learning_rate, out_shape=6):
	strategy = tf.distribute.MirroredStrategy()
	log.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
	with strategy.scope():
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, kernel_initializer='he_uniform', padding='same',
		                                 input_shape=input_dimensions))
		model.add(
			tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, kernel_initializer='he_uniform', padding='same'))
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))
		model.add(
			tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, kernel_initializer='he_uniform', padding='same'))
		model.add(
			tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, kernel_initializer='he_uniform', padding='same'))
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))
		model.add(
			tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, kernel_initializer='he_uniform', padding='same'))
		model.add(
			tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, kernel_initializer='he_uniform', padding='same'))
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_uniform'))
		model.add(tf.keras.layers.Dense(out_shape, activation='sigmoid'))
		# compile model
		opt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)
		model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', fbeta])
	return model


def create_resnet_model(input_dimensions, learning_rate, out_shape=6):
	strategy = tf.distribute.MirroredStrategy()
	log.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
	with strategy.scope():
		base_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=input_dimensions)
		x = base_model.output
		x = tf.keras.layers.GlobalAveragePooling2D()(x)
		x = tf.keras.layers.Dropout(0.2)(x)

		predictions = tf.keras.layers.Dense(out_shape, activation='sigmoid')(x)
		model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

		opt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)
		model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', fbeta])
	return model


def create_densenet_model(input_dimensions, learning_rate, out_shape=6):
	strategy = tf.distribute.MirroredStrategy()
	log.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
	with strategy.scope():
		base_model = tf.keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=input_dimensions)
		x = base_model.output
		x = tf.keras.layers.GlobalAveragePooling2D()(x)
		x = tf.keras.layers.Dropout(0.2)(x)

		predictions = tf.keras.layers.Dense(out_shape, activation='sigmoid')(x)
		model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

		opt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)
		model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', fbeta])
	return model
