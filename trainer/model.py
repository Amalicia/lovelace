from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import logging

def make_inputs(data, labels, epochs, batch_size):
	inputs = (data, labels)
	dataset = tf.data.Dataset.from_tensor_slices(inputs)

	dataset = dataset.repeat(epochs)
	dataset = dataset.batch(batch_size)
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


def create_model(input_dimensions, learning_rate, out_shape=6):
	strategy = tf.distribute.MirroredStrategy()
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
		opt = tf.keras.optimizers.RMSprop(lr=learning_rate)
		model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', fbeta])
	return model

# def create_model():
# 	# def create_model(input_dimensions, learning_rate, out_shape=6):
# 	mirrored_strategy = tf.distribute.MirroredStrategy()
# 	print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
# 	with mirrored_strategy.scope():
# 		model = tf.keras.models.Sequential()
# 		model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, kernel_initializer='he_uniform', padding='same',
# 		                                 input_shape=(224, 224, 1)))
# 		model.add(
# 			tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, kernel_initializer='he_uniform', padding='same'))
# 		model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# 		model.add(
# 			tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, kernel_initializer='he_uniform', padding='same'))
# 		model.add(
# 			tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, kernel_initializer='he_uniform', padding='same'))
# 		model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# 		model.add(
# 			tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, kernel_initializer='he_uniform', padding='same'))
# 		model.add(
# 			tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, kernel_initializer='he_uniform', padding='same'))
# 		model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# 		model.add(tf.keras.layers.Flatten())
# 		model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_uniform'))
# 		model.add(tf.keras.layers.Dense(6, activation='sigmoid'))
# 		# compile model
# 		opt = tf.keras.optimizers.RMSprop(lr=0.01)
# 		model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', fbeta])
