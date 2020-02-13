import os

import util
import model

import tensorflow as tf
import numpy as np

NUM_EPOCHS = 10
BATCH_SIZE = 50


def train_and_evaluate():
	x_train, y_train, x_test, y_test = util.load_data()

	train_samples = x_train.shape[0]
	print(train_samples)
	input_dimensions = x_train.shape[1:4]
	print(input_dimensions)

	print(x_train.shape, y_train.shape)

	test_samples = x_test.shape[0]

	keras_model = model.create_model(input_dimensions=input_dimensions)

	train_data = model.make_inputs(
		data=x_train,
		labels=y_train,
		epochs=NUM_EPOCHS,
		batch_size=BATCH_SIZE)

	keras_model.fit(train_data,
	          steps_per_epoch=int(train_samples / BATCH_SIZE),
	          epochs=NUM_EPOCHS,
	          verbose=1)


if __name__ == '__main__':
	train_and_evaluate()
