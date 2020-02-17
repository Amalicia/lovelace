import os

import util
import model
import pandas as pd

import tensorflow as tf
import numpy as np

from matplotlib import pyplot
from contextlib import redirect_stdout

NUM_EPOCHS = 1
BATCH_SIZE = 50
LEARNING_RATE = 0.01


def summarize(history):
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	pyplot.savefig('loss_acc_plot.png')
	pyplot.close()


def details(keras_model, x_test, y_test, loss, accuracy, fbeta):
	with open("model_info.txt", "w") as f:
		with redirect_stdout(f):
			keras_model.summary()

	y_pred = keras_model.predict(x_test)

	confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
	confusion_matrix_norm = np.around(confusion_matrix.asype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis],
	                                  decimals=2)

	confusion_matrix_df = pd.DataFrame(confusion_matrix_norm, index=util.TAG_MAPPING.keys(),
	                                   columns=util.TAG_MAPPING.keys())
	print(confusion_matrix_df)


def train_and_evaluate():
	x_train, y_train, x_test, y_test, x_val, y_val = util.load_data()

	train_samples = x_train.shape[0]
	print(train_samples)
	input_dimensions = x_train.shape[1:4]
	print(input_dimensions)

	print(x_train.shape, y_train.shape)

	test_samples = x_test.shape[0]
	val_samples = x_val.shape[0]

	keras_model = model.create_model(input_dimensions=input_dimensions, learning_rate=LEARNING_RATE)

	train_data = model.make_inputs(
		data=x_train,
		labels=y_train,
		epochs=NUM_EPOCHS,
		batch_size=BATCH_SIZE)

	validation_data = model.make_inputs(
		data=x_val,
		labels=y_val,
		epochs=NUM_EPOCHS,
		batch_size=val_samples
	)

	learning_rate_decay = tf.keras.callbacks.LearningRateScheduler(
		lambda epoch: LEARNING_RATE + 0.02 * (0.5 ** (1 + epoch))
	)

	history = keras_model.fit(train_data,
	                          steps_per_epoch=int(train_samples / BATCH_SIZE),
	                          epochs=NUM_EPOCHS,
	                          verbose=1,
	                          use_multiprocessing=True,
	                          validation_data=validation_data,
	                          validation_steps=1,
	                          callbacks=[learning_rate_decay])

	loss, accuracy, fbeta = keras_model.evaluate(x=x_test, y=y_test, verbose=1)
	print('>>> loss=%.3f, accuracy=%.3f' % (loss, accuracy))
	# summarize(history)
	details(keras_model, x_test, y_test, loss, accuracy, fbeta)


if __name__ == '__main__':
	train_and_evaluate()
