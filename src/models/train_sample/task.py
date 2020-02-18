import os

import util
import model
import pandas as pd

import tensorflow as tf
import numpy as np

from matplotlib import pyplot
from contextlib import redirect_stdout
from google.cloud import storage
from google.oauth2 import service_account

import time
import os
import h5py

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


def upload_file(bucket_name, source_file_name, destination_blob_name):
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(destination_blob_name)

	blob.upload_from_filename(source_file_name)
	print('File uploaded')


def remove_encoding(inv_map, prediction):
	rounded = prediction.round()
	tags = [inv_map[i] for i in range(len(rounded)) if rounded[i] == 1.0]
	return tags


def details(keras_model, x_test, y_test, loss, accuracy, fbeta):
	with open("model_info.txt", "w") as f:
		with redirect_stdout(f):
			keras_model.summary()

	inv_map = {v: k for k, v in util.TAG_MAPPING.items()}

	y_pred = keras_model.predict(x_test)
	y_pred = list(map(lambda x: remove_encoding(inv_map, x), y_pred))

	y_test = list(map(lambda x: remove_encoding(inv_map, x), y_test))

	f = open("model_info.txt", "a")
	f.write('Loss: %.3f' % loss)
	f.write('Accuracy: %.3f' % accuracy)
	f.write('F2 Score: %.3f' % fbeta)

	for i in range(len(list(y_pred))):
		f.write('Actual: {0}.   Expected: {1}\n'.format(y_test[i], y_pred[i]))

	f.close()
	upload_file('lovelace', "model_info.txt", "models/test/model_info.txt")


def save_and_upload(keras_model):
	keras_model.save('model.h5')
	upload_file('lovelace', 'model.h5', 'model.h5')


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

	train_start = time.time()
	history = keras_model.fit(train_data,
	                          steps_per_epoch=int(train_samples / BATCH_SIZE),
	                          epochs=NUM_EPOCHS,
	                          verbose=1,
	                          use_multiprocessing=True,
	                          validation_data=validation_data,
	                          validation_steps=1,
	                          callbacks=[learning_rate_decay])
	print("Training took: {0} seconds".format(time.time()-start_time))

	loss, accuracy, fbeta = keras_model.evaluate(x=x_test, y=y_test, verbose=1)
	print('>>> loss=%.3f, accuracy=%.3f, f2=%.3f' % (loss, accuracy, fbeta))
	# summarize(history)
	details(keras_model, x_test, y_test, loss, accuracy, fbeta)
	save_and_upload(keras_model)


if __name__ == '__main__':
	os.environ['GOOGLE_APPLICATION_CREDENTIALS']='../../../bry16607715-f906f68756ff.json'
	start_time = time.time()
	train_and_evaluate()
	print("Execution took: {0} seconds".format(time.time() - start_time))
