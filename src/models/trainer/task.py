from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
# from matplotlib import pyplot
from contextlib import redirect_stdout

import time
import os
import h5py
import argparse

from . import util
from . import model
from . import upload_file


def args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--job-dir', type=str, required=True)
	parser.add_argument('--model-name', type=str, required=True, help='name for created model')
	parser.add_argument('--data', choices=['SUBSET', 'ALL'], default='SUBSET', help='which dataset to use. default: SUBSET')
	parser.add_argument('--epochs', type=int, default=10, help='no of times to go through data. default: 10')
	parser.add_argument('--batch-size', type=int, default=128, help='number of images in each training step. default: 128')
	parser.add_argument('--lr', type=float, default=0.01, help='learning rate for gradient descent. default: 128')
	arguments, _ = parser.parse_known_args()
	return arguments


# def summarize(history):
# 	pyplot.subplot(211)
# 	pyplot.title('Cross Entropy Loss')
# 	pyplot.plot(history.history['loss'], color='blue', label='train')
# 	pyplot.plot(history.history['val_loss'], color='orange', label='test')
# 	# plot accuracy
# 	pyplot.subplot(212)
# 	pyplot.title('accuracy')
# 	pyplot.plot(history.history['accuracy'], color='blue', label='train')
# 	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
# 	# save plot to file
# 	pyplot.savefig('loss_acc_plot.png')
# 	pyplot.close()


def remove_encoding(inv_map, prediction):
	rounded = prediction.round()
	tags = [inv_map[i] for i in range(len(rounded)) if rounded[i] == 1.0]
	return tags


def details(args, keras_model, x_test, y_test, loss, accuracy, fbeta):
	print('Writing model details...')
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
	print('Done!')
	cloud_file = 'models/{0}/{1}_info.txt'.format(args.data.lower(), args.model_name)
	upload_file.upload_file('lovelace', "model_info.txt", cloud_file)


def save_and_upload(args, keras_model):
	print('Saving model as h5...')
	keras_model.save('model.h5')
	cloud_model = 'models/{0}/{1}.h5'.format(args.data.lower(), args.model_name)
	upload_file.upload_file('lovelace', 'model.h5', cloud_model)

	# print('Saving model as TF SavedModel')
	# export_path = os.path.join(args.job_dir, args.model_name)
	# tf.keras.models.save_model(keras_model, export_path)
	# print('Saved model as tf')
	# # tf_model = 'models/{0}/{1}.tf'.format(args.data.lower(), args.model_name)
	# # upload_file.upload_file('lovelace', 'model.tf', tf_model)


def train_and_evaluate(args):
	x_train, y_train, x_test, y_test, x_val, y_val = util.load_data(args.data.lower())
	print('Loaded data')

	train_samples = x_train.shape[0]
	input_dimensions = x_train.shape[1:4]

	test_samples = x_test.shape[0]
	val_samples = x_val.shape[0]

	keras_model = model.create_model(input_dimensions=input_dimensions, learning_rate=args.lr)

	train_data = model.make_inputs(
		data=x_train,
		labels=y_train,
		epochs=args.epochs,
		batch_size=args.batch_size)

	validation_data = model.make_inputs(
		data=x_val,
		labels=y_val,
		epochs=args.epochs,
		batch_size=val_samples
	)

	learning_rate_decay = tf.keras.callbacks.LearningRateScheduler(
		lambda epoch: args.lr + 0.02 * (0.5 ** (1 + epoch))
	)

	tensorboard_cb = tf.keras.callbacks.TensorBoard(
		os.path.join(args.job_dir, 'keras_tensorboard'),
		histogram_freq=1)

	print('Beginning training')
	train_start = time.time()
	keras_model.fit(train_data,
	                          steps_per_epoch=int(train_samples / args.batch_size),
	                          epochs=args.epochs,
	                          verbose=1,
	                          use_multiprocessing=True,
	                          validation_data=validation_data,
	                          validation_steps=1,
	                          callbacks=[learning_rate_decay, tensorboard_cb])
	print("Training took: {0} seconds".format(time.time()-train_start))

	loss, accuracy, fbeta = keras_model.evaluate(x=x_test, y=y_test, verbose=1)
	print('>>> loss=%.3f, accuracy=%.3f, f2=%.3f' % (loss, accuracy, fbeta))
	# summarize(history)
	details(args, keras_model, x_test, y_test, loss, accuracy, fbeta)
	save_and_upload(args, keras_model)



if __name__ == '__main__':
	# os.environ['GOOGLE_APPLICATION_CREDENTIALS']='../../bry16607715-f906f68756ff.json'
	start_time = time.time()
	args = args()
	tf.compat.v1.logging.set_verbosity('INFO')
	train_and_evaluate(args)
	print("Execution took: {0} seconds".format(time.time() - start_time))
