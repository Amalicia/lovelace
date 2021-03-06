from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from contextlib import redirect_stdout
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix

import time
import os
import argparse
import logging
import numpy as np
import gc

from . import util
from . import model
from . import upload_file

# tensorflow setup
sess = tf.Session()
graph = tf.get_default_graph()
tf.keras.backend.set_session(sess)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
log.addHandler(console)


# argument parser
def args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--job-dir', type=str, required=True)
	parser.add_argument('--model-name', type=str, required=True, help='name for created model')
	parser.add_argument('--data', choices=['SUBSET', 'ALL'], default='SUBSET',
	                    help='which dataset to use. default: SUBSET')
	parser.add_argument('--epochs', type=int, default=10, help='no of times to go through data. default: 10')
	parser.add_argument('--batch-size', type=int, default=128,
	                    help='number of images in each training step. default: 128')
	parser.add_argument('--lr', type=float, default=0.01, help='learning rate for gradient descent. default: 128')
	arguments, _ = parser.parse_known_args()
	return arguments


# create model details file
def details(args, keras_model, x_test, y_test, loss, accuracy, fbeta, batch_size):
	log.info('Writing model details...')
	f = open("model_info.txt", "a")
	# write model name, epochs and batch size
	f.write('**** {0} *****'.format(args.model_name))
	f.write('Epochs: {0}\nBatch Size: {1}\n\n'.format(args.epochs, args.batch_size))

	# write model summary to file
	with open("model_info.txt", "w") as f:
		with redirect_stdout(f):
			keras_model.summary()
	log.info('Written summary')

	labels = list(util.TAG_MAPPING.keys())

	# make predictions and round them to be between 0 and 1
	log.info("Making predictions")
	y_pred = keras_model.predict(x_test, batch_size=batch_size, verbose=1, steps=int(x_test.shape[0] / batch_size))
	y_pred = y_pred.round().astype(np.int)

	# write model statistics
	log.info('Writing stats')
	f = open("model_info.txt", "a")
	f.write('Loss: %.3f\n' % loss)
	f.write('Accuracy: %.3f\n' % accuracy)
	f.write('F2 Score: %.3f\n\n' % fbeta)

	log.info('Writing report + Confusion matrix')
	# write classification report
	f.write(classification_report(y_test, y_pred, target_names=labels))
	log.info("Written report")
	# write confusion matrices
	cm = multilabel_confusion_matrix(y_test, y_pred)
	for i in range(len(labels)):
		f.write('Confusion Matrix for {0}:\n'.format(labels[i]))
		f.write(np.array2string(cm[i]))
		f.write('\n\n')

	f.close()
	log.info('Done writing details!')
	# upload file
	cloud_file = 'models/{0}/{1}_info.txt'.format(args.data.lower(), args.model_name)
	upload_file.upload_file('lovelace-data', "model_info.txt", cloud_file)


# save and upload model to cloud storage
def save_and_upload(args, keras_model):
	# save and upload model
	log.info('Saving model as h5...')
	keras_model.save('model.h5')
	cloud_model = 'models/{0}/{1}.h5'.format(args.data.lower(), args.model_name)
	upload_file.upload_file('lovelace-data', 'model.h5', cloud_model)

	save_loc = '%s/%s' % (args.job_dir, args.model_name)
	tf.saved_model.save(keras_model, save_loc)


# train and evaluate model
def train_and_evaluate(args):
	# get data
	x_train, y_train, x_test, y_test, x_val, y_val = util.load_data(args.data.lower())
	log.info('Loaded data')

	train_samples = x_train.shape[0]
	input_dimensions = x_train.shape[1:4]

	test_samples = x_test.shape[0]
	val_samples = x_val.shape[0]

	# create model
	keras_model = model.create_resnet_model(input_dimensions=input_dimensions, learning_rate=args.lr)

	# create keras generator for images
	image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

	# get train and validation data as a tensorflow dataset
	train_data = model.make_inputs(x_train, y_train, batch_size=args.batch_size, generator=image_generator)
	validation_data = model.make_inputs(x_val, y_val, batch_size=args.batch_size, generator=image_generator)

	del x_train, y_train, x_val, y_val
	gc.collect()

	# create learning rate callback
	learning_rate_decay = tf.keras.callbacks.LearningRateScheduler(
		lambda epoch: args.lr + 0.02 * (0.5 ** (1 + epoch))
	)

	# create tensorboard callback
	tensorboard_cb = tf.keras.callbacks.TensorBoard(
		os.path.join(args.job_dir, 'keras_tensorboard'),
		histogram_freq=1)

	# start trainging
	log.info('Beginning training')
	train_start = time.time()
	with graph.as_default():
		tf.keras.backend.set_session(sess)
		# train model
		keras_model.fit(train_data,
		                steps_per_epoch=int(train_samples / args.batch_size),
		                epochs=args.epochs,
		                verbose=1,
		                validation_data=validation_data,
		                validation_steps=int(val_samples / args.batch_size),
		                callbacks=[learning_rate_decay, tensorboard_cb])
	log.info("Training took: {0} seconds".format(time.time() - train_start))

	del train_data, validation_data
	gc.collect()

	# find optimal batch size for the test dataset so the data can be bacthed and the GPU's will not run out of memory
	for i in range(1, 160):
		if len(x_test) % i == 0:
			div = i
	test_batch = div

	# evaluate model
	log.info("Evaluating model")
	# create tensorflow dataset representation of evaluation data
	test_inputs = model.make_inputs(x_test, y_test, batch_size=test_batch, generator=image_generator)
	log.info('Done making test inputs')
	# evaluate model
	loss, accuracy, fbeta = keras_model.evaluate(test_inputs, steps=int(test_samples / test_batch), verbose=1)
	del test_inputs
	gc.collect()
	# log out the results
	log.info('>>> loss=%.3f, accuracy=%.3f, f2=%.3f' % (loss, accuracy, fbeta))
	# create model details file
	details(args, keras_model, x_test, y_test, loss, accuracy, fbeta, test_batch)
	# save and upload model
	save_and_upload(args, keras_model)


# main function
if __name__ == '__main__':
	# os.environ['GOOGLE_APPLICATION_CREDENTIALS']='bry16607715-f906f68756ff.json'
	log.info('Starting...')
	# start time - to measure how long the process takes
	start_time = time.time()
	# get argument data
	args = args()
	# set logging verbosity
	tf.compat.v1.logging.set_verbosity('INFO')
	# train and evaluate model
	train_and_evaluate(args)
	# report time taken to run the entrie script
	log.info("Execution took: {0} seconds".format(time.time() - start_time))
