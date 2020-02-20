from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool
from six.moves import urllib
from sklearn.model_selection import train_test_split

import tempfile
import os
import urllib
import urllib.request
import requests
import time
import logging
import swifter

from . import upload_file

DATA_DIR = os.path.join(tempfile.gettempdir(), 'haemorrhage_data')

DATA_URL = "https://storage.googleapis.com/lovelace/{0}"
CSV_FILE = "train_labels.csv"
IMAGE_LOCATION = "images/"
NPZ_FILE = "full_data.npz"

NPZ_URL = 'https://storage.googleapis.com/lovelace/{0}/full_data.npz'

CSV_URL = '%s/%s' % (DATA_URL, CSV_FILE)
IMAGES_URL = '%s/%s' % (DATA_URL, IMAGE_LOCATION)

TAG_MAPPING = {
	'none': 0,
	'epidural': 1,
	'intraparenchymal': 2,
	'intraventricular': 3,
	'subarachnoid': 4,
	'subdural': 5
}

log = logging.getLogger("lovelace")
log.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
log.addHandler(console)


def _download_file(filename, url):
	temp_file, _ = urllib.request.urlretrieve(url)
	temp = open(temp_file)
	f = open(filename, "w")
	f.write(temp.read())
	f.close()
	temp.close()


def download_csv(data_dir, arg_data):
	tf.io.gfile.makedirs(data_dir)

	csv_file_path = os.path.join(data_dir, CSV_FILE)
	if not tf.io.gfile.exists(csv_file_path):
		log.info('Downloading file...')
		_download_file(csv_file_path, CSV_URL.format(arg_data))

	log.info('Returned path')
	return csv_file_path


def download_img(image_name, data_dir, arg_data):
	retries = 15
	image_file_path = os.path.join(data_dir, IMAGE_LOCATION)

	tf.io.gfile.makedirs(image_file_path)

	FULL_URL = "%s%s.png" % (IMAGES_URL.format(arg_data), image_name)
	save_loc = '%s%s.png' % (image_file_path, image_name)
	if not tf.io.gfile.exists(save_loc):
		while (retries > 0):
			try:
				urllib.request.urlretrieve(FULL_URL, save_loc)
				break
			except:
				log.warn('Retrying {0}'.format(FULL_URL))
				retries = retries - 1
				time.sleep(1)
	return image_file_path


def append_png(image):
	return image + '.png'


def create_encoder_mapping(data):
	labels = set()
	for i in range(len(data)):
		labels.update(data['Tags'][i].split(' '))

	labels = list(labels)
	labels.sort()

	labels_dict = {labels[i]: i for i in range(len(labels))}
	inv_map = {v: k for k, v in labels_dict.items()}
	print(labels_dict)
	return labels_dict, inv_map


def encode(tags):
	encoding = np.zeros(len(TAG_MAPPING), dtype='uint8')
	tags_list = tags.split(' ')
	for tag in tags_list:
		encoding[TAG_MAPPING[tag]] = 1
	return encoding.tolist()


def encode_data(data):
	data.fillna('none', inplace=True)
	data['EncodedTag'] = data.apply(lambda row: encode(row['Tags']), axis=1)
	data['ImageNo'] = data['ImageNo'].apply(append_png)


def get_image_arr(base_path):
	images = list()
	for image in os.listdir(base_path):
		pic = tf.keras.preprocessing.image.load_img(base_path + image, color_mode='grayscale', target_size=(224, 244))
		pic = tf.keras.preprocessing.image.img_to_array(pic, dtype='uint8')
		images.append(pic)
	return np.asarray(images, dtype='uint8')


def _load_data(data_arg):
	log.info('Downloading CSV')
	csv_file_path = download_csv(DATA_DIR, data_arg)
	df = pd.read_csv(csv_file_path)
	log.info('Downloading Images')

	path = df['ImageNo'].swifter.allow_dask_on_strings().apply(lambda x: download_img(DATA_DIR, x, data_arg))

	encode_data(df)
	image_arr = get_image_arr(path[0])
	labels = df['EncodedTag'].values
	labels = np.stack(labels, axis=0)

	np.savez_compressed('full_data.npz', image_arr, labels)
	upload_file.upload_file('lovelace', 'full_data.npz', data_arg)
	return image_arr, labels


def download_npz(data_dir, data_arg):
	path = './full_data.npz'

	urllib.request.urlretrieve('https://storage.googleapis.com/lovelace/{0}/full_data.npz'.format(data_arg), path)

	data = np.load(path)
	image, label = data['arr_0'], data['arr_1']
	return image, label


def load_data(data_arg):
	request = requests.get(NPZ_URL.format(data_arg))
	if request.status_code == 200:
		log.info('Loading NPZ')
		image, label = download_npz(DATA_DIR, data_arg)
	else:
		log.info('Loading CSV and images')
		image, label = _load_data(data_arg)

	log.info("Data shapes: {0}, {1}".format(image.shape, label.shape))

	x_train, x_test, y_train, y_test = train_test_split(image, label, random_state=42, test_size=0.3)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
	return x_train, y_train, x_test, y_test, x_val, y_val
