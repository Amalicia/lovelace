import numpy as np
import pandas as pd
import tensorflow as tf

from six.moves import urllib
from sklearn.model_selection import train_test_split

import tempfile
import os
import urllib
import urllib.request
import requests

DATA_DIR = os.path.join(tempfile.gettempdir(), 'haemorrhage_data')

DATA_URL = "https://storage.googleapis.com/lovelace/subset"
CSV_FILE = "train_labels.csv"
IMAGE_LOCATION = "images/"
NPZ_FILE = "full_data.npz"

NPZ_URL = 'https://storage.googleapis.com/lovelace/subset/full_data.npz'

CSV_URL = '%s/%s' % (DATA_URL, CSV_FILE)
IMAGES_URL = '%s/%s' % (DATA_URL, IMAGE_LOCATION)


def _download_file(filename, url):
	temp_file, _ = urllib.request.urlretrieve(url)
	temp = open(temp_file)
	f = open(filename, "w")
	f.write(temp.read())
	f.close()
	temp.close()


def download_csv(data_dir):
	tf.io.gfile.makedirs(data_dir)

	csv_file_path = os.path.join(data_dir, CSV_FILE)
	if not tf.io.gfile.exists(csv_file_path):
		_download_file(csv_file_path, CSV_URL)

	return csv_file_path


def download_img(data_dir, image_name):
	tf.io.gfile.makedirs(data_dir)

	image_file_path = os.path.join(data_dir, IMAGE_LOCATION)

	FULL_URL = "%s%s.png" % (IMAGES_URL, image_name)
	save_loc = '%s/%s.png' % (image_file_path, image_name)
	if not tf.io.gfile.exists(save_loc):
		urllib.request.urlretrieve(FULL_URL, save_loc)
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
	return labels_dict, inv_map


def encode(tags, mapping):
	encoding = np.zeros(len(mapping), dtype='uint8')
	tags_list = tags.split(' ')
	for tag in tags_list:
		encoding[mapping[tag]] = 1
	return encoding.tolist()


def encode_data(data):
	data.fillna('', inplace=True)
	labels_dict, inv_map = create_encoder_mapping(data)
	data['EncodedTag'] = data.apply(lambda row: encode(row['Tags'], labels_dict), axis=1)
	data['ImageNo'] = data['ImageNo'].apply(append_png)
	return labels_dict, inv_map


def get_image_arr(base_path):
	images = list()
	for image in os.listdir(base_path):
		pic = tf.keras.preprocessing.image.load_img(base_path + image, color_mode='grayscale', target_size=(224, 244))
		pic = tf.keras.preprocessing.image.img_to_array(pic, dtype='uint8')
		images.append(pic)
	return np.asarray(images, dtype='uint8')


def _load_data():
	csv_file_path = download_csv(DATA_DIR)
	df = pd.read_csv(csv_file_path)
	path = df['ImageNo'].apply(lambda x: download_img(DATA_DIR, x))

	mapping, inv_mapping = encode_data(df)
	image_arr = get_image_arr(path[0])
	labels = df['EncodedTag'].values
	labels = np.stack(labels, axis=0)

	np.savez_compressed('full_data.npz', image_arr, labels)
	return image_arr, labels


def download_npz(data_dir):
	path = './full_data.npz'

	urllib.request.urlretrieve('https://storage.googleapis.com/lovelace/subset/full_data.npz', path)

	data = np.load(path)
	image, label = data['arr_0'], data['arr_1']
	return image, label


def load_data():
	request = requests.get(NPZ_URL)
	if request.status_code == 200:
		image, label = download_npz(DATA_DIR)
	else:
		image, label = _load_data()

	print("Data shapes: {0}, {1}".format(image.shape, label.shape))

	x_train, y_train, x_test, y_test = train_test_split(image, label, random_state=42, test_size=0.3)
	return x_train, y_train, x_test, y_test

