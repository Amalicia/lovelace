from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing
from sklearn.model_selection import train_test_split
from functools import partial

import tempfile
import os
import urllib
import urllib.request
import requests
import time
import logging
import gc

from . import upload_file

# config
DATA_DIR = os.path.join(tempfile.gettempdir(), 'haemorrhage_data/{0}/')

DATA_URL = "https://storage.googleapis.com/lovelace-data/{0}"
CSV_FILE = "train_labels.csv"
IMAGE_LOCATION = "images/"
NPZ_FILE = "full_data.npz"

NPZ_URL = 'https://storage.googleapis.com/lovelace-data/{0}/full_data.npz'

CSV_URL = '%s/%s' % (DATA_URL, CSV_FILE)
IMAGES_URL = '%s/%s' % (DATA_URL, IMAGE_LOCATION)

# mapping of tag to numerical value
TAG_MAPPING = {
	'none': 0,
	'epidural': 1,
	'intraparenchymal': 2,
	'intraventricular': 3,
	'subarachnoid': 4,
	'subdural': 5
}

log = logging.getLogger(__name__)


# download file and save locally
def _download_file(filename, url):
	temp_file, _ = urllib.request.urlretrieve(url)
	temp = open(temp_file)
	f = open(filename, "w")
	f.write(temp.read())
	f.close()
	temp.close()


# download csv file
def download_csv(data_dir, arg_data):
	# create directory
	tf.io.gfile.makedirs(data_dir)

	csv_file_path = os.path.join(data_dir, CSV_FILE)
	# check if file does not exist
	if not tf.io.gfile.exists(csv_file_path):
		log.info('Downloading file...')
		_download_file(csv_file_path, CSV_URL.format(arg_data))

	log.info('Returned path')
	return csv_file_path


# download image
def download_img(image_name, image_file_path, arg_data):
	# number of retries
	retries = 15
	log.info('Downloading Image: {0}'.format(image_name))
	full_url = "%s%s.png" % (IMAGES_URL.format(arg_data), image_name)
	save_loc = '%s%s.png' % (image_file_path, image_name)
	# check if image exists
	if not tf.io.gfile.exists(save_loc):
		# repeat check retry count
		while retries > 0:
			try:
				# save image locally
				urllib.request.urlretrieve(full_url, save_loc)
				break
			except:
				# retry block
				log.warning('Retrying {0}'.format(full_url))
				retries = retries - 1
				time.sleep(1)


# append .png to a string
def append_png(image):
	return image + '.png'


# encode standard labels to be one hot encoded
def encode(tags):
	# create an array of length 6 containing just 0's
	encoding = np.zeros(len(TAG_MAPPING), dtype='uint8')
	# split the tags where there is a space
	tags_list = tags.split(' ')
	for tag in tags_list:
		# convert tag to the mapping and set the value in the array to one
		encoding[TAG_MAPPING[tag]] = 1
	return encoding.tolist()


# data encoding
def encode_data(data):
	# replace all na values in dataframe with 'none'
	data.fillna('none', inplace=True)
	#  encode the tags
	data['EncodedTag'] = data.apply(lambda row: encode(row['Tags']), axis=1)
	# append png to the image id
	data['ImageNo'] = data['ImageNo'].apply(append_png)


# laod an image and get the values of each pixel
def get_image_arr(image, base_path):
	pic = tf.keras.preprocessing.image.load_img(base_path + image, color_mode='grayscale', target_size=(224, 224))
	pic = tf.keras.preprocessing.image.img_to_array(pic, dtype='uint8')
	return pic


# load data when no npz file is present in storage
def __load_data(data_arg):
	# download csv
	log.info('Downloading CSV')
	csv_file_path = download_csv(DATA_DIR.format(data_arg), data_arg)
	df = pd.read_csv(csv_file_path)

	# create directory to download images
	image_file_path = os.path.join(DATA_DIR.format(data_arg), IMAGE_LOCATION)
	print(image_file_path)

	# set up multiprocessing
	cpu_count = multiprocessing.cpu_count()
	log.info("CPU Count: {0}".format(cpu_count))
	pool = multiprocessing.Pool(cpu_count * 2)

	if data_arg == "subset":
		# subset data
		# create directory for images
		tf.io.gfile.makedirs(image_file_path)
		log.info('Downloading Images')
		# create partial function to download images
		download_func = partial(download_img, image_file_path=image_file_path, arg_data=data_arg)
		# download images in parallel
		pool.map(download_func, df['ImageNo'].values)
	else:
		# all data
		log.info('Downloading and extracting .tgz')
		start = time.time()
		# download and extract .tgx file
		_ = tf.keras.utils.get_file("images",
		                            origin="https://storage.googleapis.com/lovelace-data/all/images.tgz",
		                            extract=True)
		image_file_path = "/root/.keras/datasets/train_images/"
		print("time take: {0}".format(time.time() - start))

	# close parallel pool
	pool.close()
	print(image_file_path)

	# encode labels in dataframe
	log.info('Encoding labels')
	encode_data(df)

	# create new parallel pool
	pool = multiprocessing.Pool(cpu_count * 2)
	log.info('Getting Image Data')
	# get pixel data from images
	image_arr_func = partial(get_image_arr, base_path=image_file_path)
	images = df['ImageNo'].values
	image_arr = pool.map(image_arr_func, images)
	image_arr = np.asarray(image_arr, dtype='uint8')

	# stack labels
	labels = df['EncodedTag'].values
	labels = np.stack(labels, axis=0)

	# create and save .npz file
	log.info('Saving NPZ...')
	np.savez_compressed('full_data.npz', image_arr, labels)
	log.info('Uploading to cloud')
	os.system("gsutil cp full_data.npz gs://lovelace-data/all/")
	# upload_file.upload_file('lovelace-data', 'full_data.npz', data_arg)
	return image_arr, labels


# function to download npz
def download_npz(data_arg):
	# local path
	path = './full_data.npz'

	# download npz
	urllib.request.urlretrieve('https://storage.googleapis.com/lovelace-data/{0}/full_data.npz'.format(data_arg), path)

	# load data
	data = np.load(path)
	image, label = data['arr_0'], data['arr_1']
	return image, label


# entry point to loading data
def load_data(data_arg):
	# check if npz exists
	request = requests.get(NPZ_URL.format(data_arg))
	if request.status_code == 200:
		log.info('Loading NPZ')
		# if npz file found, download it
		image, label = download_npz(data_arg)
	else:
		log.info('Loading CSV and images')
		# load csv and image data
		image, label = __load_data(data_arg)

	# log datashape for debugging
	log.info("Data shapes: {0}, {1}".format(image.shape, label.shape))
	# image = np.repeat(image[..., np.newaxis], 3, -1)
	# image = image[:, :, :, 0, :]
	# log.info(image.shape)

	# split data into train, text and validation
	x_train, x_test, y_train, y_test = train_test_split(image, label, random_state=42, test_size=0.3)
	del image, label
	gc.collect()
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=42, test_size=0.2)
	return x_train, y_train, x_test, y_test, x_val, y_val
