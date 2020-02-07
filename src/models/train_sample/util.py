import numpy as np
import pandas as pd
import tensorflow as tf

import tempfile
import os
import urllib

DATA_DIR = os.path.join(tempfile.gettempdir(), 'haemorrhage_data')
DATA_CSV = 'subset/train_labels.csv'

def download_csv(data_dir):
	tf.gfile.MakeDirs(data_dir)


def preprocess_csv(dataframe):

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
		labels_dict, inv_map = create_encoder_mapping(data)
		data['EncodedTag'] = data.apply(lambda row: encode(row['Tags'], labels_dict), axis=1)
		data['ImageNo'] = data['ImageNo'].apply(append_png)
		return labels_dict, inv_map

	a, b = encode_data(dataframe)

def load_data():
