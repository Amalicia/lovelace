import pandas as pd
import numpy as np
import tensorflow as tf

seed = 7
np.random.seed(seed)

raw_data = pd.read_csv('data/processed/train_labels.csv', index_col=None, dtype=str)
raw_data.fillna('', inplace=True)

def append_png(image):
    return image + '.png'


def create_encoder_mapping(data):
    labels = set()
    for i in range(len(data)):
        labels.update(data['Tags'][i].split(' '))

    labels = list(labels)
    labels.sort()

    labels_dict = {labels[i]:i for i in range(len(labels))}
    inv_map = {v:k for k, v in labels_dict.items()}
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


a, b = encode_data(raw_data)
