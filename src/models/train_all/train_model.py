import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


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
print(raw_data.head(15))

train_data, test_data = train_test_split(raw_data, train_size=0.7, shuffle=True)

image_train_gen = ImageDataGenerator(vertical_flip=True, validation_split=0.25, rescale=1.0/255.0)

train_generator = image_train_gen.flow_from_dataframe(
    dataframe=train_data,
    directory='data/raw/train_images/',
    x_col='ImageNo',
    y_col='EncodedTag',
    subset='training',
    batch_size=50,
    seed=seed,
    shuffle=True,
    color_mode='grayscale',
    validate_filenames=False,
    class_mode='categorical',
    target_size=(224, 224))

valid_generator = image_train_gen.flow_from_dataframe(
    dataframe=train_data,
    directory='data/raw/train_images/',
    x_col='ImageNo',
    y_col='EncodedTag',
    subset='validation',
    batch_size=50,
    seed=seed,
    shuffle=True,
    color_mode='grayscale',
    validate_filenames=False,
    class_mode='categorical',
    target_size=(224, 224))


def create_model():
    model = Sequential()
    model.add(Conv2D(421, (3, 3), padding='same', input_shape=(224, 224, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(842, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(842, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = create_model()
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
# STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=10,
    use_multiprocessing=True,
    verbose=2)

# serialize model to Json
model_json = model.to_json()
with open("models/baseline_0.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("models/baseline_0.h5")
print("saved model to disk")