import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


def append_png(image):
    return image + '.png'


train_data = pd.read_csv('data/processed/train_labels.csv', index_col=None)
train_data['ImageNo'].apply(append_png)

image_gen = ImageDataGenerator(vertical_flip=True, validation_split=0.25)

train_generator = image_gen.flow_from_dataframe(
    dataframe=train_data,
    directory='data/raw/train_images/',
    x_col='ImageNo',
    y_col=['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'],
    subset='training',
    batch_size=64,
    seed=7,
    shuffle=True,
    color_mode='grayscale',
    validate_filenames=False)

valid_generator = image_gen.flow_from_dataframe(
    dataframe=train_data,
    directory='data/raw/train_images/',
    x_col='ImageNo',
    y_col=['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'],
    subset='validation',
    batch_size=64,
    seed=7,
    shuffle=True,
    color_mode='grayscale',
    validate_filenames=False)
