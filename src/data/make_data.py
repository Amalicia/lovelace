import pandas as pd
import numpy as np
import enum
import os


class DataType(enum.Enum):
    train = 1
    test = 2


def process_labels(d_type):
    df = pd.read_csv('data/raw/{}_labels.csv'.format(d_type.name), index_col=False)
    df[['ID', 'ImageNo', 'Haemorrhage_Type']] = df['ID'].str.split('_', expand=True)
    df = df[['ImageNo', 'Haemorrhage_Type', 'Label']]
    df.drop_duplicates(inplace=True)
    df = df.pivot(index='ImageNo', columns='Haemorrhage_Type', values='Label').reset_index()
    df['ImageNo'] = 'ID_' + df['ImageNo']

    x = df.values

    bad_rows = []

    for i in range(x[:, 0].size):
        if i % 100000 == 0:
            print('reached {0}'.format(i))
        if os.path.isfile('data/raw/{0}_images/{1}.png'.format(d_type.name, x[i, 0])):
            continue
        else:
            bad_rows.append(i)

    df.drop(index=bad_rows, inplace=True)
    df.to_csv(r'data/processed/{}_labels.csv'.format(d_type.name), index=None, header=True)


process_labels(DataType.train)
process_labels(DataType.test)
