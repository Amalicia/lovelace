import pandas as pd


def label_haemorrhage(row):
    labels = []
    if row['epidural'] == 1:
        labels.append('epidural')
    if row['intraparenchymal'] == 1:
        labels.append('intraparenchymal')
    if row['intraventricular'] == 1:
        labels.append('intraventricular')
    if row['subarachnoid'] == 1:
        labels.append('subarachnoid')
    if row['subdural'] == 1:
        labels.append('subdural')
    return " ".join(label for label in labels)


raw_data = pd.read_csv('data/interim/train_labels.csv', index_col=None)
raw_data['Tags'] = raw_data.apply(lambda row: label_haemorrhage(row), axis=1)
raw_data = raw_data[['ImageNo', 'Tags']]

raw_data.to_csv(r'data/processed/train_labels.csv', index=None, header=True)