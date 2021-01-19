import os
import numpy as np
from scipy.io import loadmat
import xml.etree.ElementTree as ET
from tqdm import tqdm
from settings import annotation_dir, image_dir, dataset_split, classes

print("1.Process VOC...")
print("Read labels for each image ...")
annotation_paths = [os.path.join(annotation_dir, path) for path in os.listdir(annotation_dir)]
annotations = []
ids = []
for annotation_path in tqdm(annotation_paths):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    annotation = []
    for obj in root.findall('object'):
        annotation.append(obj.findtext('name'))
    annotations.append(annotation)
    ids.append(os.path.basename(os.path.splitext(annotation_path)[0]))
unique_annotations = [list(set(annotation)) for annotation in annotations]

from sklearn.preprocessing import MultiLabelBinarizer

print("Binarize the labels ...")
mlb = MultiLabelBinarizer()
mlb.fit([[c] for c in classes])
binary_annotations = mlb.transform(unique_annotations)

import pandas as pd

print("Create dataframe ...")
df_labels = pd.DataFrame({'filename': ids})
for i, c in enumerate(mlb.classes_):
    df_labels[c] = binary_annotations[:, i]

print('Split dataframe into train/val ...')
df_train_split = pd.read_csv(os.path.join(dataset_split, 'train.txt'), header=None)
df_val_split = pd.read_csv(os.path.join(dataset_split, 'val.txt'), header=None)

df_train_split['train_set'] = 1
df_val_split['train_set'] = 0
df_train_val_split = pd.concat([df_train_split, df_val_split])
df_train_val_split.columns = ['filename', 'train_set']
df_train_val_split['filename'] = df_train_val_split['filename'].astype(str)

df = df_train_val_split.merge(df_labels, on='filename')
df['filename'] = df['filename'].apply(lambda f: os.path.join(image_dir, f + ".jpg"))

df_train = df[df['train_set'] == 1]
df_test = df[df['train_set'] == 0]

df_train = df_train.drop(columns=['train_set'])
df_test = df_test.drop(columns=['train_set'])

if not os.path.exists('experiment'):
    os.mkdir('experiment')
df_train.to_csv("experiment/data_train.csv", index=None)
df_test.to_csv("experiment/data_test.csv", index=None)

print('Process SBD training augmentated data...')
train_aug = pd.concat([pd.read_csv("data/benchmark/benchmark_RELEASE/dataset/train.txt", header=None),
                          pd.read_csv("data/benchmark/benchmark_RELEASE/dataset/val.txt", header=None)]).values.ravel()
train = df_train['filename'].apply(lambda row: os.path.splitext(os.path.basename(row))[0]).values
test = df_test['filename'].apply(lambda row: os.path.splitext(os.path.basename(row))[0]).values
full_data = (list(set(train_aug).union(set(train)) - set(test) ))

filenames = []
annotations = []
np_classes = np.array(classes)
for id in tqdm(set(np.intersect1d(full_data, train_aug))-set(train)):
    label_path = "data/benchmark/benchmark_RELEASE/dataset/inst/{}.mat".format(id)
    filename = "data/benchmark/benchmark_RELEASE/dataset/img/{}.jpg".format(id)

    annotations.append(np_classes[loadmat(label_path)['GTinst'][0, 0][2].ravel().astype(int) - 1])
    filenames.append(filename)
unique_aug_annotations = [list(set(annotation)) for annotation in annotations]
binary_aug_annotations = mlb.transform(unique_aug_annotations)

print("Create dataframe ...")
df_aug_labels = pd.DataFrame({'filename': filenames})
for i, c in enumerate(mlb.classes_):
    df_aug_labels[c] = binary_aug_annotations[:, i]
df_aug_labels = pd.concat([df_train, df_aug_labels])
df_aug_labels.to_csv("experiment/data_train_aug.csv", index=None)