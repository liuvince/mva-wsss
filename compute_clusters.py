import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--features_dir", type=str, default="experiment/vit_features_dir")
parser.add_argument("--data_train_in", type=str, default="experiment/data_train.csv")
parser.add_argument("--data_val_in", type=str, default="experiment/data_test.csv")
parser.add_argument("--data_train_out", type=str, default="experiment/data_train_clusters.csv")
parser.add_argument("--data_val_out", type=str, default="experiment/data_test_clusters.csv")
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

kd_dir = args.features_dir
train_aug = pd.read_csv(args.data_train_in)
test = pd.read_csv(args.data_val_in)

filenames = []
data = []
for id in tqdm(sorted(os.listdir(kd_dir))):
    filenames.append(os.path.splitext(id)[0])
    data.append(np.load(os.path.join(kd_dir, id)))
data = np.concatenate(data, axis=0)
df_data = pd.DataFrame({'tmp_filename': filenames})
for i in range(data.shape[1]):
    df_data[str(i)] = data[:, i]

train_aug['tmp_filename'] = train_aug['filename'].apply(
    lambda filename: os.path.splitext(os.path.basename(filename))[0])
test['tmp_filename'] = test['filename'].apply(lambda filename: os.path.splitext(os.path.basename(filename))[0])
train_aug = train_aug.sort_values(by='tmp_filename')
test = test.sort_values(by='tmp_filename')
train_aug['train_data'] = 1
test['train_data'] = 0
dataset = pd.concat([train_aug, test], axis=0)

dataset_cluster = pd.DataFrame()
dataset_cluster['tmp_filename'] = dataset['tmp_filename']

dataset_cluster = dataset_cluster.sort_values(by='tmp_filename')
dataset = dataset.sort_values(by='tmp_filename')
df_data = df_data.sort_values(by='tmp_filename')

from FINCH_Clustering.python.finch import FINCH

for i in tqdm(range(1, 21)):
    idx_bool = dataset.iloc[:, i].values == 1
    c, num_clust, req_c = FINCH(df_data.values[np.where(idx_bool)[0], 1:], verbose=False)
    for j in range(num_clust[-1]):
        dataset_cluster.loc[:, "cluster_{}_{}".format(i - 1, j)] = 0
        dataset_cluster.loc[idx_bool, "cluster_{}_{}".format(i - 1, j)] = pd.get_dummies(c[:, -1]).iloc[:, j].values

new_dataset = dataset.merge(dataset_cluster, on='tmp_filename').drop(
    columns=['tmp_filename'])

new_dataset[new_dataset['train_data'] == 1].drop(columns='train_data').to_csv(args.data_train_out, index=None)
new_dataset[new_dataset['train_data'] == 0].drop(columns='train_data').to_csv(args.data_val_out, index=None)

print(new_dataset.shape[1] - 21)