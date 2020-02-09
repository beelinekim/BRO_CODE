import pandas as pd
import numpy as np
from glob import glob
from posixpath import join
import matplotlib.pyplot as plt
import os

HEIGHT = 137
WIDTH = 236

data_dir = 'D:/workspace/kaggle/bengali/bengaliai-cv19'
parquet_lst = glob(join(data_dir, 'train_image_data_0.parquet'))

# view files
for dirname, _, filenames in os.walk(data_dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# parquet read, parquet reshape
def read_parquet(file):
    df = pd.read_parquet(file)
    return df.iloc[:, 0], df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

image_index, images = read_parquet(parquet_lst[0])

fig, ax = plt.subplots(5, 5, figsize=(16, 8))
ax = ax.flatten()

# image view 25개
for i in range(25):
    ax[i].imshow(images[i], cmap='Greys')
plt.show()

train_df = pd.read_csv(join(data_dir, 'train.csv'))
# 5개 정도 데이터를 확인
train_df.head()

# train csv의 형태
train_df.shape

class_map_df = pd.read_csv(join(data_dir, 'class_map.csv'))



