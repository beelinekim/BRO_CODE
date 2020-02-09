import pandas as pd
import os
from posixpath import join
import numpy as np
import cv2

# train > image_id, grapheme_root, vowel_diacritic, consonant_diacritic, grapheme
# parquet > imageid, image

def read_parquet(paths, HEIGHT, WIDTH):
    df0 = pd.read_parquet(paths[0])
    df1 = pd.read_parquet(paths[1])
    df2 = pd.read_parquet(paths[2])
    df3 = pd.read_parquet(paths[3])
    data_full = pd.concat([df0, df1, df2, df3], ignore_index=True)
    return data_full.iloc[:, 0], data_full.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

def crop(img):
    foreground = np.where(img != 1)
    return img[np.min(foreground[0]) : np.max(foreground[0]) + 1,
           np.min(foreground[1]) : np.max(foreground[1]) + 1]

def resize(img, size, pad=16):
    # npad = ((pad, pad), (pad, pad))
    # return np.pad(cv2.resize(img, (size, size)), npad, mode='constant', constant_values=np.max(img))
    return cv2.resize(img, (size, size))

def normalize(img):
    img = img / np.max(img)
    return img
