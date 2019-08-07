import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import glob
import os
from posixpath import join
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

data_path = 'D:/kaggle_data'
train_img_path = join(data_path, 'train')
class_csv = 'D:/kaggle_data/class.csv'
train_csv = 'D:/kaggle_data/train.csv'
test_csv = 'D:/kaggle_data/test.csv'
train_list = glob.glob('D:/kaggle_data/train/*')
resize_dir_path = join(data_path, 'resized_images')
seed = 196

image_size = (224, 224)
batch_size = 32

df_class = pd.read_csv(class_csv)
df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

if os.path.exists(resize_dir_path) is False:
    os.mkdir(resize_dir_path)

# resize
for i in df_train['img_file']:
    if os.path.exists(join(resize_dir_path, i)) is False:
        img = join(train_img_path, i)
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img, image_size)
        save_name = join(resize_dir_path, i)
        cv2.imwrite(save_name, resized_img)

# type 변환
df_train['class'] = df_train['class'].astype('str')

# data split
train_data, val_data = train_test_split(df_train[['img_file', 'class']],
                                          train_size=0.9, random_state=seed, stratify=df_train['class'])
test_data = df_test[['img_file']]

# image data generator 설정 정의
train_datagen = ImageDataGenerator(horizontal_flip=True,    #수평 반전
                                   zoom_range=0.15,        #확대 & 축소
                                   width_shift_range=0.1,  #수평방향 이동
                                   height_shift_range=0.1,  #수직방향 이동
                                   rescale=1./255,
                                   preprocessing_function=preprocess_input)

val_datagen = ImageDataGenerator(rescale=1./255,
                                 preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(rescale=1./255,
                                  preprocessing_function=preprocess_input)

# train generator 생성
train_generator = train_datagen.flow_from_dataframe(dataframe=train_data,
                                                    directory=resize_dir_path,
                                                    x_col='img_file',
                                                    y_col='class',
                                                    target_size=image_size,
                                                    color_mode='rgb',
                                                    class_mode='categorical',
                                                    batch_size=batch_size,
                                                    seed=seed)

valid_generator = val_datagen.flow_from_dataframe(dataframe=val_data,
                                                  directory=resize_dir_path,
                                                  x_col='img_file',
                                                  y_col='class',
                                                  target_size=image_size,
                                                  color_mode='rgb',
                                                  class_mode='categorical',
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  seed=seed)

test_generator = val_datagen.flow_from_dataframe(dataframe=test_data,
                                                 directory=resize_dir_path,
                                                 x_col='img_file',
                                                 y_col=None,
                                                 target_size=image_size,
                                                 color_mode='rgb',
                                                 class_mode=None,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 seed=seed)

from tensorflow.compat.v1.keras.applications import ResNet50
base_model = ResNet50(include_top=True, weights='imagenet')
x = base_model.layers[-2].output
del base_model.layers[-1:]
x = Dense(196, activation='softmax', name='predictions')(x)

model = Model(base_model.input, x)
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.0001)
filepath = 'model_{val_acc:.2f}_{val_loss:.2f}.h5'
# early_stopping = EarlyStopping()


def get_steps(num_samples, batch_size):
    if (num_samples % batch_size) > 0 :
        return (num_samples // batch_size) + 1
    else :
        return num_samples // batch_size


callbacks = [EarlyStopping(monitor='val_acc',
                           patience=10,
                           mode='max',
                           verbose=1),
             ReduceLROnPlateau(monitor='val_acc',
                               factor=.5,
                               patience=5,
                               min_lr=0.00001,
                               mode='max',
                               verbose=1),
             ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
             ]
model.fit_generator(train_generator,
                    epochs=10000,
                    validation_data=valid_generator,
                    verbose=1,
                    callbacks=callbacks,
                    validation_steps=get_steps(len(val_data), batch_size),
                    steps_per_epoch=get_steps(len(train_data), batch_size)
                    )

test_generator.reset()    #Generator 초기화
prediction = model.predict_generator(
    generator=test_generator,
    steps=get_steps(len(test_data), batch_size),
    verbose=1
)

predicted_class_indices = np.argmax(prediction, axis=1)

# Generator class dictionary mapping
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

submission = pd.read_csv(join(data_path, 'sample_submission.csv'))
submission["class"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()