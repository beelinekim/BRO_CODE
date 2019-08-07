import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import glob
import os
from posixpath import join
import cv2
import gc
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

data_path = 'D:/kaggle_data'
train_img_path = join(data_path, 'train')
class_csv = 'D:/kaggle_data/class.csv'
train_csv = 'D:/kaggle_data/train.csv'
test_csv = 'D:/kaggle_data/test.csv'
train_list = glob.glob('D:/kaggle_data/train/*')
resize_dir_path = join(data_path, 'resized_images')
seed = 196

image_size = (300, 300)

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
                                   preprocessing_function=preprocess_input)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# train generator 생성
train_generator = train_datagen.flow_from_dataframe(dataframe=train_data,
                                                    directory=resize_dir_path,
                                                    x_col='img_file',
                                                    y_col='class',
                                                    target_size=image_size,
                                                    color_mode='rgb',
                                                    class_mode='categorical',
                                                    batch_size=32,
                                                    seed=seed)

valid_generator = val_datagen.flow_from_dataframe(dataframe=val_data,
                                                  directory=resize_dir_path,
                                                  x_col='img_file',
                                                  y_col='class',
                                                  target_size=image_size,
                                                  color_mode='rgb',
                                                  class_mode='categorical',
                                                  batch_size=32,
                                                  shuffle=True,
                                                  seed=seed)

test_generator = val_datagen.flow_from_dataframe(dataframe=test_data,
                                                 directory=resize_dir_path,
                                                 x_col='img_file',
                                                 y_col=None,
                                                 target_size=image_size,
                                                 color_mode='rgb',
                                                 class_mode=None,
                                                 batch_size=32,
                                                 shuffle=True,
                                                 seed=seed)

# model 생성 - VGG16
# base_model = VGG16(weights='imagenet',
#                    include_top=False,
#                    input_shape=(100, 100, 3))

# model gen
inputs = keras.Input(shape=(300, 300, 3), name='img')
x = layers.Conv2D(64, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(10, activation='softmax')(x)

base_model = keras.Model(inputs, outputs, name='toy_resnet')

prediction = Dense(196, activation='softmax', kernel_initializer='he_normal')(base_model.output)

# 최종 모델 생성
model = Model(base_model.input, prediction)
model.summary()
model.compile(optimizer=Adam(lr=0.0001, epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])

# callback 설정
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.0001)
filepath = 'model_{val_acc:.2f}_{val_loss:.2f}.h5'
model_ckpt = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
callbacks = [learning_rate_reduction, model_ckpt]

# steps_per_epoch 설정 함수
def get_steps(num_samples, batch_size):
    if (num_samples % batch_size) > 0 :
        return (num_samples // batch_size) + 1
    else :
        return num_samples // batch_size

# 학습과정 데이터 저장
history = model.fit_generator(train_generator,
                              steps_per_epoch=get_steps(len(train_data), 32),
                              validation_data=valid_generator,
                              validation_steps=get_steps(len(val_data), 32),
                              epochs=2,
                              callbacks=callbacks,
                              verbose=1)

gc.collect()

test_generator.reset()    #Generator 초기화
prediction = model.predict_generator(generator=test_generator,
                                     steps=get_steps(len(test_data), 32),
                                     verbose=1
                                     )

