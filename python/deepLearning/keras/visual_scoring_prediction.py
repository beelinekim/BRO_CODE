#################### visual scoring training

import numpy as np
import nibabel as nib
from posixpath import join
from tensorflow.keras import models, optimizers, Input, Sequential, Model, utils, activations
from tensorflow.keras.layers import Layer, Add, Activation, Softmax, Flatten, Dense, Conv2D, MaxPool2D, \
    BatchNormalization, GlobalAveragePooling2D, AveragePooling2D, concatenate, Lambda, MaxPooling2D, Dropout, \
    Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import Sequence
import pandas as pd
from ast import literal_eval
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
import pickle
import os
from glob import glob
import tensorflow.keras.backend as K
from scipy.ndimage import rotate

# RTX option
K.set_floatx('float16')
K.set_epsilon(0.0001)


# GPU 0번 사용
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# data generator 3D
class DataGenerator(Sequence):
    def __init__(self, image1_list=None, image2_list=None, image3_list=None, image4_list=None, field_data=None, label_data=None,
                 aug=None, dim=None, batchsize=1, channels=None, classes=None, shuffle=None):
        super(DataGenerator, self).__init__()
        self.image1_list = image1_list
        self.image2_list = image2_list
        self.image3_list = image3_list
        self.image4_list = image4_list
        self.label_data = pd.read_excel(label_data)
        self.field_data = field_data
        self.aug = aug
        self.dim = dim
        self.channels = channels
        self.classes = classes
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.image1_list) // self.batchsize

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image1_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        global x_json, y
        indexes = self.indexes[idx * self.batchsize: (idx + 1) * self.batchsize]
        x = np.array([self.preprocessing(self.image1_list[i], self.image2_list[i], self.image3_list[i], self.image4_list[i])
                      for i in indexes])
        y_label1 = self.label_data['label1']
        y_label2 = self.label_data['label2']
        y_label3 = self.label_data['label3']
        y_label1 = [y_label1[k] for k in indexes]
        y_label2 = [y_label2[k] for k in indexes]
        y_label3 = [y_label3[k] for k in indexes]
        y_label1 = np.array(y_label1)
        y_label2 = np.array(y_label2)
        y_label3 = np.array(y_label3)

        if self.aug == True:
            # cutmix
            x, y = self.cutmix(x, [y_label1, y_label2, y_label3])
            x = [x[:, :, i, :, :] for i in range(18)]

        elif self.aug == None:
            x = [x[:, :, i, :, :] for i in range(18)]

        # using field data
        if self.field_data is not None:
            image3_volume_label1 = [literal_eval(open(self.field_data[j]).read())['label1'] for j in indexes]
            peri_volume = [literal_eval(open(self.field_data[j]).read())['label2'] for j in indexes]
            deep_volume = [literal_eval(open(self.field_data[j]).read())['label3'] for j in indexes]
            x_json = np.concatenate([image3_volume_label1, peri_volume, deep_volume])
            x_json = np.expand_dims(x_json, axis=0)
            x.append(x_json)

        # image4 crop 임시 저장(heatmap processing을 위해)
        self.image4_crop = [self.save_crop_3d(self.image4_list[i], mode='fix') for i in indexes]

        if self.aug is True:
            return x, y
        elif self.aug is None:
            return x, [utils.to_categorical(y_label1, self.classes),
                       utils.to_categorical(y_label2, self.classes),
                       utils.to_categorical(y_label3, self.classes)]

    def cutmix(self, x=None, y=None, prob=.5):
        global mix1_label
        for i in range(self.batchsize):
            if np.random.random() <= prob:
                cut_point = self.dim[0] // 2
                candidate = np.random.choice(self.indexes, 1)[0]
                # x
                mix1 = x[-1, ...][:cut_point, ...]
                mix2 = np.array(self.preprocessing(self.image1_list[candidate], self.image2_list[candidate],
                                                   self.image3_list[candidate], self.image4_list[candidate]))[cut_point:, ...]
                # mix2 = nib.load(self.image1_list[candidate]).get_fdata()[cut_point:, ...]
                x[i] = np.concatenate([mix1, mix2], axis=0)
                x[i] = np.expand_dims(x[i], axis=0)
                # y
                for iter, scale_name in zip(range(len(y)), ['label1', 'label2', 'label3']):
                    if y[iter][i] != self.label_data[scale_name][candidate]:
                        mix1_label = utils.to_categorical(y[iter], self.classes)
                        mix2_label = utils.to_categorical(self.label_data[scale_name][candidate], self.classes)
                        y[iter] = (mix1_label + mix2_label) / 2

                    else:
                        y[iter] = utils.to_categorical(y[iter], self.classes)
            else:
                for y_iter, y_element in enumerate(y):
                    y[y_iter] = utils.to_categorical(y_element, self.classes)

        return x, y

    def rotation_3d(self, img, angle=None, axes=(0, 1), mode='nearest'):
        return rotate(img, angle, reshape=False, mode=mode, axes=axes)

    def random_rotation_3d(self, x, mode='nearest', prob=.5):
        for i in range(self.batchsize):
            if np.random.random() <= prob:
                angle_range = (0, 20)
                angle_lst = [np.random.randint(*(angle_range)) for _ in range(3)]
                axes_lst = [(0, 1), (0, 2), (1, 2)]
                np.random.shuffle(axes_lst)

                x[i] = self.rotation_3d(x[i], angle_lst[0], axes_lst[0], mode)
                x[i] = self.rotation_3d(x[i], angle_lst[1], axes_lst[1], mode)
                x[i] = self.rotation_3d(x[i], angle_lst[2], axes_lst[2], mode)
                x[i] = np.expand_dims(x[i], axis=0)

        return x

    def horizontal_flip_3d(self, x, prob=.5):
        for i in range(self.batchsize):
            if np.random.random() <= prob:
                x[i] = x[i][:, :, ::-1]
                x[i] = np.expand_dims(x[i], axis=0)
        return x

    def preprocessing(self, image1, image2, image3, image4):
        # crop image
        image1_img = self.save_crop_3d(image1, mode='fix')
        image2_img = self.save_crop_3d(image2, mode='fix')
        image3_img = self.save_crop_3d(image3, mode='fix')
        image4_img = self.save_crop_3d(image4, mode='fix')

        # percentile normalize
        image1_per_norm = self.percentile_normalization(image1_img)
        image2_per_norm = self.percentile_normalization(image2_img)

        # intensity normalize
        image1_intnorm = self.intensity_normalization(image1_per_norm, image4_img, 1)
        image2_intnorm = self.intensity_normalization(image2_per_norm, image4_img, 1)

        # min_max normalize
        image1_norm = self.max_norm(image1_intnorm)
        image2_norm = self.max_norm(image2_intnorm)
        image3_norm = self.max_norm(image3_img)

        # concatenate
        concatenated_img = self.concat_nifti_files(image1_norm, image2_norm, image3_norm)
        concatenated_img = concatenated_img.get_fdata()
        return concatenated_img

    def percentile_normalization(self, nifti, percentile: int = 1):
        """percentile min-max normalization from mricron

        Args:
            nifti (np.ndarray): input array
            percentile (int, optional): Defaults to 1.

        Returns:
            np.array: normalized nifti
        """
        nifti_array = nifti.get_fdata()
        min_percentile = np.percentile(nifti_array, percentile)
        max_percentile = np.percentile(nifti_array, 100 - percentile)

        # limit maximum intensity of nifti by max_percentile
        nifti_array[nifti_array >= max_percentile] = max_percentile

        # limit minimum intensity of nifti by min_percentile
        nifti_array[nifti_array <= min_percentile] = min_percentile

        nifti = nib.Nifti1Image(nifti_array, nifti.affine, nifti.header)

        return nifti

    def intensity_normalization(self, img, image4, tmean=None):
        if tmean is None:
            ValueError("Please type the tmean.")

        # icv 아닌 부분 제거
        non_icv_img = self.remove_nonicv(img, image4)
        non_icv_img_data = non_icv_img.get_fdata()

        # 평균 계산
        img_mean = np.sum(non_icv_img_data) / np.count_nonzero(non_icv_img_data)

        # 평균과 tmean의 비율 계산
        ratio = tmean / img_mean

        # tmean : 평균과 tmean의 비율을 전체 복셀에 적용
        result = non_icv_img.get_fdata() * ratio

        result = nib.Nifti1Image(result, img.affine, img.header)
        return result

    def remove_nonicv(self, img, image4):
        img_data = img.get_fdata()
        image4_data = image4.get_fdata()

        # CSF 부분까지 1 나머지 0
        icv = np.where((image4_data == 1) | (image4_data == 2) | (image4_data == 3) |
                       (image4_data == 4) | (image4_data == 5) | (image4_data == 6), 1, image4_data)
        icv = np.where((icv != 1), 0, icv)

        # 영상과 icv를 곱해서 icv가 0인 부분을 제거
        result_nii = img_data * icv
        result_nii = nib.Nifti1Image(result_nii, img.affine, img.header)

        return result_nii

    def save_crop_3d(self, img_file, minimum=None, maximum=None, mode=None):
        global cropped_img_data

        img = nib.load(img_file)
        img_data = img.get_fdata()

        # crop
        # min, max fix
        if mode == 'fix':
            cropped_img_data = img_data[17:181, 4:22, 11:188]
        # save cropped image
        cropped_img_data = nib.Nifti1Image(cropped_img_data, img.affine, img.header)

        return cropped_img_data

    def concat_nifti_files(self, image1, image2, image3):
        image1_img = image1.get_fdata()
        image2_img = image2.get_fdata()
        image3_img = image3.get_fdata()
        img = np.stack((image3_img, image2_img, image1_img), axis=-1)
        concat_img = nib.Nifti1Image(img, image2.affine, image2.header)
        return concat_img

    def max_norm(self, img):
        save_max_value = []
        img_data = img.get_fdata()
        max_value = np.max(img_data)
        save_max_value.append(max_value)
        result = img_data / np.max(save_max_value)
        result = nib.Nifti1Image(result, img.affine, img.header)
        return result

    def crop_foreground_3d(self, img, edge_only=True):
        """ Crop 3-dimensional image into non-zero foreground
       Args:
           img (np.array):  3-dimensional array
           edge_only (bool, optional): only returns top and bottom edges
       Returns:
           img (np.array): cropped img
           edges (tuple(np.array, np.array)): top and bottom edge of img
       """
        true_points = np.argwhere(img)
        top_edge = true_points.min(axis=0)
        bottom_edge = true_points.max(axis=0)
        if edge_only:
            return top_edge, bottom_edge
        # coordinate return
        return img[top_edge[0]:bottom_edge[0],
               top_edge[1]:bottom_edge[1],
               top_edge[2]:bottom_edge[2]]


data_path = 'D:/workspace'
train_image1_list = glob(join(data_path, 'train/*/store/nifti/image1_mni6.nii.gz'))
train_image2_list = glob(join(data_path, 'train/*/store/nifti/image2_mni6.nii.gz'))
train_image3_list = glob(join(data_path, 'train/*/store/nifti/image3_lesion_divided.nii.gz'))
train_image4_list = glob(join(data_path, 'train/*/store/nifti/image4_mni6.nii.gz'))
valid_image1_list = glob(join(data_path, 'valid/*/store/nifti/image1_mni6.nii.gz'))
valid_image2_list = glob(join(data_path, 'valid/*/store/nifti/image2_mni6.nii.gz'))
valid_image3_list = glob(join(data_path, 'valid/*/store/nifti/image3_lesion_divided.nii.gz'))
valid_image4_list = glob(join(data_path, 'valid/*/store/nifti/image4_mni6.nii.gz'))
train_label_data = join(data_path, 'visual_scoring_train.xlsx')
valid_label_data = join(data_path, 'visual_scoring_valid.xlsx')
image_shape = (164, 18, 177)

seed = 33
batchsize = 1

train_generator = DataGenerator(image1_list=train_image1_list,
                                image2_list=train_image2_list,
                                image3_list=train_image3_list,
                                image4_list=train_image4_list,
                                label_data=train_label_data,
                                aug=True,
                                dim=image_shape,
                                classes=4,
                                channels=3,
                                shuffle=True,
                                batchsize=batchsize
                                )

valid_generator = DataGenerator(image1_list=valid_image1_list,
                                image2_list=valid_image2_list,
                                image3_list=valid_image3_list,
                                image4_list=valid_image4_list,
                                label_data=valid_label_data,
                                aug=None,
                                dim=image_shape,
                                classes=4,
                                channels=3,
                                shuffle=False,
                                batchsize=batchsize
                                )

initializer = initializers.he_normal()
activation = 'relu'
regularizer = regularizers.l2(0.)


def conv_block(input, name='CNN'):
    out = Conv2D(64, 3, 1, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name=name + '_conv1')(input)
    out = BatchNormalization(name=name + '_bn1')(out)
    out = Activation(activation)(out)
    out = MaxPooling2D(2, padding='same')(out)

    out = Conv2D(128, 3, 1, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name=name + '_conv2')(out)
    out = BatchNormalization(name=name + '_bn2')(out)
    out = Activation(activation)(out)
    out = MaxPooling2D(2, padding='same')(out)

    out = Conv2D(256, 3, 1, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name=name + '_conv3')(out)
    out = BatchNormalization(name=name + '_bn3')(out)
    out = Activation(activation)(out)
    out = MaxPooling2D(2, padding='same')(out)

    return out


def conv_dense_block(inputs, name='CNN'):
    out = Conv2D(128, 3, 1, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name=name + '_conv4')(inputs)
    out = BatchNormalization(name=name + '_bn4')(out)
    out = Activation(activation)(out)
    out = MaxPooling2D(2, padding='same')(out)

    out = Conv2D(64, 3, 1, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name=name + '_conv5')(out)
    out = BatchNormalization(name=name + '_bn5')(out)
    out = Activation(activation)(out)
    out = MaxPooling2D(2, padding='same')(out)

    out = Flatten()(out)

    out = Dense(1024, kernel_initializer=initializer, kernel_regularizer=regularizer)(out)
    out = Activation(activation)(out)
    out = Dropout(0.1)(out)

    out = Dense(200, kernel_initializer=initializer, kernel_regularizer=regularizer)(out)
    out = Activation(activation)(out)
    out = Dropout(0.1)(out)

    out = Dense(4, 'softmax', name=name)(out)

    return out


def model(image_image1, image_input2, image_input3, image_input4, image_input5, image_input6, image_input7,
          image_input8, image_input9, image_image10, image_image11, image_image12, image_image13, image_image14,
          image_image15, image_image16, image_image17, image_image18):
    out1 = conv_block(image_image1, name='CNN1')
    out2 = conv_block(image_input2, name='CNN2')
    out3 = conv_block(image_input3, name='CNN3')
    out4 = conv_block(image_input4, name='CNN4')
    out5 = conv_block(image_input5, name='CNN5')
    out6 = conv_block(image_input6, name='CNN6')
    out7 = conv_block(image_input7, name='CNN7')
    out8 = conv_block(image_input8, name='CNN8')
    out9 = conv_block(image_input9, name='CNN9')
    out10 = conv_block(image_image10, name='CNN10')
    out11 = conv_block(image_image11, name='CNN11')
    out12 = conv_block(image_image12, name='CNN12')
    out13 = conv_block(image_image13, name='CNN13')
    out14 = conv_block(image_image14, name='CNN14')
    out15 = conv_block(image_image15, name='CNN15')
    out16 = conv_block(image_image16, name='CNN16')
    out17 = conv_block(image_image17, name='CNN17')
    out18 = conv_block(image_image18, name='CNN18')

    out = Concatenate()([o for o in [out1, out2, out3, out4, out5, out6, out7, out8, out9,
                                     out10, out11, out12, out13, out14, out15, out16, out17, out18]])

    # out = Concatenate()([Flatten()(o) for o in [out1, out2, out3, out4, out5, out6, out7, out8, out9,
    #                                             out10, out11, out12, out13, out14, out15, out16, out17, out18]])

    out_label1 = conv_dense_block(out, name='label1')
    out_label2 = conv_dense_block(out, name='label2')
    out_label3 = conv_dense_block(out, name='label3')

    return out_label1, out_label2, out_label3


input_image_shape = (164, 177, 3)
image_image1 = Input(shape=input_image_shape)
image_input2 = Input(shape=input_image_shape)
image_input3 = Input(shape=input_image_shape)
image_input4 = Input(shape=input_image_shape)
image_input5 = Input(shape=input_image_shape)
image_input6 = Input(shape=input_image_shape)
image_input7 = Input(shape=input_image_shape)
image_input8 = Input(shape=input_image_shape)
image_input9 = Input(shape=input_image_shape)
image_image10 = Input(shape=input_image_shape)
image_image11 = Input(shape=input_image_shape)
image_image12 = Input(shape=input_image_shape)
image_image13 = Input(shape=input_image_shape)
image_image14 = Input(shape=input_image_shape)
image_image15 = Input(shape=input_image_shape)
image_image16 = Input(shape=input_image_shape)
image_image17 = Input(shape=input_image_shape)
image_image18 = Input(shape=input_image_shape)

out_label1, out_label2, out_label3 = model(image_image1, image_input2, image_input3, image_input4, image_input5,
                                      image_input6, image_input7, image_input8, image_input9, image_image10,
                                      image_image11, image_image12, image_image13, image_image14, image_image15,
                                      image_image16, image_image17, image_image18)

model = Model(inputs=[image_image1, image_input2, image_input3, image_input4, image_input5, image_input6, image_input7,
                      image_input8, image_input9, image_image10, image_image11, image_image12, image_image13,
                      image_image14, image_image15, image_image16, image_image17, image_image18],
              outputs=[out_label1, out_label2, out_label3])
model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['acc'])
model.summary(line_length=130)

modelName = 'ver.json'
weightName = 'ver'


def model_to_json(model, filename):
    model_json = model.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)


model_to_json(model, join('D:/workspace', modelName))

learningRateReduction = ReduceLROnPlateau(monitor='val_loss',
                                          patience=3,
                                          verbose=1,
                                          factor=0.9,
                                          min_lr=0.000001)
earlyStopping = EarlyStopping(monitor='val_loss', patience=15)
modelCkpt = ModelCheckpoint(join('D:/workspace',
                                 weightName + '_{epoch:02d}-{val_loss:.4f}-{val_label1_acc:.4f}.h5'),
                            monitor='val_label1_acc',
                            verbose=1,
                            save_best_only=True)
callbacks = [modelCkpt]

history = model.fit_generator(
    epochs=30,
    generator=train_generator,
    validation_data=valid_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(valid_generator),
    verbose=1,
    callbacks=callbacks
)

# save
save_path = 'D:/workspace'
with open(os.path.join(save_path, weightName + '_history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)
# load
# with open(history_path, 'rb') as f:
#     history = pickle.load(f)

#################### visualization

# load
import pickle
history_path = 'D:/workspace/AI_Framework/AQUA_fazeka/ver2/ver203_history.pkl'
with open(history_path, 'rb') as f:
    history = pickle.load(f)
import matplotlib.pyplot as plt
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(history['loss'], 'y', label='train loss')
loss_ax.plot(history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

fig, acc_ax = plt.subplots()
acc_ax = acc_ax.twinx()
acc_ax.plot(history['label1_acc'], 'b', label='train_label1_acc')
acc_ax.plot(history['peri_acc'], 'g', label='train_label2_acc')
acc_ax.plot(history['deep_acc'], 'y', label='train_label3_acc')
acc_ax.plot(history['val_label1_acc'], 'm', label='val_label1_acc')
acc_ax.plot(history['val_label2_acc'], 'c', label='val_label2_acc')
acc_ax.plot(history['val_label3_acc'], 'black', label='val_label3_acc')
acc_ax.set_xlabel('epoch')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')


###################### inference


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras import utils
from posixpath import join
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from glob import glob
import cv2
from scipy import misc
import json
import pandas as pd

# RTX option
K.set_floatx('floaimage16')
K.set_epsilon(0.0001)

# cudnn initialize failed 방지
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def heatmap_post_processing(heatmap, image4, image3):
    # load
    image4_data = image4[0].get_fdata()
    image3_data = image3[0].get_fdata()
    heatmap_img = nib.load(heatmap)
    heatmap_data = heatmap_img.get_fdata()
    heatmap_thresh = np.where((heatmap_data > np.percentile(heatmap_data, 90)), 0, heatmap_data)

    image4_wm_slices = []
    mul_heat_image3_count = []

    # WM 부분만 heatmap을 뽑을 때 적용하기 위해 WM 복셀의 갯수를 카운트
    for z_slice in range(18):
        image4_wm_slices.append(np.count_nonzero(
            np.where((image4_data != 2), 0, image4_data)[:, z_slice, :]))

    mul_heat_image3 = image3_data * heatmap_thresh

    # WM 복셀의 갯수가 0이 아닌 slice에서, image3와 heatmap의 복셀이 겹치는 수를 카운트
    for slice_idx in np.nonzero(image4_wm_slices)[0]:
        mul_heat_image3_count.append(np.sum(mul_heat_image3[:, slice_idx, :]))
        # heatmap_thresh_count.append(np.count_nonzero(heatmap_data[:, slice_idx, :]))

    # 겹치는 복셀의 갯수가 가장 많은 slice의 index를 추출
    heatmap_max_vox_cnt = mul_heat_image3_count.index(np.max(mul_heat_image3_count))
    heatmap_output = heatmap_thresh[:, heatmap_max_vox_cnt, :]

    # 2d이미지를 만들기 위한 처리
    x = 255 / np.max(heatmap_output)
    heatmap_output = np.uint8(heatmap_output * x)
    heatmap_output = cv2.applyColorMap(heatmap_output, cv2.COLORMAP_JET)
    height, width, channel = heatmap_output.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
    heatmap_output = cv2.warpAffine(heatmap_output, matrix, (width, height))

    # 이미지 저장
    misc.imsave(heatmap.replace('\\', '/').split('/store/nifti/')[0] + '/store/figure/best_heatmap.png', heatmap_output)


def generate_grad_cam(img_tensor, model, class_index, activation_layer):
    """
    params:
    -------
    img_tensor: image
    model: pretrained model
    class_index: true label
    activation_layer: layer name

    return:
    grad_cam: grad_cam 히트맵
    """
    class_index = int(class_index)
    model_input = model.input
    model_inputs = []
    for i in range(18):
        globals()['model_input{}'.format(i)] = model_input[i]
        model_inputs.append(globals()['model_input{}'.format(i)])
    softmax_input = model.output[0].op.inputs[0][0, class_index]
    layer_output = model.get_layer(activation_layer).output

    # 해당 액티베이션 레이어의 아웃풋(a_k)과
    # 소프트맥스 함수 인풋의 a_k에 대한 gradient를 구한다.
    img_tensor = [np.expand_dims(img, axis=0) for img in img_tensor]
    get_output = K.function([model_inputs], [layer_output, K.gradients(softmax_input, layer_output)[0], model.output])
    [conv_output, grad_val, model_output] = get_output([img_tensor])

    # remove batch dim
    conv_output = conv_output[0]
    grad_val = grad_val[0]

    # 구한 gradient를 픽셀 가로세로로 평균내서 weight를 구한다.
    weights = np.mean(grad_val, axis=(0, 1))

    # 추출한 conv_output에 weight를 곱하고 합하여 grad_cam을 얻는다.
    grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        grad_cam += w * conv_output[:, :, k]

    grad_cam = zoom(grad_cam, ((164 / conv_output.shape[0]), (177 / conv_output.shape[1])))

    # ReLU를 씌워 음수를 0으로 만든다.
    grad_cam = np.maximum(grad_cam, 0)

    return grad_cam

# data generator 3D
class DataGenerator(Sequence):
    def __init__(self, image1_list=None, image2_list=None, image3_list=None, image4_list=None, field_data=None, label_data=None,
                 aug=None, dim=None, batchsize=1, channels=None, classes=None, shuffle=None):
        super(DataGenerator, self).__init__()
        self.image1_list = image1_list
        self.image2_list = image2_list
        self.image3_list = image3_list
        self.image4_list = image4_list
        self.label_data = pd.read_excel(label_data)
        self.field_data = field_data
        self.aug = aug
        self.dim = dim
        self.channels = channels
        self.classes = classes
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.image1_list) // self.batchsize

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image1_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        global x_json
        indexes = self.indexes[idx * self.batchsize: (idx + 1) * self.batchsize]
        x = np.array([self.preprocessing(self.image1_list[i], self.image2_list[i], self.image3_list[i], self.image4_list[i])
                      for i in indexes])

        y_label1 = self.label_data['label1']
        y_label2 = self.label_data['label2']
        y_label3 = self.label_data['label3']

        y_label1 = [y_label1[k] for k in indexes]
        y_label2 = [y_label2[k] for k in indexes]
        y_label3 = [y_label3[k] for k in indexes]

        y_label1 = np.array(y_label1)
        y_label2 = np.array(y_label2)
        y_label3 = np.array(y_label3)

        x = [x[:, :, i, :, :] for i in range(18)]

        # image4 crop, image3 crop 임시 저장(heatmap processing을 위해)
        self.image4_crop = [self.save_crop_3d(self.image4_list[i], mode='fix') for i in indexes]
        self.image3_crop = [self.save_crop_3d(self.image3_list[i], mode='fix') for i in indexes]

        return x, [utils.to_categorical(y_label1, self.classes),
                   utils.to_categorical(y_label2, self.classes),
                   utils.to_categorical(y_label3, self.classes)]

    def preprocessing(self, image1, image2, image3, image4):
        # crop image
        image1_img = self.save_crop_3d(image1, mode='fix')
        image2_img = self.save_crop_3d(image2, mode='fix')
        image3_img = self.save_crop_3d(image3, mode='fix')
        image4_img = self.save_crop_3d(image4, mode='fix')

        # percentile normalize
        image1_per_norm = self.percentile_normalization(image1_img)
        image2_per_norm = self.percentile_normalization(image2_img)

        # intensity normalize
        image1_intnorm = self.intensity_normalization(image1_per_norm, image4_img, 1)
        image2_intnorm = self.intensity_normalization(image2_per_norm, image4_img, 1)

        # min_max normalize
        image1_norm = self.max_norm(image1_intnorm)
        image2_norm = self.max_norm(image2_intnorm)
        image3_norm = self.max_norm(image3_img)

        # concatenate
        concatenated_img = self.concat_nifti_files(image1_norm, image2_norm, image3_norm)
        concatenated_img = concatenated_img.get_fdata()
        return concatenated_img

    def percentile_normalization(self, nifti, percentile: int = 1):
        """percentile min-max normalization from mricron

        Args:
            nifti (np.ndarray): input array
            percentile (int, optional): Defaults to 1.

        Returns:
            np.array: normalized nifti
        """
        nifti_array = nifti.get_fdata()
        min_percentile = np.percentile(nifti_array, percentile)
        max_percentile = np.percentile(nifti_array, 100 - percentile)

        # limit maximum intensity of nifti by max_percentile
        nifti_array[nifti_array >= max_percentile] = max_percentile

        # limit minimum intensity of nifti by min_percentile
        nifti_array[nifti_array <= min_percentile] = min_percentile

        nifti = nib.Nifti1Image(nifti_array, nifti.affine, nifti.header)

        return nifti

    def intensity_normalization(self, img, image4, tmean=None):
        if tmean is None:
            ValueError("Please type the tmean.")

        # icv 아닌 부분 제거
        non_icv_img = self.remove_nonicv(img, image4)
        non_icv_img_data = non_icv_img.get_fdata()

        # 평균 계산
        img_mean = np.sum(non_icv_img_data) / np.count_nonzero(non_icv_img_data)

        # 평균과 tmean의 비율 계산
        ratio = tmean / img_mean

        # tmean : 평균과 tmean의 비율을 전체 복셀에 적용
        result = non_icv_img.get_fdata() * ratio

        result = nib.Nifti1Image(result, img.affine, img.header)
        return result

    def remove_nonicv(self, img, image4):
        img_data = img.get_fdata()
        image4_data = image4.get_fdata()

        # CSF 부분까지 1 나머지 0
        icv = np.where((image4_data == 1) | (image4_data == 2) | (image4_data == 3) |
                       (image4_data == 4) | (image4_data == 5) | (image4_data == 6), 1, image4_data)
        icv = np.where((icv != 1), 0, icv)

        # 영상과 icv를 곱해서 icv가 0인 부분을 제거
        result_nii = img_data * icv
        result_nii = nib.Nifti1Image(result_nii, img.affine, img.header)

        return result_nii

    def save_crop_3d(self, img_file, minimum=None, maximum=None, mode=None):
        global cropped_img_data

        img = nib.load(img_file)
        img_data = img.get_fdata()

        # crop
        # min, max fix
        if mode == 'fix':
            cropped_img_data = img_data[17:181, 4:22, 11:188]
        # save cropped image
        cropped_img_data = nib.Nifti1Image(cropped_img_data, img.affine, img.header)

        return cropped_img_data

    def concat_nifti_files(self, image1, image2, image3):
        image1_img = image1.get_fdata()
        image2_img = image2.get_fdata()
        image3_img = image3.get_fdata()
        img = np.stack((image3_img, image2_img, image1_img), axis=-1)
        concat_img = nib.Nifti1Image(img, image2.affine, image2.header)
        return concat_img

    def max_norm(self, img):
        save_max_value = []
        img_data = img.get_fdata()
        max_value = np.max(img_data)
        save_max_value.append(max_value)
        result = img_data / np.max(save_max_value)
        result = nib.Nifti1Image(result, img.affine, img.header)
        return result

    def crop_foreground_3d(self, img, edge_only=True):
        """ Crop 3-dimensional image into non-zero foreground
       Args:
           img (np.array):  3-dimensional array
           edge_only (bool, optional): only returns top and bottom edges
       Returns:
           img (np.array): cropped img
           edges (tuple(np.array, np.array)): top and bottom edge of img
       """
        true_points = np.argwhere(img)
        top_edge = true_points.min(axis=0)
        bottom_edge = true_points.max(axis=0)
        if edge_only:
            return top_edge, bottom_edge
        # coordinate return
        return img[top_edge[0]:bottom_edge[0],
               top_edge[1]:bottom_edge[1],
               top_edge[2]:bottom_edge[2]]


data_path = 'D:/workspace'
test_image1_list = glob(join(data_path, 'test/*/store/nifti/image1_mni6.nii.gz'))
test_image2_list = glob(join(data_path, 'test/*/store/nifti/image2_mni6.nii.gz'))
test_image3_list = glob(join(data_path, 'test/*/store/nifti/image3_lesion_divided.nii.gz'))
test_image4_list = glob(join(data_path, 'test/*/store/nifti/image4_mni6.nii.gz'))
test_label_data = join(data_path, 'visual_scoring_test.xlsx')

image_shape = (164, 18, 177)

seed = 33
batchsize = 1

test_generator = DataGenerator(image1_list=test_image1_list,
                               image2_list=test_image2_list,
                               image3_list=test_image3_list,
                               image4_list=test_image4_list,
                               label_data=test_label_data,
                               aug=None,
                               classes=4,
                               channels=3,
                               shuffle=False,
                               batchsize=batchsize
                               )


json_file = open("D:/workspace/visual_scoring_prediction/model/model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("D:/workspace/visual_scoring_prediction/weight/weight.h5")
loaded_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])

# prediction
prediction_label1 = np.argmax(loaded_model.predict_generator(test_generator)[0], axis=1)
prediction_label2 = np.argmax(loaded_model.predict_generator(test_generator)[1], axis=1)
prediction_label3 = np.argmax(loaded_model.predict_generator(test_generator)[2], axis=1)
softmax_label1 = loaded_model.predict_generator(test_generator)[0]
softmax_label2 = loaded_model.predict_generator(test_generator)[1]
softmax_label3 = loaded_model.predict_generator(test_generator)[2]
gt_label1 = [np.argmax(i[0]) for _, i in test_generator]
gt_label2 = [np.argmax(i[1]) for _, i in test_generator]
gt_label3 = [np.argmax(i[2]) for _, i in test_generator]

for image_data, pred, image4, smax_label1, smax_label2, smax_label3 in \
        zip(test_generator, prediction_label1, test_image4_list, softmax_label1, softmax_label2, softmax_label3):

    slices = []
    for i in range(18):
        slices.append(image_data[0][i][-1, :, :, :])

    # grad cam 처리
    grad_cams = [generate_grad_cam(slices, loaded_model, pred, 'CNN{}_bn3'.format(i + 1)) for i in range(18)]
    grad_cams = np.swapaxes(np.array(grad_cams), 0, 1)

    # grad cam 저장
    img_cam = nib.Nifti1Image(grad_cams, test_generator.image4_crop[0].affine, test_generator.image4_crop[0].header)

    # scale들의 softmax를 dict로 만들기
    dict_of_softmax = {"visual_scoring": {"visual_scoring_scale": {i: j for i, j in zip(range(4), smax_label1.tolist())},
                                   "periventricular_white_matter": {i: j for i, j in zip(range(4), smax_label2.tolist())},
                                   "deep_white_matter": {i: j for i, j in zip(range(4), smax_label3.tolist())}}}

    # 저장 경로 설정
    save_gradcam = image4.replace('\\', '/').split('/store/nifti/')[0] + '/store/nifti/visual_scoring_attention.nii.gz'
    save_softmax = image4.replace('\\', '/').split('/store/nifti/')[0] + '/store/result/visual_scoring.json'

    # softmax 저장
    with open(save_softmax, 'w', encoding='utf-8') as f:
        json.dump(dict_of_softmax, f, indent="\t")

    # gradcam 저장
    nib.save(img_cam, save_gradcam)

    # heatmap 저장
    heatmap_post_processing(save_gradcam, test_generator.image4_crop, test_generator.image3_crop)