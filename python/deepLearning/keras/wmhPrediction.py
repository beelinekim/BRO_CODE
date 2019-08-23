import numpy as np
import nibabel as nib
from posixpath import join
from tensorflow.keras import models, optimizers, Input, Sequential, Model, utils, activations
from tensorflow.keras.layers import Layer, Add, Activation, Softmax, Flatten, Dense, Conv3D, MaxPool3D, \
    BatchNormalization, GlobalAveragePooling3D, AveragePooling3D, concatenate, Lambda, MaxPooling3D, Dropout
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from glob import glob
import pandas as pd

class DataGenerator(Sequence):
    def __init__(self,
                 pathList=None,
                 labelImagePath=None,
                 inputFieldData=None,
                 labelFieldData=None,
                 batchSize=2,
                 dim=None,
                 channels=None,
                 classes=None,
                 shuffle=None,
                 ):
        super(DataGenerator, self).__init__()
        self.pathList = pathList,
        self.labelImagePath = labelImagePath,
        self.inputFieldData = inputFieldData,
        self.labelFieldData = labelFieldData,
        self.dim = dim,
        self.channels = channels,
        self.classes = classes
        self.batchSize = batchSize,
        self.shuffle = shuffle
        # self.inputImagePath = self.inputImagePath[0]
        self.pathList = self.pathList[0]
        self.labelImagePath = self.labelImagePath[0]
        self.inputFieldData = self.inputFieldData[0]
        self.labelFieldData = self.labelFieldData[0]
        self.dim = self.dim[0]
        self.channels = self.channels[0]
        self.batchSize = self.batchSize[0]

        self.on_epoch_end()

    # 각 epoch 는 배치 index 0 ~ 배치 총 크기만큼 될 수 있다.
    # 이 부분을 __len__으로 컨트롤한다.
    def __len__(self):
        dataLength = len(self.pathList) // self.batchSize
        return dataLength

    # batch 처리가 주어진 index에 따라 호출될 때 generator는 __getitem__을 호출한다.
    # batch size만큼 들어가는 것을 계산해서 리턴
    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batchSize : (idx + 1) * self.batchSize]

        # find input image list
        pathListTemp = [self.pathList['id'][k] for k in indexes]
        labelListTemp = [self.pathList['avg'][kk] for kk in indexes]

        # generate data
        X, Y = self.__data_generation(pathListTemp, labelListTemp)

        return X, Y

    # shuffle을 통해 각 batch마다 이상적인 데이터셋을 학습시키는 것을 방지
    def on_epoch_end(self):
        self.indexes = self.pathList.index.values
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # generation process에서 데이터의 배치를 생성
    def __data_generation(self, pathListTemp, labelListTemp):
        # initialization
        dim_x = self.dim[0]
        dim_y = self.dim[1]
        dim_z = self.dim[2]
        x = np.empty((self.batchSize, dim_x, dim_y, dim_z, self.channels))
        y = np.empty((self.batchSize,))

        # generate data
        for i, id, label in zip(range(len(pathListTemp)), pathListTemp, labelListTemp):
            # store sample
            x[i,] = nib.load(id).get_fdata()

            # store class
            y[i] = label

        y = y.astype(np.int32)
        return x, keras.utils.to_categorical(y, num_classes=self.classes)


dataPath = dataPath
flairImagePath = glob(flair.nii.gz)
t1ImagePath = glob(t1.nii.gz)
wmhImagePath = glob(wmh.nii.gz)
inputImagePath = glob(concat.nii.gz)
labelFieldData = pd.read_excel(label.xlsx).astype('str')
imageShape = (x, y, z)
seed = 33

# concat_nifti_files(flairImagePath, t1ImagePath, wmhImagePath)

trainData, valData = train_test_split(labelFieldData[['id', 'avg']],
                                      train_size=0.9,
                                      random_state=seed,
                                      stratify=labelFieldData['avg'])

trainGenerator = DataGenerator(pathList=trainData,
                               dim=imageShape,
                               classes=4,
                               channels=3,
                               shuffle=True
                               )

validGenerator = DataGenerator(pathList=valData,
                               dim=imageShape,
                               classes=4,
                               channels=3,
                               shuffle=True
                               )

inputImageShape = imageShape
imageInput = Input(shape=inputImageShape, name='input_image')

conv1 = Conv3D(16, 7, 2, 'same', activation='relu', name='image_conv1')(imageInput)
conv2 = Conv3D(32, 3, 2, 'same', activation='relu', name='image_conv2')(conv1)
conv3 = Conv3D(64, 3, 2, 'same', activation='relu', name='image_conv3')(conv2)
conv4 = Conv3D(32, 3, 2, 'same', activation='relu', name='image_conv4')(conv3)
conv5 = Conv3D(16, 3, 2, 'same', activation='relu', name='image_conv5')(conv4)

flatten = Flatten(name='flatten')(conv5)
dense1 = Dense(200, 'relu', name='dense1')(flatten)
dense2 = Dense(50, 'relu', name='dense2')(dense1)
dense3 = Dense(10, 'relu', name='dense3')(dense2)
dense4 = Dense(4, 'softmax', name='dense4')(dense3)

model = Model(inputs=[imageInput], outputs=[dense4])
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
model.summary()

learningRateReduction = ReduceLROnPlateau(monitor='val_loss',
                                          patience=3,
                                          verbose=1,
                                          factor=0.5,
                                          min_lr=0.001)
earlyStopping = EarlyStopping(monitor='val_acc', patience=3)
callbacks = [learningRateReduction, earlyStopping]
modelCkpt = ModelCheckpoint('./checkpoint.h5',
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True)

history = model.fit_generator(
    epochs=500,
    generator=trainGenerator,
    validation_data=validGenerator,
    steps_per_epoch=len(trainGenerator.pathList),
    validation_steps=len(validGenerator.pathList),
    verbose=1,
    callbacks=callbacks
)

model.save_weights('save_basemodel.h5', save_format='h5')