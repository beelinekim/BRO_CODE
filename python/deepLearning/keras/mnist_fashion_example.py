from __future__ import absolute_import, division, print_function, unicode_literals

# tensorflow, tf.keras import
import tensorflow as tf
from tensorflow import keras

# helper library import
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# data check
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# 신경망 모델에 넣기 전 값의 범위를 0~1 사이로 조정
train_images = train_images / 255.0
test_images = test_images / 255.0

# 처음 25개 이미지와 클래스 이름 출력
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# 모델 구성
# 첫 번째 층 : Flatten > 2차원의 이미지를 1차원 배열로 변환
# 픽셀을 펼친 후 두 개의 dense층이 연속해서 연결됨.
# 첫 번째 dense층은 128개의 노드를 가지고, 마지막 층은 10개 노드의 softmax층
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# loss function : 훈련하는 동안 모델의 오차를 측정, 올바른 방향으로 학습하도록 이 함수를 최소화해야함.
# optimizer : 데이터와 loss function을 바탕으로 모델의 업데이트 방법을 결정
# metrics : 정확도를 사용
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
model.fit(train_images, train_labels, epochs=5)