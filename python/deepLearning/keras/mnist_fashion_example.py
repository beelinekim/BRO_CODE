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

# �Ű�� �𵨿� �ֱ� �� ���� ������ 0~1 ���̷� ����
train_images = train_images / 255.0
test_images = test_images / 255.0

# ó�� 25�� �̹����� Ŭ���� �̸� ���
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# �� ����
# ù ��° �� : Flatten > 2������ �̹����� 1���� �迭�� ��ȯ
# �ȼ��� ��ģ �� �� ���� dense���� �����ؼ� �����.
# ù ��° dense���� 128���� ��带 ������, ������ ���� 10�� ����� softmax��
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# loss function : �Ʒ��ϴ� ���� ���� ������ ����, �ùٸ� �������� �н��ϵ��� �� �Լ��� �ּ�ȭ�ؾ���.
# optimizer : �����Ϳ� loss function�� �������� ���� ������Ʈ ����� ����
# metrics : ��Ȯ���� ���
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# �� �Ʒ�
model.fit(train_images, train_labels, epochs=5)