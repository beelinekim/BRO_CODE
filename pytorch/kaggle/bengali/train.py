import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as torchtransforms
import cv2
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm
import torchvision
import torch.nn.functional as F
import time
import torch
from glob import glob
import utils
from posixpath import join
import loader
import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = 'D:/workspace/kaggle/bengali/bengaliai-cv19'
TRAIN_PATHS = glob(join(DATA_DIR, 'train_image_data*'))
TRAIN_CSV = glob(join(DATA_DIR, 'train.csv'))

TRAIN_AUG = torchtransforms.Compose([
    torchtransforms.ToTensor(),
    torchtransforms.Normalize([0.485, 0.456, 0.406], [0,229, 0.224, 0.225])
])

HEIGHT = 137
WIDTH = 236
SIZE = 128
BATCH_SIZE = 64
TRAIN_SHUFFLE = True

train_dataset = loader.BanDataset(TRAIN_PATHS, TRAIN_CSV, HEIGHT, WIDTH, SIZE)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=TRAIN_SHUFFLE,
                          num_workers=2)

model = model.ResNet34().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
criterion = nn.CrossEntropyLoss()
epochs = 50
model.train()
losses = []
accs = []

def train():
    # training loop

    running_loss = 0.0
    running_acc = 0.0

    for i, data in enumerate(train_loader):
        # input 얻기
        x, (label1, label2, label3) = data
        inputs = x.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        label3 = label3.to(device)

        optimizer.zero_grad()

        out1, out2, out3 = model(inputs.unsqueeze(1).float())

        loss1 = criterion(out1, label1)
        loss2 = criterion(out2, label2)
        loss3 = criterion(out3, label3)

        running_loss += loss1 + loss2 + loss3
        running_acc += (out1.argmax(1)==label1).float().mean()
        running_acc += (out2.argmax(1) == label2).float().mean()
        running_acc += (out3.argmax(1) == label3).float().mean()

        (loss1 + loss2 + loss3).backward()
        optimizer.step()

    losses.append(running_loss / len(train_loader))
    accs.append(running_acc / len(train_loader) * 3)
    print('acc : {:.2f}%'.format(running_acc / (len(train_loader) * 3)))
    print('loss : {:.4f}'.format(running_loss / len(train_loader)))

def main(epochs):
    for epoch in range(epochs):
        print('epochs {}/{} '.format(epoch + 1, epochs))
        train()
    torch.save(model.state_dict(), 'resnet34_50epochs_saved_weights.pth')

if __name__ == '__main__':
    main(epochs)




