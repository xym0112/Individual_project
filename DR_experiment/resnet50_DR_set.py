from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import zipfile
from MNIST_experiements.attacks import fgsm_attack

import os
import csv

os.environ['KMP_DUPLICATE_LIB_OK']='True'

BATCH_SIZE = 128
USE_CUDA = True
NUM_EPOCHS = 10
TRAIN_SIZE = 5000
TEST_SIZE = 1000

print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
dtype = torch.float32 

# Train images
data_dir = '/vol/bitbucket/yx3017/train'
label_dir = 'Individual_project/Dataset/trainLabels.csv'


class DRDataset(Dataset):
    def __init__(self, label_dict, root_dir, transform=None):
        self.label_dict = label_dict
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return TRAIN_SIZE + TEST_SIZE

    def __getitem__(self, image_name):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(os.path.join(self.root_dir, image_name))
        label = label_dict[image_name]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

def label_loading(path):
    # Get train labels dictionary
    reader = csv.reader(open(path))
    label_dict = {}
    for k, v in reader:
        # Make the labels binary
        if v != '0':
            v = '1'
        label_dict[k] = v

    return label_dict

labels = label_loading(label_dir)
dr_data = DRDataset(label_dict=labels, root_dir=data_dir)
loader_train = DataLoader(dr_data[:5000], BATCH_SIZE, shuffle=True)
loader_test = DataLoader(dr_data[5000:6000], BATCH_SIZE, shuffle=False)

sample = dr_data['33708_right']
plt.imshow(sample)

# model = models.resnet50(pretrained=False, progress=True).to(device)