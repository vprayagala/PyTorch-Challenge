# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:17:21 2018

@author: vprayagala2
"""
#%%
from torch import nn
import torch.nn.functional as F
#%%
#Build own classifier network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # convolutional layer (sees 244x244x3 image tensor)
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        # convolutional layer (sees 122x122x6 tensor)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        # convolutional layer (sees 61x61x12 tensor)
        self.conv3 = nn.Conv2d(12, 24, 3, padding=2)
        # convolutional layer (sees 30x30x24 tensor)
        self.conv4 = nn.Conv2d(24, 48, 3, padding=2)
        # convolutional layer (sees 10x10x48 tensor)
        self.conv5 = nn.Conv2d(48, 96 , 3, padding=1)
                
        # max pooling layer
        self.pool1 = nn.MaxPool2d(2, 2)
        # max pooling layer
        self.pool2 = nn.MaxPool2d(3, 3)
        # linear layer (128 * 10 * 10 -> 500)
        self.fc1 = nn.Linear(96 * 10 * 10, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 102)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))
        x = self.pool1(F.relu(self.conv4(x)))
        x = self.poo2(F.relu(self.conv5(x)))
        # flatten image input
        x = x.view(-1, 96 * 10 * 10)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x