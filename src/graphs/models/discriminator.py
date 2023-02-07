import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(config.feature_size, 256)  # 6*6 from image dimension
        self.bn = nn.BatchNorm1d(256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(256, 1)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.fc1(x)
        # If the size is a square you can only specify a single number
        x = self.leaky_relu(self.bn(x))

        x = self.fc2(x)

        return x