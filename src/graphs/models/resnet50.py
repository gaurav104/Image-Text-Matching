"""
ResNet50
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from graphs.weights_initializer import init_model_weights

# from ..weights_initializer import weights_init


class ResNet50(nn.Module):
    def __init__(self, config):
        super(ResNet50,self).__init__()

        if config.run_on_cluster:

            pretrained_weights_path = os.path.join(config.project_directory, 'pretrained_weights')
            PATH = os.path.join(pretrained_weights_path, 'resnet50-19c8e357.pth')
        else:
            PATH = './pretrained_weights/resnet50-19c8e357.pth'
        
        resnet50_model = models.resnet50(pretrained=False)
        resnet50_model.load_state_dict(torch.load(PATH))
        # self.input = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv_layers = nn.Sequential(*list(resnet50_model.children())[:8])
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        
        self.conv_images = nn.Conv2d(2048, config.feature_size, 1, bias=False)
        


    def forward(self,x):
        # N, C, H, W = x.size()
  
        x = self.conv_layers(x)
        out = self.gap(x)
        image_embeddings = self.conv_images(out).squeeze()

        
        return image_embeddings