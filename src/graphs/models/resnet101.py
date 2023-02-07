"""
ResNet101
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from graphs.weights_initializer import init_model_weights

# from ..weights_initializer import weights_init


class ResNet101(nn.Module):
    def __init__(self, config):
        super(ResNet101,self).__init__()


        if config.run_on_cluster:

            pretrained_weights_path = os.path.join(config.project_directory, 'pretrained_weights')
            PATH = os.path.join(pretrained_weights_path, 'resnet101-5d3b4d8f.pth')
        else:
            PATH = '../pretrained_weights/resnet101-5d3b4d8f.pth'
        
        resnet101_model = models.resnet101(pretrained=False)
        resnet101_model.load_state_dict(torch.load(PATH))
        # self.input = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv_layers = nn.Sequential(*list(resnet101_model.children())[:8])
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        # self.classifier = nn.Conv2d(in_channels=2048, out_channels=out_channels, kernel_size=(1,1), bias=False)
        self.conv_images = nn.Conv2d(2048, config.feature_size, 1, bias=False)

        # self.apply(init_model_weights)


    def forward(self,x):
        # N, C, H, W = x.size()
        # x = self.input(x)     
        x = self.conv_layers(x)
        out = self.gap(x)
        # cam = F.interpolate(cam, size = (H,W), mode='bilinear', align_corners=True)
        # logits = self.gap(cam)
        image_embeddings = self.conv_images(out).squeeze()
       
        
        return image_embeddings



class ResNet101Hyp(nn.Module):
    def __init__(self, config, normalize=None, K=0.1):
        super(ResNet101Hyp,self).__init__()

        self.normalize = normalize
        self.K = K
        self.inner_radius = 2 * self.K / (1 + np.sqrt(1 + 4 * self.K * self.K))
        self.epsilon = 1e-5
        self.inner_radius_h = self.arctanh(torch.tensor(self.inner_radius))
        self.output_dim = config.feature_size

        if config.run_on_cluster:

            pretrained_weights_path = os.path.join(config.project_directory, 'pretrained_weights')
            PATH = os.path.join(pretrained_weights_path, 'resnet101-5d3b4d8f.pth')
        else:
            PATH = '../pretrained_weights/resnet101-5d3b4d8f.pth'
        
        resnet101_model = models.resnet101(pretrained=False)
        resnet101_model.load_state_dict(torch.load(PATH))
        # self.input = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv_layers = nn.Sequential(*list(resnet101_model.children())[:8])
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        # self.classifier = nn.Conv2d(in_channels=2048, out_channels=out_channels, kernel_size=(1,1), bias=False)
        self.conv_images = nn.Conv2d(2048, config.feature_size, 1, bias=False)

        # self.apply(init_model_weights)

    @staticmethod
    def arctanh(x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    def mob_add(self, u, v):
        v = v + 1e-15
        tf_dot_u_v = 2. * torch.sum(u*v, dim=1, keepdim=True)
        tf_norm_u_sq = torch.sum(u*u, dim=1, keepdim=True)
        tf_norm_v_sq = torch.sum(v*v, dim=1, keepdim=True)
        denominator = 1. + tf_dot_u_v + tf_norm_v_sq * tf_norm_u_sq
        tf_dot_u_v = tf_dot_u_v.repeat(1, self.embedding_dim)
        tf_norm_u_sq = tf_norm_u_sq.repeat(1, self.output_dim)
        tf_norm_v_sq = tf_norm_v_sq.repeat(1, self.output_dim)
        denominator = denominator.repeat(1, self.output_dim)
        result = (1. + tf_dot_u_v + tf_norm_v_sq) / denominator * u + (1. - tf_norm_u_sq) / denominator * v
        return self.soft_clip(result)

    def lambda_x(self, x):
        return 2. / (1 - torch.norm(x, p=2, dim=1, keepdim=True).repeat(1, self.output_dim))

    def exp_map_x(self, x, v):
        v = v + 1e-15
        norm_v = torch.norm(v, p=2, dim=1, keepdim=True).repeat(1, self.output_dim)
        second_term = torch.tanh(self.lambda_x(x) * norm_v / 2) * v/norm_v
        return self.mob_add(x, second_term)


    def soft_clip(self, x):
        original_shape = x.shape
        x = x.view(-1, original_shape[-1])
        # direction = F.normalize(x, dim=1)
        # norm = torch.norm(x, dim=1, keepdim=True)
        # x = direction * (norm + self.inner_radius)

        with torch.no_grad():
            norm = torch.norm(x, dim=1, keepdim=True).repeat(1, self.output_dim)
            x[norm <= self.inner_radius] = (1e-6+x[norm <= self.inner_radius]) / (1e-6+norm[norm <= self.inner_radius]) * self.inner_radius
            x[norm >= 1.0] = x[norm >= 1.0]/norm[norm >= 1.0]*(1.0-self.epsilon)
        return x.view(original_shape)


    def forward(self,x):
        # N, C, H, W = x.size()
        # x = self.input(x)     
        x = self.conv_layers(x)
        out = self.gap(x)
        # cam = F.interpolate(cam, size = (H,W), mode='bilinear', align_corners=True)
        # logits = self.gap(cam)
        x = self.conv_images(out).squeeze()

        x = x + 1e-15

        original_shape = x.shape
        x = x.view(-1, original_shape[-1])

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = x_norm.repeat(1, self.output_dim)

        x = torch.tanh(torch.clamp(self.inner_radius_h + x_norm, min=-15.0, max=15.0))*F.normalize(x)

        x = x.view(original_shape)

        if self.normalize == 'unit_norm':
            original_shape = x.shape
            x = x.view(-1, original_shape[-1])
            x = F.normalize(x, p=2, dim=1)
            x = x.view(original_shape)
        elif self.normalize == 'max_norm':
            original_shape = x.shape
            x = x.view(-1, original_shape[-1])
            norm_x = torch.norm(x, p=2, dim=1)
            x[norm_x > 1.0] = F.normalize(x[norm_x > 1.0], p=2, dim=1)
            x = x.view(original_shape)
        else:
            if self.K:
                return self.soft_clip(x)
            else:
                return x
        return x

       
        
        # return image_embeddings