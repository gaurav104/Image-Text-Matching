import torch.nn as nn
from .bilstm import BiLSTM
from .bilstm import BiLSTM_Bert, BiLSTM_BertOnly
from .mobilenet import MobileNetV2
from .resnet50 import ResNet50
from .resnet101 import ResNet101
# from .resnet import resnet50


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.image_model == 'mobilenet_v2':
            self.image_model = MobileNetV2(config)
            # self.image_model.apply(self.image_model.weight_init)
        elif config.image_model == 'resnet50':
            self.image_model = ResNet50(config)
        elif config.image_model == 'resnet101':
            self.image_model = ResNet101(config)


        self.bilstm = BiLSTM(config)

        # self.bilstm.apply(self.bilstm.weight_init)

        if config.image_model == 'mobilenet_v2':
            inp_size = 1280
        
        if config.image_model == 'resnet50' or config.image_model == 'resnet101':
            inp_size = 2048
        # shorten the tensor using 1*1 conv
        
        # self.conv_text = nn.Conv2d(config.num_lstm_units, config.feature_size, 1)

        



    def forward(self, images, text, text_length):
        image_embeddings = self.image_model(images)
        text_embeddings = self.bilstm(text, text_length)
#         image_embeddings, text_embeddings= self.build_joint_embeddings(image_features, text_features)

#         print(image_embeddings.size())
        return image_embeddings, text_embeddings


#     def build_joint_embeddings(self, images_features, text_features):
        
#         #images_features = images_features.permute(0,2,3,1)
#         #text_features = text_features.permute(0,3,1,2)
        
        

# #         return image_embeddings, text_embeddings
class ModelBert(nn.Module):
    def __init__(self, config):
        super(ModelBert, self).__init__()
        if config.image_model == 'mobilenet_v2':
            self.image_model = MobileNetV2(config)
            # self.image_model.apply(self.image_model.weight_init)
        elif config.image_model == 'resnet50':
            self.image_model = ResNet50(config)
        elif config.image_model == 'resnet101':
            self.image_model = ResNet101(config)

        self.bilstm = BiLSTM_Bert(config)
        # self.bilstm.apply(self.bilstm.weight_init)

        if config.image_model == 'mobilenet_v2':
            inp_size = 1280
        
        if config.image_model == 'resnet50' or config.image_model == 'resnet101':
            inp_size = 2048
        # shorten the tensor using 1*1 conv
        
        # self.conv_text = nn.Conv2d(config.num_lstm_units, config.feature_size, 1)

        



    def forward(self, images, text, text_length, attention_mask):
        image_embeddings = self.image_model(images)
        text_embeddings = self.bilstm(text, text_length, attention_mask)
#         image_embeddings, text_embeddings= self.build_joint_embeddings(image_features, text_features)

#         print(image_embeddings.size())
        return image_embeddings, text_embeddings

class ModelBertOnly(nn.Module):
    def __init__(self, config):
        super(ModelBertOnly, self).__init__()
        if config.image_model == 'mobilenet_v2':
            self.image_model = MobileNetV2(config)
            # self.image_model.apply(self.image_model.weight_init)
        elif config.image_model == 'resnet50':
            self.image_model = ResNet50(config)
        elif config.image_model == 'resnet101':
            self.image_model = ResNet101(config)

        self.bilstm = BiLSTM_BertOnly(config)
        # self.bilstm.apply(self.bilstm.weight_init)

        if config.image_model == 'mobilenet_v2':
            inp_size = 1280
        
        if config.image_model == 'resnet50' or config.image_model == 'resnet101':
            inp_size = 2048
        # shorten the tensor using 1*1 conv
        
        # self.conv_text = nn.Conv2d(config.num_lstm_units, config.feature_size, 1)

    def forward(self, images, text, text_length, attention_mask):
        image_embeddings = self.image_model(images)
        text_embeddings = self.bilstm(text, text_length, attention_mask)
#         image_embeddings, text_embeddings= self.build_joint_embeddings(image_features, text_features)

#         print(image_embeddings.size())
        return image_embeddings, text_embeddings
