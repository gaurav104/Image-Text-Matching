import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import subprocess
from transformers import BertTokenizer, BertModel


"""
Neural Networks model : Bidirection LSTM
"""


class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()

        self.hidden_dim = config.hidden_dim_size

        # V = config.vocab_size
        # D = config.text_embedding

        # word embedding
        


        if config.run_on_cluster:
            output_bytes = subprocess.check_output("echo $SLURM_TMPDIR", shell=True)
            output_string = output_bytes.decode('utf-8').strip()
            # image_dir = os.path.join(output_string, '{}/images'.format(config.dataset))
            anno_dir = os.path.join(output_string, '{}/processed_data'.format(config.dataset))
            directory = os.path.join(anno_dir, 'weights_matrix_300_idx.pkl')
    #             config.project_directory + 'pretrained_models/glove_embeddings/weights_matrix_{}_idx.pkl'.format(D)
        else:
            directory = '../Datasets/{}/processed_data/weights_matrix_300_idx.pkl'.format(config.dataset)

        weights_matrix = pickle.load(open(directory, 'rb'))
        # self.logger.info("Embedding Vocavulary Loaded")

        V, _= weights_matrix.shape
        D = config.text_embedding
        self.embed = nn.Embedding(V, D, padding_idx=0)
        
        if config.pretrained_embeddings:
            self.embed.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
        
        if config.embed_non_trainable:
            self.embed.weight.requires_grad = False
            
        if config.bidirectional:
            self.conv_text = nn.Conv2d(config.hidden_dim_size*2, config.feature_size, 1, bias=False)
        else:
            self.conv_text = nn.Conv2d(config.hidden_dim_size, config.feature_size, 1, bias=False)

#         weights_matrix_300_idx.pkl
        
        # self.bilstm = nn.ModuleList()
        self.bidirectional = config.bidirectional
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=1, dropout=0, bidirectional=self.bidirectional, batch_first=True, bias=True)
        
        # self.bidirectional = config.bidirectional
        # if self.bidirectional:
        #     self.bilstm = nn.LSTM(D, config.num_lstm_units, num_layers=1, dropout=0, bidirectional=True, bias=False)

    def forward(self, text, text_length):
        

        embed = self.embed(text) # [B, seq_length, embedding_size]

        B, S, E = embed.size()

        seq_padded = nn.utils.rnn.pack_padded_sequence(embed, text_length, batch_first=True, enforce_sorted=False)

        # unidirectional lstm

        out, _ = self.bilstm(seq_padded)

        output_unpacked , length = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)


        bilstm_out, _ = torch.max(output_unpacked, dim = 1, keepdim=False)#/text_length.view(B, 1)

        # bilstm_out = self.bilstm_out(embed, text_length, 0)
        
        # if self.bidirectional:
        #     index_reverse = list(range(embed.shape[0]-1, -1, -1))
        #     index_reverse = torch.LongTensor(index_reverse).cuda()
        #     embed_reverse = embed.index_select(0, index_reverse)
        #     text_length_reverse = text_length.index_select(0, index_reverse)
        #     bilstm_out_bidirection = self.bilstm_out(embed_reverse, text_length_reverse, 1)
        #     bilstm_out_bidirection_reverse = bilstm_out_bidirection.index_select(0, index_reverse)
        #     bilstm_out = torch.cat([bilstm_out, bilstm_out_bidirection_reverse], dim=2)
        # bilstm_out, _ = torch.max(bilstm_out, dim=1)
        # bilstm_out = bilstm_out.unsqueeze(2).unsqueeze(2)
        
        text_embeddings = self.conv_text(bilstm_out.unsqueeze(-1).unsqueeze(-1)).squeeze()
        return text_embeddings

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data, 1)
            nn.init.constant(m.bias.data, 0)

class BiLSTMHyp(nn.Module):
    def __init__(self, config, normalize=None, K=0.1):
        super(BiLSTMHyp, self).__init__()

        self.hidden_dim = config.hidden_dim_size

        # V = config.vocab_size
        # D = config.text_embedding

        # word embedding
        self.normalize = normalize
        self.K = K
        self.inner_radius = 2 * self.K / (1 + np.sqrt(1 + 4 * self.K * self.K))
        self.epsilon = 1e-5
        self.inner_radius_h = self.arctanh(torch.tensor(self.inner_radius))
        self.output_dim = config.feature_size


        if config.run_on_cluster:
            output_bytes = subprocess.check_output("echo $SLURM_TMPDIR", shell=True)
            output_string = output_bytes.decode('utf-8').strip()
            # image_dir = os.path.join(output_string, '{}/images'.format(config.dataset))
            anno_dir = os.path.join(output_string, '{}/processed_data'.format(config.dataset))
            directory = os.path.join(anno_dir, 'weights_matrix_300_idx.pkl')
    #             config.project_directory + 'pretrained_models/glove_embeddings/weights_matrix_{}_idx.pkl'.format(D)
        else:
            directory = '../Datasets/{}/processed_data/weights_matrix_300_idx.pkl'.format(config.dataset)

        weights_matrix = pickle.load(open(directory, 'rb'))
        # self.logger.info("Embedding Vocavulary Loaded")

        V, _= weights_matrix.shape
        D = config.text_embedding
        self.embed = nn.Embedding(V, D, padding_idx=0)
        
        if config.pretrained_embeddings:
            self.embed.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
        
        if config.embed_non_trainable:
            self.embed.weight.requires_grad = False
            
        if config.bidirectional:
            self.conv_text = nn.Conv2d(config.hidden_dim_size*2, config.feature_size, 1, bias=False)
        else:
            self.conv_text = nn.Conv2d(config.hidden_dim_size, config.feature_size, 1, bias=False)

#         weights_matrix_300_idx.pkl
        
        # self.bilstm = nn.ModuleList()
        self.bidirectional = config.bidirectional
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=1, dropout=0, bidirectional=self.bidirectional, batch_first=True, bias=True)
        
        # self.bidirectional = config.bidirectional
        # if self.bidirectional:
        #     self.bilstm = nn.LSTM(D, config.num_lstm_units, num_layers=1, dropout=0, bidirectional=True, bias=False)

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


    def forward(self, text, text_length):
        

        embed = self.embed(text) # [B, seq_length, embedding_size]

        B, S, E = embed.size()

        seq_padded = nn.utils.rnn.pack_padded_sequence(embed, text_length, batch_first=True, enforce_sorted=False)

        # unidirectional lstm

        out, _ = self.bilstm(seq_padded)

        output_unpacked , length = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)


        bilstm_out, _ = torch.max(output_unpacked, dim = 1, keepdim=False)#/text_length.view(B, 1)

        # bilstm_out = self.bilstm_out(embed, text_length, 0)
        
        # if self.bidirectional:
        #     index_reverse = list(range(embed.shape[0]-1, -1, -1))
        #     index_reverse = torch.LongTensor(index_reverse).cuda()
        #     embed_reverse = embed.index_select(0, index_reverse)
        #     text_length_reverse = text_length.index_select(0, index_reverse)
        #     bilstm_out_bidirection = self.bilstm_out(embed_reverse, text_length_reverse, 1)
        #     bilstm_out_bidirection_reverse = bilstm_out_bidirection.index_select(0, index_reverse)
        #     bilstm_out = torch.cat([bilstm_out, bilstm_out_bidirection_reverse], dim=2)
        # bilstm_out, _ = torch.max(bilstm_out, dim=1)
        # bilstm_out = bilstm_out.unsqueeze(2).unsqueeze(2)
        
        x = self.conv_text(bilstm_out.unsqueeze(-1).unsqueeze(-1)).squeeze()

        x = x + 1e-15

        original_shape = x.shape
        x = x.view(-1, original_shape[-1])

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = x_norm.repeat(1, self.output_dim)
        # make sure the argument to tanh is less than 15.0
        # x[x_norm > 15.0] = x[x_norm > 15.0]/x_norm[x_norm > 15.0]*15.0
        # x_norm[x_norm > 15.0] = 15.0

        # perform exp0(x)
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




class BiLSTM_Bert(nn.Module):
    def __init__(self, config):
        super(BiLSTM_Bert, self).__init__()

        self.hidden_dim = config.hidden_dim_size
        self.bidirectional = config.bidirectional

        # V = config.vocab_size
        D = config.text_embedding

        # word embedding
        # self.embed = nn.Embedding(V, D, padding_idx=0)
        
    #     if config.pretrained_embeddings:
    #         if config.run_on_cluster:
    #             directory = os.path.join(config.project_directory, 'pretrained_weights/glove_embeddings/weights_matrix_{}_idx_min2.pkl'.format(D))
    # #             config.project_directory + 'pretrained_models/glove_embeddings/weights_matrix_{}_idx.pkl'.format(D)
    #         else:
    #             directory = '../pretrained_weights/glove_embeddings/weights_matrix_{}_idx_min2.pkl'.format(D)

    #         weights_matrix = pickle.load(open(directory, 'rb'))
        
    #         self.embed.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
        
        

        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        for param in self.bert_encoder.parameters():
            param.requires_grad = False



        if self.bidirectional:
            self.conv_text = nn.Conv2d(config.hidden_dim_size*2, config.feature_size, 1, bias=False)
        else:
            self.conv_text = nn.Conv2d(config.hidden_dim_size, config.feature_size, 1, bias=False)

#         weights_matrix_300_idx.pkl
        
        # self.bilstm = nn.ModuleList()
        
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=1, bidirectional=self.bidirectional, batch_first=True, bias=True)

        
        # self.bidirectional = config.bidirectional
        # if self.bidirectional:
        #     self.bilstm = nn.LSTM(D, config.num_lstm_units, num_layers=1, dropout=0, bidirectional=True, bias=False)


    def forward(self, text, text_length, attention_mask):
        

        outputs = self.bert_encoder(text, attention_mask) # [B, ['CLS']+seq_length+['SEP'], embedding_size]
        embed = outputs[2][-1]

        B, S, E = embed.size()

        seq_padded = nn.utils.rnn.pack_padded_sequence(embed[:,1:], text_length-2, batch_first=True, enforce_sorted=False)

        # unidirectional lstm

        out, _ = self.bilstm(seq_padded)

        output_unpacked , _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)


        bilstm_out, _ = torch.max(output_unpacked, dim = 1, keepdim=False)#/text_length.view(B, 1)

        # bilstm_out = self.bilstm_out(embed, text_length, 0)
        
        # if self.bidirectional:
        #     index_reverse = list(range(embed.shape[0]-1, -1, -1))
        #     index_reverse = torch.LongTensor(index_reverse).cuda()
        #     embed_reverse = embed.index_select(0, index_reverse)
        #     text_length_reverse = text_length.index_select(0, index_reverse)
        #     bilstm_out_bidirection = self.bilstm_out(embed_reverse, text_length_reverse, 1)
        #     bilstm_out_bidirection_reverse = bilstm_out_bidirection.index_select(0, index_reverse)
        #     bilstm_out = torch.cat([bilstm_out, bilstm_out_bidirection_reverse], dim=2)
        # bilstm_out, _ = torch.max(bilstm_out, dim=1)
        # bilstm_out = bilstm_out.unsqueeze(2).unsqueeze(2)
        
        text_embeddings = self.conv_text(bilstm_out.unsqueeze(-1).unsqueeze(-1)).squeeze()
        return text_embeddings

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data, 1)
            nn.init.constant(m.bias.data, 0)

class BiLSTM_BertOnly(nn.Module):
    def __init__(self, config):
        super(BiLSTM_BertOnly, self).__init__()

        self.hidden_dim = config.hidden_dim_size
        self.bidirectional = config.bidirectional

        # V = config.vocab_size
        D = config.text_embedding

        # word embedding
        # self.embed = nn.Embedding(V, D, padding_idx=0)
        
    #     if config.pretrained_embeddings:
    #         if config.run_on_cluster:
    #             directory = os.path.join(config.project_directory, 'pretrained_weights/glove_embeddings/weights_matrix_{}_idx_min2.pkl'.format(D))
    # #             config.project_directory + 'pretrained_models/glove_embeddings/weights_matrix_{}_idx.pkl'.format(D)
    #         else:
    #             directory = '../pretrained_weights/glove_embeddings/weights_matrix_{}_idx_min2.pkl'.format(D)

    #         weights_matrix = pickle.load(open(directory, 'rb'))
        
    #         self.embed.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
        
        

        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        if config.embed_non_trainable:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.bert_encoder.parameters():
                param.requires_grad = True



        if self.bidirectional:
            self.conv_text = nn.Conv2d(768, config.feature_size, 1)
        else:
            self.conv_text = nn.Conv2d(768, config.feature_size, 1)

#         weights_matrix_300_idx.pkl
        
        # self.bilstm = nn.ModuleList()
        
        # self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=1, bidirectional=self.bidirectional, batch_first=True, bias=True)

        
        # self.bidirectional = config.bidirectional
        # if self.bidirectional:
        #     self.bilstm = nn.LSTM(D, config.num_lstm_units, num_layers=1, dropout=0, bidirectional=True, bias=False)


    def forward(self, text, text_length, attention_mask):
        

        outputs = self.bert_encoder(text, attention_mask) # [B, ['CLS']+seq_length+['SEP'], embedding_size]
        embed = outputs[2][-1]

        B, S, E = embed.size()

        
        bilstm_out = embed[:,0,:] #embedding corresponding to CLS token

        # bilstm_out = self.bilstm_out(embed, text_length, 0)
        
        # if self.bidirectional:
        #     index_reverse = list(range(embed.shape[0]-1, -1, -1))
        #     index_reverse = torch.LongTensor(index_reverse).cuda()
        #     embed_reverse = embed.index_select(0, index_reverse)
        #     text_length_reverse = text_length.index_select(0, index_reverse)
        #     bilstm_out_bidirection = self.bilstm_out(embed_reverse, text_length_reverse, 1)
        #     bilstm_out_bidirection_reverse = bilstm_out_bidirection.index_select(0, index_reverse)
        #     bilstm_out = torch.cat([bilstm_out, bilstm_out_bidirection_reverse], dim=2)
        # bilstm_out, _ = torch.max(bilstm_out, dim=1)
        # bilstm_out = bilstm_out.unsqueeze(2).unsqueeze(2)
        
        text_embeddings = self.conv_text(bilstm_out.unsqueeze(-1).unsqueeze(-1)).squeeze()
        return text_embeddings



    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data, 1)
            nn.init.constant(m.bias.data, 0)