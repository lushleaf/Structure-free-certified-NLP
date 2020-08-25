# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):

    def __init__(self):
        self.model_name = 'TextCNN'
        self.dropout = 0.2                                            
        self.require_improvement = 1000                                 
        self.num_classes =  5                                           
        self.n_vocab = 30522                                                
        self.num_epochs = 20                                            
        self.batch_size = 128                                          
        self.pad_size = 32                                              
        self.learning_rate = 1e-3                                       
        self.embed = 300
        self.filter_sizes = (3,)                                   
        self.num_filters = 100                                          


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        config = Config()
        
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.fc1 = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_filters * len(config.filter_sizes))
        self.fc2 = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

        self.loss = torch.nn.CrossEntropyLoss().cuda()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = input_ids
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return [torch.nn.functional.cross_entropy(out, labels), out]
