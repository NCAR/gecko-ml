from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import *
import torch.nn.functional as F
from torch import nn

import logging
import torch


logger = logging.getLogger(__name__)


class DenseNet(nn.Module):

    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_dims = [100, 50], 
                 dropouts = [0.2, 0.2]):
        
        super(DenseNet, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        #self.embedding = nn.Embedding(train_data_set.num_timesteps, 16)
        
        self.model = [
            nn.Linear(input_size, hidden_dims[0]),
            #nn.BatchNorm1d(num_features=hidden_dims[0]),
            nn.LeakyReLU()
        ]
        if len(hidden_dims) > 1:
            if dropouts[0] > 0.0:
                self.model.append(nn.Dropout(dropouts[0]))
            for i in range(len(hidden_dims)-1):
                self.model.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                #self.model.append(nn.BatchNorm1d(num_features=hidden_dims[i+1]))
                self.model.append(nn.LeakyReLU())
                if dropouts[i+1] > 0.0:
                    self.model.append(nn.Dropout(dropouts[i+1]))
        self.model.append(nn.Linear(hidden_dims[-1], output_size))
        self.model.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        #x1, x2 = x
        #x1 = self.embedding(x1)
        #x = torch.cat([x1, x2], 1)
        x = self.model(x)
        return x