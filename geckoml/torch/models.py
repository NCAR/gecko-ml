from typing import List
from torch import nn
import logging
import torch


logger = logging.getLogger(__name__)


class DenseNet(nn.Module):

    def __init__(self,
                 hidden_dims: List[int] = [100, 50], 
                 dropouts: List[float] = [0.2, 0.2],
                 batch_norm: bool = False,
                 verbose: bool = True,):
        
        super(DenseNet, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.dropouts = dropouts
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.model = None
        
    def build(self, 
              input_size: int, 
              output_size: int):
        
        if self.verbose:
            logger.info(
                f"Building a DenseNet having input size {input_size}, output size {output_size},"
            )
            logger.info(
                f"layer sizes {self.hidden_dims}, and dropouts {self.dropouts}"
            )
        
        self.model_list = []
        self.model_list.append(nn.Linear(input_size, self.hidden_dims[0]))
        if self.batch_norm:
            self.model_list.append(nn.BatchNorm1d(num_features=self.hidden_dims[0]))
        self.model_list.append(nn.LeakyReLU())
        if len(self.hidden_dims) > 1:
            if self.dropouts[0] > 0.0:
                self.model_list.append(nn.Dropout(self.dropouts[0]))
            for i in range(len(self.hidden_dims)-1):
                self.model_list.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
                if self.batch_norm:
                    self.model_list.append(nn.BatchNorm1d(num_features=self.hidden_dims[i+1]))
                self.model_list.append(nn.LeakyReLU())
                if self.dropouts[i+1] > 0.0:
                    self.model_list.append(nn.Dropout(self.dropouts[i+1]))
        self.model_list.append(nn.Linear(self.hidden_dims[-1], output_size))
        self.model_list.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.model_list)

    def forward(self, 
                x: torch.FloatTensor):
        
        if self.model is None:
            raise OSError(f"You must call DenseNet.build before using the model. Exiting.")
        
        x = self.model(x)
        return x