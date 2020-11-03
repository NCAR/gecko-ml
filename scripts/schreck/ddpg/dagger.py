#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import traceback
import random
import pickle
import torch
import yaml
import copy
import time 
import os

from ddpg.reader import LoadGeckoPandasExperiment
from ddpg.replay_buffer import ReplayBuffer
from ddpg.trainer import Trainer
#from ddpg.checkpoint import *
from ddpg.losses import GeckoLoss
from ddpg.ddpg import DDPG

from holodecml.vae.optimizers import *
from holodecml.vae.tqdm import tqdm
from holodecml.vae.checkpointer import *

from torch.optim.lr_scheduler import *
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt

from torch.autograd import Variable


# ### Load the configuration file

# In[2]:


with open("results/dagger/config.yml") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)


# ### Load the data splits

# In[3]:


experiment_data = LoadGeckoPandasExperiment(**config["data"])


# In[4]:


fn = "../clustered/experiment_data/experiment_train_test_val_splits.pkl"
with open(fn, "rb") as fid:
    train, valid, test = pickle.load(fid)


# In[5]:


train_data_set = LoadGeckoPandasExperiment(
    **config["data"],
    experiment_subset = train,
    x_data = experiment_data.x,
    y_data = experiment_data.y
)


# In[6]:


config["data"]["shuffle"] = False


# In[7]:


valid_data_set = LoadGeckoPandasExperiment(
    **config["data"],
    experiment_subset = valid,
    x_data = experiment_data.x,
    y_data = experiment_data.y
)


# In[8]:


test_data_set = LoadGeckoPandasExperiment(
    **config["data"],
    experiment_subset = test,
    x_data = experiment_data.x,
    y_data = experiment_data.y
)


# ### Get GPU device

# In[26]:


is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")


device = "cpu"

if is_cuda:
    torch.backends.cudnn.benchmark = True
    
print(f'Preparing to use device {device}')


# ### Load the models

# In[27]:


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
            nn.LeakyReLU() # nn.SELU()
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


# In[75]:


policy = DenseNet(**config["model"])
expert = DenseNet(**config["model"])


# In[76]:


fid = torch.load("results/pretrain/best.pt", map_location=lambda storage, loc: storage)

policy.load_state_dict(fid["model_state_dict"])
expert.load_state_dict(fid["model_state_dict"])

# critic_dict = torch.load("results/100/critic_best.pt", map_location=lambda storage, loc: storage)
# policy.critic.load_state_dict(critic_dict["model_state_dict"])
# policy.critic_target = copy.deepcopy(policy.critic)

# actor_dict = torch.load("results/100/actor_best.pt", map_location=lambda storage, loc: storage)
# policy.actor.load_state_dict(actor_dict["model_state_dict"])
# policy.actor_target = copy.deepcopy(policy.actor)


# ### Load the replay buffer

# In[77]:


class ReplayBuffer(object):
    
    def __init__(self, state_dim, action_dim, device, min_size=int(1e4), max_size=int(1e6)):
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0 

        self.input = np.zeros((self.max_size, state_dim))
        self.output = np.zeros((self.max_size, state_dim))    
        self.device = device

    def add(self, x, y):
        self.input[self.ptr] = x
        self.output[self.ptr] = y
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.input[ind]).to(self.device),
            torch.FloatTensor(self.output[ind]).to(self.device)
        )


# In[78]:


replay_buffer = ReplayBuffer(**config["replay_buffer"], device = device)


# In[79]:


optimizer = LookaheadDiffGrad(policy.parameters(),
                              lr=config["optimizer"]["lr"])
#                              weight_decay=1e-2)


# In[84]:


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
class Trainer:

    def __init__(self, 
                 policy, 
                 expert,
                 optimizer,
                 train_generator, 
                 validation_generator, 
                 test_generator, 
                 replay_buffer,
                 device,
                 start_epoch = 0,
                 epochs = 100,
                 batches_per_epoch = 1e10,
                 experiment_batch_size = 1,
                 batch_size = 64,
                 max_timestep = 1439,
                 teacher_force = False, 
                 gamma = 1.0,
                 clip = 1.0,
                 max_state = 1.0,
                 expl_noise = 0.1):
        
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.experiment_batch_size = experiment_batch_size
        
        self.train_gen = train_generator
        self.valid_gen = validation_generator
        self.test_gen = test_generator
        
        self.replay_buffer = replay_buffer
        self.max_timestep = max_timestep
        
        self.policy = policy
        self.expert = expert
        self.optimizer = optimizer
        
        self.teacher_force = teacher_force
        self.gamma = gamma
        
        self.device = device
        self.clip = clip
        self.max_state = max_state
        self.expl_noise = expl_noise 
        
        self.beta = np.linspace(1.0, 0.0, self.epochs, endpoint = True)
                
        #############################################
        #
        # Gradient clipping through hook registration
        #
        #############################################
        for p in self.policy.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))
        logger.info(f"Clipping gradients to range [-{clip}, {clip}]")
        
        #############################################
        #
        # Set expert policy to eval mode
        #
        #############################################
        self.expert.eval()
        
        #############################################
        #
        # Fix the number of batches per epoch
        #
        #############################################
        batches_per_epoch = int(self.train_gen.__len__()) 
        if self.batches_per_epoch > batches_per_epoch:
            self.batches_per_epoch = batches_per_epoch
        
        #############################################
        #
        # Load the training experiments and randomly shuffle
        #
        #############################################
        self.train_experiments = list(range(len(self.train_gen.experiment_subset)))
        self.train_experiments_copy = list(range(len(self.train_gen.experiment_subset)))
        random.shuffle(self.train_experiments)
        if self.experiment_batch_size > 1:
            self.train_experiments = list(chunks(
                self.train_experiments, 
                self.experiment_batch_size
            ))
        self.reshuffle = 0 # when == len(self.train_experiments), reshuffle
    
    def train_one_epoch(self, epoch):
        
        self.policy.train()
        
        if self.reshuffle == len(self.train_experiments):
            random.shuffle(self.train_experiments)
            if self.experiment_batch_size > 1:
                self.train_experiments = list(chunks(
                    self.train_experiments_copy, 
                    self.experiment_batch_size
                ))
            self.reshuffle = 0
        
        batch_group_generator = tqdm(
            enumerate(self.train_experiments[self.reshuffle:]),
            total=min(self.batches_per_epoch,len(self.train_experiments[self.reshuffle:])),
            leave=True
        )
        
        cost = self.tf_annealer(epoch)
        
        total_loss = []
        for batch_idx, exp in batch_group_generator:
            for i, (x,y,w) in enumerate(self.train_gen.__getitem__(exp)):
                x = x.to(self.device)
                y = y.to(self.device)
                w = w.to(self.device)

                if i == 0:
                    x_pred = x.clone()
                else:
                    x_pred[:, :29] = y_pred 
                    
                y_pred = self.policy(x_pred)
                y_expert = self.expert(x_pred if epoch > 0 else x) #assumes we start with expert as policy
                        
                for j in range(x.size(0)):
                    self.replay_buffer.add(
                        y_pred[j].cpu().detach().numpy(), 
                        y_expert[j].cpu().detach().numpy()
                    )
        
            # Train on memory buffer subset of D u D(i)
            if (self.replay_buffer.size >= self.replay_buffer.min_size):
                
                y_predict, y_expert = self.replay_buffer.sample(self.batch_size)
                
                # Compute DAgger loss
                loss = Variable(
                    F.mse_loss(y_predict, y_expert).type(torch.float32), requires_grad=True
                ).to(self.device)
                total_loss.append(loss.item())
                
                # Optimize the learner policy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                for curr_param, expert_param in zip(self.policy.parameters(), self.expert.parameters()):
                    curr_param.data.copy_(
                        self.beta[epoch] * expert_param.data + (1 - self.beta[epoch]) * curr_param.data
                    )
                                 
            # update tqdm
            ave_loss = np.mean(total_loss) if len(total_loss) > 0 else 0.0
            to_print = f"Epoch {epoch + 1} training loss: {ave_loss:.3f}"
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()
            
            self.reshuffle += 1
            if batch_idx > 0 and (batch_idx % self.batches_per_epoch) == 0:
                break
            
        return np.mean(total_loss)
    
        
    def test(self, epoch):
        self.policy.eval()
        with torch.no_grad():
            batch_group_generator = tqdm(
                enumerate(range(len(self.valid_gen.experiment_subset))),
                total=len(self.valid_gen.experiment_subset), 
                leave=True
            )
            total_loss = []
            for batch_idx, exp in batch_group_generator:
                for i, (x,y,w) in enumerate(self.valid_gen.__getitem__(exp)):
                    x = x.to(self.device)
                    y = y.to(self.device)
        
                    if i == 0:
                        x_pred = x.clone()
                    else:
                        x_pred[:, :29] = y_pred
                        
                    y_pred = self.policy(x_pred)
                    total_loss.append(F.mse_loss(y_pred, y).item())
                    
                # update tqdm
                ave_loss = np.mean(total_loss)
                to_print = f"Epoch {epoch + 1} validation loss: {ave_loss:.3f}"
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

        return np.mean(total_loss)
    
    def train(self,
              scheduler,
              early_stopping,
              metrics_logger):
            
        logger.info(
            f"Training the model for up to {self.epochs} epochs starting at epoch {self.start_epoch}"
        )

        flag = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        for episode in range(self.start_epoch, self.epochs):
            train_loss = self.train_one_epoch(episode)
            test_loss = self.test(episode)

            if scheduler is not None:
                scheduler.step(test_loss if flag else episode)
            
            early_stopping(
                episode, 
                test_loss, # we want to maximize the reward rather than min a loss
                self.policy, 
                self.optimizer
            )

            # Write results to the callback logger 
            result = {
                "episode": episode,
                "train_loss": train_loss,
                "valid_loss": test_loss,
                "lr_actor": early_stopping.print_learning_rate(self.optimizer),
                "forcing_score": self.tf_annealer(episode) if self.teacher_force else 1.0
            }
            metrics_logger.update(result)

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
                
    def tf_annealer(self, epoch):
        return 1.0 * self.gamma ** epoch


# ### Load the trainer class

# In[85]:


trainer = Trainer(
    policy, 
    expert,
    optimizer,
    train_data_set, 
    valid_data_set, 
    test_data_set, 
    replay_buffer,
    device = device,
    **config["trainer"]
)


# ### Load the callbacks

# In[86]:


# Initialize LR annealing scheduler 
if "ReduceLROnPlateau" in config["callbacks"]:
    schedule_config = config["callbacks"]["ReduceLROnPlateau"]
    scheduler = ReduceLROnPlateau(trainer.optimizer, **schedule_config)
    #logging.info(
    #    f"Loaded ReduceLROnPlateau learning rate annealer with patience {schedule_config['patience']}"
    #)
elif "ExponentialLR" in config["callbacks"]:
    schedule_config = config["callbacks"]["ExponentialLR"]
    scheduler = ExponentialLR(trainer.optimizer, **schedule_config)
    #logging.info(
    #    f"Loaded ExponentialLR learning rate annealer with reduce factor {schedule_config['gamma']}"
    #)

# Early stopping
checkpoint_config = config["callbacks"]["EarlyStopping"]
early_stopping = EarlyStopping(**checkpoint_config)

# Write metrics to csv each epoch
metrics_logger = MetricsLogger(**config["callbacks"]["MetricsLogger"])


# ### Train the models

# In[ ]:


trainer.train(scheduler, early_stopping, metrics_logger)


# In[ ]:




