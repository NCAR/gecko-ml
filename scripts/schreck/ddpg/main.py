#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import pickle
import torch
import ddpg.ddpg as ddpg
import traceback
import yaml
import copy
import os

import time 

from ddpg.reader import LoadGeckoPandasExperiment
from ddpg.replay_buffer import ReplayBuffer
from ddpg.trainer import Trainer
from ddpg.checkpoint import *
from ddpg.ddpg import DDPG

from holodecml.vae.optimizers import *
from holodecml.vae.tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *


# In[2]:


with open("config.yml") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)


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

# In[9]:


is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")

if is_cuda:
    torch.backends.cudnn.benchmark = True

print(f'Preparing to use device {device}')


# ### Load the models

# In[10]:


epochs = 1000
start_epoch = 0 

state_dim = 35
action_dim = 29 
max_state = 1.0
batch_size = 1024


# In[11]:


policy = DDPG(state_dim, action_dim, max_state, device)


# In[12]:


replay_buffer = ReplayBuffer(state_dim, action_dim, device, min_size = 1e4, max_size = 1e5)


# In[13]:


class Trainer:

    def __init__(self, 
                 policy, 
                 train_generator, 
                 validation_generator, 
                 test_generator, 
                 replay_buffer,
                 device,
                 start_epoch = 0,
                 epochs = 100,
                 batch_size = 64,
                 clip = 1.0,
                 max_state = 1.0,
                 expl_noise = 0.1):
        
        
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.policy = policy
        
        self.train_gen = train_generator
        self.valid_gen = validation_generator
        self.test_gen = test_generator
        
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        
        self.device = device
        self.clip = clip
        self.max_state = max_state
        
        self.expl_noise = expl_noise 
        
        
        # Gradient clipping through hook registration
        for p in self.policy.actor.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))
        for p in self.policy.critic.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))
        logger.info(f"Clipping gradients to range [-{clip}, {clip}]")
        
    
    def train_one_epoch(self, epoch, steps = 1e10):
        
        self.policy.actor.train()
        self.policy.critic.train()
        
        batches_per_epoch = int(self.train_gen.__len__()) 
        
        if batches_per_epoch > steps:
            batches_per_epoch = steps
        
        experiments = range(len(self.train_gen.experiment_subset))        
        batch_group_generator = tqdm(
            enumerate(experiments),
            total=batches_per_epoch, 
            leave=True
        )
        
        total_pts = 0
        episode_reward = []
        for batch_idx, exp in batch_group_generator:
            experiment_reward = []
            for i, (x,y,w) in enumerate(self.train_gen.__getitem__(exp)):
                x = x.view(1, x.size(0)).to(self.device)
                y = y.view(1, y.size(0)).to(self.device)
                w = w.to(self.device)
                
                state = x if i == 0 else next_state
                
                action = self.policy.actor(state)
                noise = torch.empty(self.policy.action_dim).normal_(
                    mean=0,std=self.max_state * self.expl_noise
                )
                action += noise.to(self.device) 
                #np.random.normal(0, self.max_state * self.expl_noise, size=self.policy.action_dim)
                
                next_state = state.clone()
                next_state[:, :self.policy.action_dim] += action # next state
                next_state = next_state.clamp(0.0, self.max_state)
                
                reward = torch.mean(1.0 - torch.abs((next_state[:, :self.policy.action_dim] - y)))
                value = self.policy.critic(next_state, action)
                
                self.replay_buffer.add(
                    state.cpu().detach().numpy(), 
                    action.cpu().detach().numpy(), 
                    next_state.cpu().detach().numpy(), 
                    reward.cpu().detach().numpy(), 
                    int(bool(i == 1438))
                )
                episode_reward.append(value.item())
                experiment_reward.append(value.item())
                total_pts += 1
        
                # Train agent after collecting sufficient data
                if (self.replay_buffer.min_size < total_pts) and epoch == 0:
                    continue
                else:
                    self.policy.train(self.replay_buffer, self.batch_size)
                    
            if batch_idx > 0 and (batch_idx % steps) == 0:
                break
                                
            # update tqdm
            ep_reward = np.mean(episode_reward)
            to_print = f"Episode {epoch} training reward: {ep_reward:.3f}"
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()
            
        return np.mean(episode_reward)
    
        
    def test(self, epoch):

        self.policy.actor.eval()
        self.policy.critic.eval()

        with torch.no_grad():
            batch_group_generator = tqdm(
                enumerate(range(len(self.valid_gen.experiment_subset))),
                total=len(self.valid_gen.experiment_subset), 
                leave=True
            )

            episode_reward = []
            for batch_idx, exp in batch_group_generator:
                experiment_reward = []
                for i, (x,y,w) in enumerate(self.valid_gen.__getitem__(exp)):
                    x = x.view(1, x.size(0)).to(self.device)
                    y = y.view(1, y.size(0)).to(self.device)
                    state = x if i == 0 else next_state.clone()
                    action = self.policy.actor(state).detach()
                    next_state = state.detach()
                    next_state[:, :self.policy.action_dim] += action # next state
                    next_state = next_state.clamp(0.0, self.max_state)    
                    reward = self.policy.critic(next_state, action).detach()
                    experiment_reward.append(reward.item())
                    episode_reward.append(reward.item())
                    #print(exp, i, batch_idx, reward.item())

                # update tqdm
                exp_reward = np.mean(episode_reward)
                to_print = f"Episode {epoch} validation reward: {exp_reward:.3f}"
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

        return np.mean(episode_reward)
    
    def train(self,
              scheduler,
              early_stopping,
              metrics_logger,
              steps = 1e10):
            
        logger.info(
            f"Training the model for up to {self.epochs} epochs starting at epoch {self.start_epoch}"
        )

        flag = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        for episode in range(self.start_epoch, self.epochs):
            train_loss = self.train_one_epoch(episode, steps)
            test_loss = self.test(episode)

            if scheduler is not None:
                scheduler.step(test_loss if flag else episode)
            
            early_stopping(
                episode, 
                test_loss, 
                self.policy.actor, 
                self.policy.actor_optimizer,
                self.policy.critic, 
                self.policy.critic_optimizer
            )

            # Write results to the callback logger 
            result = {
                "episode": episode,
                "train_loss": train_loss,
                "valid_loss": test_loss,
                "lr_actor": early_stopping.print_learning_rate(self.policy.actor_optimizer),
                "lr_critic": early_stopping.print_learning_rate(self.policy.critic_optimizer)
            }
            metrics_logger.update(result)

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break


# In[14]:


trainer = Trainer(
    policy, 
    train_data_set, 
    valid_data_set, 
    test_data_set, 
    replay_buffer,
    device = device,
    start_epoch = start_epoch,
    epochs = epochs,
    batch_size = batch_size
)


# In[15]:


# # Initialize LR annealing scheduler 
# if "ReduceLROnPlateau" in config["callbacks"]:
#     schedule_config = config["callbacks"]["ReduceLROnPlateau"]
#     scheduler = ReduceLROnPlateau(trainer.optimizer, **schedule_config)
#     #logging.info(
#     #    f"Loaded ReduceLROnPlateau learning rate annealer with patience {schedule_config['patience']}"
#     #)
# elif "ExponentialLR" in config["callbacks"]:
#     schedule_config = config["callbacks"]["ExponentialLR"]
#     scheduler = ExponentialLR(trainer.optimizer, **schedule_config)
#     #logging.info(
#     #    f"Loaded ExponentialLR learning rate annealer with reduce factor {schedule_config['gamma']}"
#     #)

scheduler = None

# Early stopping
checkpoint_config = config["callbacks"]["EarlyStopping"]
early_stopping = EarlyStopping(**checkpoint_config)

# Write metrics to csv each epoch
metrics_logger = MetricsLogger(**config["callbacks"]["MetricsLogger"])


# In[ ]:


trainer.train(scheduler, early_stopping, metrics_logger, steps = 100)


# In[ ]:


# batch_group_generator = tqdm(
#     enumerate(range(len(valid_data_set.experiment_subset))),
#     total=len(valid_data_set.experiment_subset), 
#     leave=True
# )


# In[ ]:


#policy.actor.eval()
#policy.critic.eval()


# In[ ]:


# epoch = 0
# episode_reward = []
# for batch_idx, exp in batch_group_generator:
#     experiment_reward = []
#     for i, (x,y,w) in enumerate(valid_data_set.__getitem__(exp)):
#         x = x.view(1, x.size(0)).to(device)
#         y = y.view(1, y.size(0)).to(device)
#         state = x if i == 0 else next_state.clone()
#         action = policy.actor(state).detach()
#         next_state = state.detach()
#         next_state[:, :policy.action_dim] += action # next state
#         next_state = next_state.clamp(0, 1.0)    
#         reward = policy.critic(next_state, action).detach()
#         experiment_reward.append(reward.item())
#         episode_reward.append(reward.item())
        
#         print(next_state, y, reward.item())
        
#     exp_reward = np.mean(episode_reward)
#     to_print = f"Episode {epoch} avg validation reward: {exp_reward:.3f}"
#     batch_group_generator.set_description(to_print)
#     batch_group_generator.update()


# In[ ]:




