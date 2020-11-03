from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import *
import torch.nn.functional as F
from torch import nn

import numpy as np
import logging
import random
import torch
import tqdm


logger = logging.getLogger(__name__)


class BaseTrainer:
    
    def __init__(self, 
                 model, 
                 optimizer,
                 train_gen, 
                 valid_gen, 
                 dataloader, 
                 valid_dataloader,
                 start_epoch = 0,
                 epochs = 100,
                 window_size = 10,
                 teacher_force = True,
                 gamma = 0.5,
                 device = "cpu",
                 clip = 1.0,
                 path_save = "./"):
        
        self.model = model
        self.outsize = model.output_size
        self.optimizer = optimizer
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.batch_size = dataloader.batch_size
        self.path_save = path_save
        self.device = device
        
        self.start_epoch = start_epoch 
        self.epochs = epochs
        self.window_size = window_size
        
        self.teacher_force = teacher_force
        self.gamma = gamma
        
        #self.criterion = nn.MSELoss()
        
        timesteps = self.train_gen.num_timesteps
        self.time_range = list(range(timesteps))
                
        # Gradient clipping through hook registration
        for p in self.model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))
        logger.info(f"Clipping gradients to range [-{clip}, {clip}]")
        
        # Create the save directory if it does not exist
        try:
            os.makedirs(path_save)
        except:
            pass
        
    def criterion(self, y_true, y_pred):
        
        y_true_precursor = y_true[:, :, 0]
        y_pred_precursor = y_pred[:, :, 0]
        
        y_true_gas = y_true[:, :, 1:15]
        y_pred_gas = y_pred[:, :, 1:15]
        
        y_true_aero = y_true[:, :, 15:]
        y_pred_aero = y_pred[:, :, 15:]
        
        mse_precursor = nn.MSELoss()(y_true_precursor, y_pred_precursor)
        mse_gas = nn.MSELoss()(y_true_gas, y_pred_gas)
        mse_aero = nn.MSELoss()(y_true_aero, y_pred_aero)
        mse = mse_precursor + mse_gas + mse_aero
        
        kld_gas = nn.KLDivLoss()(
            F.log_softmax(y_pred_gas),
            F.softmax(y_true_gas)
        )
        kld_aero = nn.KLDivLoss()(
            F.log_softmax(y_pred_aero),
            F.softmax(y_true_aero)
        )
        return mse + (kld_gas + kld_aero)
        
    def train_one_epoch(self, epoch):
        
        self.model.train()
        batches_per_epoch = int(np.ceil(self.train_gen.__len__() / self.batch_size))
        batch_group_generator = tqdm.tqdm(
            self.dataloader,
            total=batches_per_epoch, 
            leave=True
        )
        
#         if epoch == 0:
#             self.idx = 0
#         else:
#             self.idx += 1 
#         self.window = self.time_range[self.idx * self.window_size : (self.idx + 1) * self.window_size]
        
#         if len(self.window) == 0:
#             self.idx = 0
#             self.window = self.time_range[self.idx * self.window_size : (self.idx + 1) * self.window_size]
            
        epoch_losses = {"loss": []}
        for (x, y) in batch_group_generator:
            x = x.to(self.device)
            y = y.to(self.device)

            #window = random.choice(self.time_range)
            #window = [window + i for i in range(self.window_size)]
            #window = [x for x in window if x <= max(self.time_range)]
            
            #window = random.sample(self.time_range, self.window_size) # only works when forcing = 1.0
                        
            y_true, y_pred, weights = [], [], []
            for i in range(y.size(1)):
                next_x = self.model(x[:,i,:])
                y_true.append(y[:, i])
                y_pred.append(next_x)                                
                if i < (y.size(1)-1):
                    if (epoch == self.epochs - 1) or (not self.teacher_force):
                        x = x.clone()
                        x[:, i+1, :self.outsize] = next_x                        
                    else:
                        cost = self.tf_annealer(epoch) 
                        idx = [bn for bn in range(x.size(0)) if cost < random.random()]
                        if len(idx) > 0:
                            x = x.clone() # next line is in-place op, messes up grad. comp. 
                            x[idx, i+1, :self.outsize] = next_x[idx]  
                            
            y_true = torch.stack(y_true).permute(1,0,2)
            y_pred = torch.stack(y_pred).permute(1,0,2)
            loss = self.criterion(y_true, y_pred)   
            epoch_losses["loss"].append(loss.item())

            # backprop after experiment
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update tqdm
            to_print = "loss: {:.3f}".format(np.mean(epoch_losses["loss"]))
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()
            
        return np.mean(epoch_losses["loss"])
            
    def test(self, epoch):

        self.model.eval()
        batches_per_epoch = int(np.ceil(self.valid_gen.__len__() / self.batch_size))

        with torch.no_grad():

            batch_group_generator = tqdm.tqdm(
                self.valid_dataloader,
                total=batches_per_epoch, 
                leave=True
            )
            
            epoch_losses = {"loss": []}
            for (x, y) in batch_group_generator:
                x = x.to(self.device)
                y = y.to(self.device)
                y_true, y_pred = [], []
                for i in range(y.size(1)):
                    next_x = self.model(x[:,i,:])
                    y_true.append(y[:, i])
                    y_pred.append(next_x)
                    if i < (y.shape[1]-1):
                        x[:, i+1, :self.outsize] = next_x # never "force" on eval
                y_true = torch.stack(y_true).permute(1,0,2)
                y_pred = torch.stack(y_pred).permute(1,0,2)
                loss = self.criterion(y_true, y_pred)
                epoch_losses["loss"].append(loss.item())

                # update tqdm
                to_print = "val_loss: {:.3f}".format(np.mean(epoch_losses["loss"]))
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()
            
        return np.mean(epoch_losses["loss"]) 
    
    
    def train(self,
              scheduler,
              early_stopping,
              metrics_logger):
        
        logger.info(
            f"Training the model for up to {self.epochs} epochs starting at epoch {self.start_epoch}"
        )
        
        flag = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_one_epoch(epoch)
            test_loss = self.test(epoch)

            scheduler.step(test_loss if flag else epoch)
            early_stopping(epoch, test_loss, self.model, self.optimizer)

            # Write results to the callback logger 
            result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": test_loss,
                "lr": early_stopping.print_learning_rate(self.optimizer),
                "teacher_forcing_score": self.tf_annealer(epoch) if self.teacher_force else 1.0
            }
            metrics_logger.update(result)

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

    def tf_annealer(self, epoch):
        return 1.0 * self.gamma ** epoch # 1/(1 + self.decay * epoch) 