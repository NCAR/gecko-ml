from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import *
from torch import nn

import torch.nn.functional as F
import numpy as np
import logging
import optuna
import random
import torch
import os

from geckoml.torch.data_reader import *
from geckoml.torch.models import *

from aimlutils.torch.optimizers import *
from aimlutils.utils.tqdm import tqdm


logger = logging.getLogger(__name__)


class BoxBaseTrainer:
    
    def __init__(self, 
                 train_gen, 
                 valid_gen, 
                 dataloader, 
                 valid_dataloader,
                 model_conf,
                 optimizer_conf,
                 input_size,
                 output_size,
                 start_epoch = 0,
                 epochs = 100,
                 batches_per_epoch = 1e10,
                 window_size = 10,
                 teacher_force = True,
                 gamma = 0.5,
                 device = "cpu",
                 clip = 1.0,
                 path_save = "./",
                 loss_weights = [1, 1, 1, 1]):
        
        # Exit if the save directory does not exist
        if not os.path.isdir(path_save):
            raise OSError(
                f"You must create the Trainer save directory {path_save} before proceeding"
            )
        
        self.input_size = input_size 
        self.output_size = output_size
        
        # Initialize and build a model
        self.model = DenseNet(**model_conf)
        self.model.build(self.input_size, self.output_size)
        self.model = self.model.to(device)
        
        # Initialize an optimizer
        optimizer_type = optimizer_conf.pop("type")
        self.optimizer = LoadOptimizer(
            optimizer_type, 
            self.model.parameters(), 
            optimizer_conf["lr"], 
            optimizer_conf["weight_decay"]
        )
        
        # Set up the rest of the attributes
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.batch_size = dataloader.batch_size
        self.path_save = path_save
        self.device = device
        self.loss_weights = loss_weights
        
        self.start_epoch = start_epoch 
        self.epochs = epochs
        self.batches_per_epoch = batches_per_epoch
        self.window_size = window_size
        
        self.teacher_force = teacher_force
        self.gamma = gamma

        timesteps = self.train_gen.num_timesteps
        self.time_range = list(range(timesteps))
        
        # Print some config details 
        logger.info(f"Loaded a Trainer object with parameters:")
        logger.info(
            f"Training will start at epoch {self.start_epoch} and go for {self.epochs} epochs"
        )
        logger.info(f"box-mode using {timesteps} per experiment")
        logger.info(f"batch-size: {self.batch_size}")
        logger.info(f"box window size: {self.window_size}")
        logger.info(f"loss weights used during training: {self.loss_weights}")
        logger.info(f"teacher-force: {self.teacher_force}")
        if self.teacher_force:
            logger.info(f"teacher-forcing annealing rate: {self.gamma}")
        logger.info(f"results will be saved in {self.path_save}")
        
        # Gradient clipping through hook registration
        for p in self.model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))
        logger.info(f"Clipping gradients to range [-{clip}, {clip}]")
        
    def criterion(self, y_true, y_pred):
        
        if self.model.training:
            a, b, c, d = self.loss_weights
        else: # Do not use weights during validation
            a, b, c, d = 1.0, 1.0, 1.0, 1.0
        
        y_true_precursor = y_true[:, :, 0]
        y_pred_precursor = y_pred[:, :, 0]
        
        y_true_gas = y_true[:, :, 1:15]
        y_pred_gas = y_pred[:, :, 1:15]
        
        y_true_aero = y_true[:, :, 15:]
        y_pred_aero = y_pred[:, :, 15:]
        
        mse_precursor = a * nn.MSELoss()(y_true_precursor, y_pred_precursor)
        mse_gas = b * nn.MSELoss()(y_true_gas, y_pred_gas)
        mse_aero = c * nn.MSELoss()(y_true_aero, y_pred_aero)
        mse = (mse_precursor + mse_gas + mse_aero) / (a + b + c)
        
        kld_gas = nn.KLDivLoss()(
            F.log_softmax(y_pred_gas),
            F.softmax(y_true_gas)
        )
        kld_aero = nn.KLDivLoss()(
            F.log_softmax(y_pred_aero),
            F.softmax(y_true_aero)
        )
        
        return (mse + d * (kld_gas + kld_aero)) / (1.0 + d)
        
    def train_one_epoch(self, epoch):
        
        self.model.train()
        batches_per_epoch = int(np.ceil(self.train_gen.__len__() / self.batch_size))
        
        if batches_per_epoch > self.batches_per_epoch:
            batches_per_epoch = self.batches_per_epoch
        
        batch_group_generator = tqdm(
            self.dataloader,
            total=batches_per_epoch, 
            leave=True
        )    
        epoch_losses = {"loss": []}
        for batch_idx, (x, y) in enumerate(batch_group_generator):
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.window = random.sample(self.time_range, self.window_size) 
                        
            y_true, y_pred, weights = [], [], []
            for i in range(y.size(1)):
                next_x = self.model(x[:,i,:])
                
                if i in self.window:
                    y_true.append(y[:, i])
                    y_pred.append(next_x)                                
                
                if i < (y.size(1)-1):
                    if (epoch == self.epochs - 1) or (not self.teacher_force):
                        x = x.clone()
                        x[:, i+1, :self.output_size] = next_x                        
                    else:
                        cost = self.tf_annealer(epoch) 
                        idx = [bn for bn in range(x.size(0)) if cost < random.random()]
                        if len(idx) > 0:
                            x = x.clone() # next line is in-place op, messes up grad. comp. 
                            x[idx, i+1, :self.output_size] = next_x[idx]  
                            
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
            
            if batch_idx % batches_per_epoch == 0 and batch_idx > 0:
                break
            
        return np.mean(epoch_losses["loss"])
            
    def test(self, epoch):

        self.model.eval()
        batches_per_epoch = int(np.ceil(self.valid_gen.__len__() / self.batch_size))

        with torch.no_grad():

            batch_group_generator = tqdm(
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
                        x[:, i+1, :self.output_size] = next_x # never "force" on eval
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
              metrics_logger,
              trial = None):
        
        logger.info(
            f"Training the model for up to {self.epochs} epochs starting at epoch {self.start_epoch}"
        )
        
        flag = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        val_loss = []
        for epoch in range(self.start_epoch, self.epochs):
            try:
                train_loss = self.train_one_epoch(epoch)
                test_loss = self.test(epoch)

                if trial:
                    trial.report(test_loss, step=epoch+1)
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
                val_loss.append(test_loss)
            
            except Exception as E: # CUDA memory overflow
                if "CUDA" in str(E):
                    logger.info(
                        "Failed to train the model due to GPU memory overflow."
                    )
                    raise optuna.TrialPruned() if trial else OSError(f"{str(E)}")
                else:
                    raise OSError(f"{str(E)}")
            
            if trial:
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
                
        return np.mean(val_loss)
    

    def tf_annealer(self, epoch):
        return 1.0 * self.gamma ** epoch # 1/(1 + self.decay * epoch) 