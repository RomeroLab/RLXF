#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from collections import OrderedDict
from torchtext import vocab # This package can give problems sometimes, it may be necessary to downgrade to a specific version
from pytorch_lightning.loggers import CSVLogger
from random import choice
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn import metrics
import torchmetrics
import enum
import argparse
from argparse import ArgumentParser
import os
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import csv


# In[2]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
df = pd.read_pickle("./CreiLOV_4cluster_df.pkl") # load preprocessed CreiLOV data

# Model architecture defined
class MLP(pl.LightningModule):
    """PyTorch Lightning Module that defines model and training"""
      
    # define network
    def __init__(self, learning_rate, batch_size, epochs, slen):
        super().__init__()
        
        # Creates an embedding layer in PyTorch and initializes it with the pretrained weights stored in aaindex
        self.embed = nn.Embedding.from_pretrained(torch.eye(21),freeze=True)
        self.slen = slen # CreiLOV sequence length
        self.ndim = self.embed.embedding_dim # dimensions of AA embedding
        self.dropout = nn.Dropout(p=0.2)
        
        # fully connected neural network
        ldims = [self.slen*self.ndim,400,1]
        self.dropout = nn.Dropout(p=0.1)
        self.linear_1 = nn.Linear(ldims[0], ldims[1])
        self.linear_2 = nn.Linear(ldims[1], ldims[2])
        
        # learning rate
        self.learning_rate = learning_rate
        self.save_hyperparameters('learning_rate', 'batch_size', 'epochs', 'slen') # log hyperparameters to file
             
    # MLP (fully-connected neural network with one hidden layer)
    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        x = self.embed(x)
        x = x.view(-1,self.ndim*self.slen)
        x = self.linear_1(x)
        x = self.dropout(x)  # Add dropout after the first fully connected layer
        x = F.relu(x) # Activation function
        x = self.linear_2(x)
        return x
      
    def training_step(self, batch, batch_idx):
        sequence,scores = batch
        sequence,scores = sequence.to(device), scores.to(device)
        scores = scores.unsqueeze(1)  # Add an extra dimension to the target tensor
        output = self(sequence)
        loss = nn.MSELoss()(output, scores) # Calculate MSE
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step = False, on_epoch=True) # reports MSE loss to model
        return loss

    def validation_step(self, batch, batch_idx):
        sequence,scores = batch
        scores = scores.unsqueeze(1)  # Add an extra dimension to the target tensor
        output = self(sequence)
        loss = nn.MSELoss()(output, scores) # Calculate MSE
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step = False, on_epoch=True) # reports MSE loss to model
        return loss

    def test_step(self, batch):
        sequence,scores = batch
        output = self(sequence)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.001) # Weight Decay to penalize too large of weights
        return optimizer
    
    def predict(self, sequence):
        # ind = torch.tensor(aa2ind(list(sequence))) # Convert the amino acid sequence to a tensor of indices
        # x = ind.view(1,-1) # Add a batch dimension to the tensor (put here instead of forward function)
        pred = self(sequence) # Apply the model to the tensor to get the prediction
        return pred


# In[7]:


######################################### hyperparameter that can be altered #########################################
# Altering hyperparameters can sometimes change model performance or training time
learning_rate = 1e-6 # important to optimize this
batch_size = 128 # typically powers of 2: 32, 64, 128, 256, ...
epochs = 2000 # rounds of training
slen = len(WT) # length of protein
num_models = 100 # number of models in ensemble
patience = 400
######################################### hyperparameter that can be altered #########################################
