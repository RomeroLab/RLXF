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

# ProtDataModule splits the data into three different datasets.
# Training data contains all 1-4 mutation variants in DMS dataset. 5 mutants are split into validation and test sets
class ProtDataModule(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data splitting"""

    def __init__(self, data_frame, batch_size, splits_path=None):
        # Call the __init__ method of the parent class
        super().__init__()

        # Store the batch size
        self.batch_size = batch_size
        self.data_df = data_frame
        
        if splits_path is not None:
            train_indices, val_indices, test_indices = self.load_splits(splits_path)
            # print(test_indices)
            
            # Shuffle the indices to ensure that the data from each cluster is mixed. Do I want this?
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)
            
            # Store the indices for the training, validation, and test sets
            self.train_idx = train_indices
            self.val_idx = val_indices
            self.test_idx = test_indices
                
        else:
            # New logic for splitting based on mutation count
            self.data_df['MutationCount'] = self.data_df['Mutations'].apply(self.count_mutations)
            train_indices = []
            val_indices = []
            test_indices = []
            
            gen = torch.Generator()
            gen.manual_seed(0)
            random.seed(0)

            # Proteins with 1 to 4 mutations go to training set
            train_indices.extend(self.data_df[self.data_df['MutationCount'] <= 4].index.tolist())

            # Proteins with 5 mutations are split between training and validation
            five_mut_indices = self.data_df[self.data_df['MutationCount'] == 5].index.tolist()
            random.shuffle(five_mut_indices)
            split_index = int(len(five_mut_indices) * 0.75)
            val_indices.extend(five_mut_indices[:split_index])
            test_indices.extend(five_mut_indices[split_index:])
            
            # Shuffle the indices to ensure that the data from each cluster is mixed
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)
            
            # Store the indices for the training, validation, and test sets
            self.train_idx = train_indices
            self.val_idx = val_indices
            self.test_idx = test_indices
            # print(test_indices)

    def count_mutations(self, mutation_str):
        """Count the number of mutations in the mutation string."""
        return len(mutation_str.split(','))

    # Prepare_data is called from a single GPU. Do not use it to assign state (self.x = y). Use this method to do
    # things that might write to disk or that need to be done only from a single process in distributed settings.
    def prepare_data(self):
        pass

    # Assigns train, validation and test datasets for use in dataloaders.
    def setup(self, stage=None):
              
        # Assign train/validation datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_data_frame = self.data_df.iloc[list(self.train_idx)]
            self.train_ds = SeqFcnDataset(train_data_frame)
            val_data_frame = self.data_df.iloc[list(self.val_idx)]
            self.val_ds = SeqFcnDataset(val_data_frame)
                    
        # Assigns test dataset for use in dataloader
        if stage == 'test' or stage is None:
            test_data_frame = self.data_df.iloc[list(self.test_idx)]
            self.test_ds = SeqFcnDataset(test_data_frame)
            
    #The DataLoader object is created using the train_ds/val_ds/test_ds objects with the batch size set during initialization of the class and shuffle=True.
    def train_dataloader(self):
        return data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return data_utils.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)
    def test_dataloader(self):
        return data_utils.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True)
    
    def save_splits(self, path):
        """Save the data splits to a file at the given path"""
        with open(path, 'wb') as f:
            pickle.dump((self.train_idx, self.val_idx, self.test_idx), f)

    def load_splits(self, path):
        """Load the data splits from a file at the given path"""
        with open(path, 'rb') as f:
            self.train_idx, self.val_idx, self.test_idx = pickle.load(f)
            
            train_indices = self.train_idx
            val_indices = self.val_idx
            test_indices = self.test_idx
            
        return train_indices, val_indices, test_indices

    def get_split_dataframes(self):
        """Return the split dataframes for training, validation, and testing."""
        # Ensure the setup method has been called to populate the indices
        self.setup()

        # Create the split DataFrames
        train_df = self.data_df.iloc[list(self.train_idx)]
        val_df = self.data_df.iloc[list(self.val_idx)]
        test_df = self.data_df.iloc[list(self.test_idx)]

        return train_df, val_df, test_df


# SeqFcnDataset is a data handling class.
# I convert amino acid sequences to torch tensors for model inputs
# I convert mean to torch tensors
class SeqFcnDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein sequence-function data"""

    def __init__(self, data_frame):
        self.data_df = data_frame

    def __getitem__(self, idx):
        sequence = torch.tensor(aa2ind(list(self.data_df.Sequence.iloc[idx]))) # Extract sequence at index idx
        labels = torch.tensor(self.data_df.iloc[idx, 8].tolist()).float() # Extract log mean fitness score for sequence at index idx and convert to a list
        return sequence, labels

    def __len__(self):
        return len(self.data_df)

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
