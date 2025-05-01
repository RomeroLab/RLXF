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
    

# # # load preprocessed CreiLOV data
# df = pd.read_pickle("CreiLOV_4cluster_df.pkl") # May need to edit after I get MDS results
# # # print(df)

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap

# SeqFcnDataset is a data handling class.
# I convert amino acid sequences to torch tensors for model inputs
# I convert mean to torch tensors
class SeqFcnDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein sequence-function data"""

    def __init__(self, data_frame):
        self.data_df = data_frame

    def __getitem__(self, idx):
        sequence = torch.tensor(aa2ind(list(self.data_df.Sequence.iloc[idx]))) # Extract sequence at index idx
        labels = torch.tensor(self.data_df.iloc[idx, 8].tolist()).float() # Extract mean fitness score for sequence at index idx and convert to a list
        return sequence, labels

    def __len__(self):
        return len(self.data_df)

# ProtDataModule splits the data into three different datasets.
# Training data is used during model training
# Validation data is used to evaluate the model after epochs
# We want validation loss to be less than the training loss while both values decrease
# If validation loss > training loss, the model is likely overfit and cannot generalize to samples outside training set
# Testing data is used to evaluate the model after model training is complete
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
            
            # Initialize empty lists to hold the indices for the training, validation, and test sets
            train_indices = []
            val_indices = []
            test_indices = []
            
            gen = torch.Generator()
            gen.manual_seed(0)
            
            # Loop over each unique cluster in the DataFrame
            for cluster in self.data_df['Cluster'].unique():
                # Get the indices of the rows in the DataFrame that belong to the current cluster
                cluster_indices = self.data_df[self.data_df['Cluster'] == cluster].index.tolist()
                # Define the fractions of the data that should go to the training, validation, and test sets
                train_val_test_split = [0.8, 0.1, 0.1]
                # Calculate the number of samples that should go to each set based on the fractions defined above
                n_train_val_test = np.round(np.array(train_val_test_split)*len(cluster_indices)).astype(int)
                # If the sum of the calculated numbers of samples is less than the total number of samples in the cluster,
                # increment the number of training samples by 1
                if sum(n_train_val_test)<len(cluster_indices): n_train_val_test[0] += 1 # necessary when round is off by 1
                # If the sum of the calculated numbers of samples is more than the total number of samples in the cluster,
                # decrement the number of training samples by 1
                if sum(n_train_val_test)>len(cluster_indices): n_train_val_test[0] -= 1 
                # Split the indices of the current cluster into training, validation, and test sets
                train, val, test = data_utils.random_split(cluster_indices,n_train_val_test,generator=gen)
                # Add the indices for the current cluster to the overall lists of indices
                train_indices.extend(train.indices)
                val_indices.extend(val.indices)
                test_indices.extend(test.indices)
            
            # Shuffle the indices to ensure that the data from each cluster is mixed
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)
            
            # Store the indices for the training, validation, and test sets
            self.train_idx = train_indices
            self.val_idx = val_indices
            self.test_idx = test_indices
            # print(test_indices)

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
            
    #The DataLoader object is created using the train_ds/val_ds/test_ds objects with the batch size set during
    # initialization of the class and shuffle=True.
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


# PTLModule is the actual neural network. Model architecture can be altered here.
class CNN(pl.LightningModule):
    """PyTorch Lightning Module that defines model and training"""
      
    # define network
    def __init__(self, slen, ks, learning_rate, epochs, batch_size, factor_2=2, factor_3=2, dim_4=400, dim_5=50):
        super().__init__()
        
        # Creates an embedding layer in PyTorch and initializes it with the pretrained weights stored in aaindex
        self.embed = nn.Embedding(len(AAs), 16) # maps integer indices (a.a.'s)' to 16-dimensional vectors
        # self.embed = nn.Embedding.from_pretrained(torch.eye(len(AAs)), freeze=True) for one hot encoding
        self.slen = slen # CreiLOV sequence length
        self.ndim = self.embed.embedding_dim # dimensions of AA embedding
        self.ks = ks # kernel size describes how many positions the neural network sees in each convolution
        
        conv_out_dim = factor_3*self.ndim # determines output size of last conv layer
        self.nparam = slen*conv_out_dim # desired (flattened) output size for last convolutional layer
        # self.dropout1 = nn.Dropout(p=0)
        self.dropout2 = nn.Dropout(p=0.2)
        pad = int((self.ks - 1)/2)

        # Convolutional layers block
        self.enc_conv_1 = torch.nn.Conv1d(in_channels= self.ndim, out_channels=factor_2*self.ndim, kernel_size=ks, padding=pad)
        self.enc_conv_2 = torch.nn.Conv1d(in_channels= factor_2*self.ndim, out_channels=conv_out_dim, kernel_size=ks, padding=pad) 
        
        # Fully connected layers block
        self.linear1 = nn.Linear(self.nparam, dim_4)
        self.linear2 = nn.Linear(dim_4,dim_5)
        self.linear3 = nn.Linear(dim_5,1)
        # print(self.nparam)
        
        # learning rate
        self.learning_rate = learning_rate
        self.save_hyperparameters('learning_rate', 'batch_size', 'ks', 'epochs', 'slen') # log hyperparameters to file
             
    def forward(self, x):
        
        x = self.embed(x) # PyTorch will learn embedding with the specified dimensions
        
        # Convolutional layers block
        x = x.permute(0,2,1) # swap length and channel dims        
        x = self.enc_conv_1(x)   
        x = F.leaky_relu(x) # this is an activation fucniton and is non-linear component of a neural network
        # x = self.dropout1(x)
        
        x = self.enc_conv_2(x)
        x = F.leaky_relu(x) # this is an activation fucniton and is non-linear component of a neural network
        # x = self.dropout1(x)
        
        # Fully connected layers block
        x = x.view(-1,self.nparam) # flatten (input for linear/FC layers must be 1D)
        x = self.linear1(x)
        x = self.dropout2(x)  # Add dropout after the first fully connected layer
        x = F.relu(x) # this is an activation fucniton and is non-linear component of a neural network
        
        x = x.view(-1,dim_4)
        x = self.linear2(x)
        x = self.dropout2(x)  # Add dropout after the first fully connected layer
        x = F.relu(x) # this is an activation fucniton and is non-linear component of a neural network

        x = x.view(-1,dim_5)
        x = self.linear3(x)
        
        return x
      
    def training_step(self, batch, batch_idx):
        sequence,scores = batch
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
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) # No weight decay
        return optimizer
    
    def predict(self, sequence):
        # ind = torch.tensor(aa2ind(list(sequence))) # Convert the amino acid sequence to a tensor of indices
        x = sequence.view(1,-1) # Add a batch dimension to the tensor (put here instead of forward function)
        pred = self(x) # Apply the model to the tensor to get the prediction
        return pred # .detach().numpy() # Detach the prediction from the computation graph and convert it to a NumPy array

    def predict_2(self, sequence):
        # ind = torch.tensor(aa2ind(list(sequence))) # Convert the amino acid sequence to a tensor of indices
        # x = sequence.view(1,-1) # Add a batch dimension to the tensor (put here instead of forward function)
        pred = self(sequence) # Apply the model to the tensor to get the prediction
        return pred # .detach().numpy() # Detach the prediction from the computation graph and convert it to a NumPy array
    
# 'pred.detach().numpy()' is necessary because PyTorch tensors are part of a computational graph that tracks the operations performed on them, and the prediction tensor would otherwise retain a reference to this graph even after
# it has been returned from the function. By detaching the tensor, we break its connection to the graph and prevent any further computation on it. Converting the tensor to a NumPy array is useful for downstream processing,
# as many scientific libraries and tools in Python operate on NumPy arrays rather than PyTorch tensors.

######################################### hyperparameter that can be altered #########################################
# Altering hyperparameters can sometimes change model performance or training time
batch_size = 32 # typically powers of 2: 32, 64, 128, 256, ...
slen = len(WT) # length of protein
learning_rate = 0.000001 # important to optimize this
ks = 33 # kernel size, determines how many positions neural network sees in each convolution layer
epochs = 1000 # rounds of training
WD = 0.001
factor_2 = 2
factor_3 = 2
dim_4 = 400
dim_5 = 50
######################################### hyperparameter that can be altered #########################################
