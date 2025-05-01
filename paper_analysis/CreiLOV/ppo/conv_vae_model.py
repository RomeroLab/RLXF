#!/usr/bin/env python
# coding: utf-8

# In[2]:


### Importing Modules
import numpy as np
import pandas as pd
import pickle
from Bio import AlignIO
import math
import argparse
from argparse import ArgumentParser
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pytorch_lightning as pl
from collections import OrderedDict
from torchtext import vocab
from pytorch_lightning.loggers import CSVLogger
import random
from random import choice
import pathlib
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap

# Load Data from MSA for ConvVAE format
def get_msa_from_fasta(filename):
    import Bio.SeqIO
    with open(filename, "rt") as fh: 
        return [r[1] for r in Bio.SeqIO.FastaIO.SimpleFastaParser(fh)]

# SeqFcnDataset is a data handling class.
# I convert amino acid sequences to torch tensors for model inputs
# I convert mean to torch tensors
class SeqDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein sequence-function data"""

    def __init__(self, data_frame):
        self.data_df = data_frame

    def __getitem__(self, idx):
        sequence = torch.tensor(aa2ind(list(self.data_df.Sequence.iloc[idx]))) # Extract sequence at index idx
        # labels = torch.tensor(self.data_df.iloc[idx, 4].tolist()).float() # Extract mean fitness score for sequence at index idx and convert to a list
        return sequence # , labels

    def __len__(self):
        return len(self.data_df)

# ProtMSA is a helper class for loading and formating the data from MSA to a torch tensor
class ProtMSA(torch.utils.data.Dataset):
   
    def __init__(self, MSA):
        self.MSA = MSA

    def __getitem__(self, idx):
        # index the MSA
        sequence = torch.tensor(aa2ind(list(self.MSA[idx])))
        return sequence

    def __len__(self):
        return len(self.MSA)

# ProtDataModule is a helper class for splitting data into Training, Validation, and Test splits
class ProtDataModule(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data splitting"""

    def __init__(self, MSA, batch_size, sample_weights, splits_path=None):
        super().__init__()
        
        self.MSA = MSA
        self.batch_size = batch_size
        train_val_test_split = [0.9 - 100/243682, 0.1, 100/243682] # 100 Test Sequences
        self.sample_weights = sample_weights
        if self.sample_weights is not None:
            assert(len(self.MSA) == self.sample_weights.shape[0])

        if splits_path is not None:
            train_idx, val_idx, test_idx = self.load_splits(splits_path)
            # print(test_idx)
            
            # Store the indices for the training, validation, and test sets
            self.train_idx = train_idx
            self.val_idx = val_idx
            self.test_idx = test_idx
                
        else:
            n_train_val_test = np.round(np.array(train_val_test_split)*len(MSA)).astype(int) # Calculate the number of samples that should go to each set based on the fractions defined above
            if sum(n_train_val_test)<len(MSA): n_train_val_test[0] += 1 # If the sum of the calculated numbers of samples is less than the total number of samples in the cluster, increment the number of training samples by 1
            if sum(n_train_val_test)>len(MSA): n_train_val_test[0] -= 1 # If the sum of the calculated numbers of samples is more than the total number of samples in the cluster, decrement the number of training samples by 1
            self.train_idx, self.val_idx, self.test_idx = data_utils.random_split(list(range(len(MSA)),n_train_val_test))

    def prepare_data(self):
        # prepare_data is called from a single GPU. Do not use it to assign state (self.x = y)
        # use this method to do things that might write to disk or that need to be done only from a single process
        # in distributed settings.
        pass
        
    def setup(self, stage=None):
              
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_MSA = [self.MSA[i] for i in self.train_idx]
            self.train_MSA = ProtMSA(train_MSA)
            self.train_sample_weights = self.sample_weights[self.train_idx]
            
            val_MSA = [self.MSA[i] for i in self.val_idx]
            self.val_MSA = ProtMSA(val_MSA)
            self.val_sample_weights = self.sample_weights[self.val_idx]
            
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            test_MSA = [self.MSA[i] for i in self.test_idx]
            self.test_MSA = ProtMSA(test_MSA)
            self.test_sample_weights = self.sample_weights[self.test_idx]

    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.sample_weights is not None:
            sampler = data_utils.WeightedRandomSampler(
                         weights=self.train_sample_weights,
                         num_samples=len(self.train_sample_weights), 
                                 replacement=False)
            shuffle = False
        return data_utils.DataLoader(self.train_MSA, sampler=sampler,
                    batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self):
        sampler=None
        shuffle = True
        if self.sample_weights is not None:
            sampler = data_utils.WeightedRandomSampler(
                         weights=self.val_sample_weights,
                         num_samples=len(self.val_sample_weights), 
                                 replacement=False)
            shuffle = False
        return data_utils.DataLoader(self.val_MSA, sampler=sampler,
                        batch_size=self.batch_size, shuffle=shuffle)

    def test_dataloader(self):
        sampler = None
        return data_utils.DataLoader(self.test_MSA, sampler=sampler,
                batch_size=self.batch_size)

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

class ConvVAE(pl.LightningModule):
    def __init__(self, slen=119, ks=17, nlatent=64, learning_rate=0.0001, epochs=1000, n_cycle=1, factor_2=16, factor_3=1, dim_4=400):
        super().__init__()
        
        # The VAE uses a probabilistic approach to encoding the input data, which is why it generates a mean and a
        # variance for each of the latent variables. This is done to ensure that the encoded representation is not
        # overfitting to the input data and is instead learning the underlying structure of the data.

        # During training, the model minimizes two types of loss: the reconstruction loss and the Kullback-Leibler
        # (KL) divergence. The reconstruction loss measures how well the model is able to reconstruct the input data
        # from the encoded representation, while the KL divergence measures how well the encoded representation
        # follows a standard normal distribution.

        # The mean and variance of the latent variables are used to generate samples from the latent space during
        # training. The mean is used as the center of a normal distribution, and the variance is used to scale the
        # distribution. This sampling is done to generate new, novel sequences that are similar to the ones seen 
        # during training.
      
        ## GENERAL INPUT PARAMETERS
        self.slen = slen # sequence length
        self.ks = ks # kernel size
        self.nlatent = nlatent # num latent vars
        self.learning_rate = learning_rate
        self.pad = 1
        self.n_cycle = n_cycle
        
        ################################################################################################
        # Introducing Cyclical Annealing Schedule
        
        #  function  = 1 − cos(a), where a scans from 0 to pi/2
        def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
            L = np.ones(n_epoch+1)
            period = n_epoch/n_cycle
            step = (stop-start)/(period*ratio) # step is in [0,1]

            for c in range(n_cycle):
                v , i = start , 0
                while v <= stop and (int(i+c*period) < n_epoch):
                    L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
                    v += step
                    i += 1
            return L
        
        # Annealing parameters
        self.n_epochs = epochs
        self.current_ep = 0
        self.kl_weights = frange_cycle_cosine(0, 1, self.n_epochs,self.n_cycle)
        ################################################################################################
    
        ## ENCODER
        # self.embed = nn.Embedding(21, 16)
        self.embed = nn.Embedding.from_pretrained(torch.eye(21),freeze=True)
        self.edim = self.embed.embedding_dim # dimensions of AA embedding
        output_conv1 = self.slen - self.ks + 2*self.pad + 1
        output_conv2 = output_conv1 - self.ks + 2*self.pad + 1
        self.nparam = output_conv2 * factor_3 * self.edim
        self.enc_conv_1 = torch.nn.Conv1d(in_channels=  self.edim, out_channels=factor_2*self.edim, kernel_size=ks, padding=self.pad)
        self.enc_conv_2 = torch.nn.Conv1d(in_channels=factor_2*self.edim, out_channels=factor_3*self.edim, kernel_size=ks, padding=self.pad)
        self.linear_postConv = torch.nn.Linear(self.nparam,dim_4)
        self.z_mean = torch.nn.Linear(dim_4,nlatent)
        self.z_log_var = torch.nn.Linear(dim_4,nlatent)

        ## DECODER
        self.dec_linear_1 = torch.nn.Linear(nlatent,dim_4)
        self.dec_linear_2 = torch.nn.Linear(dim_4,self.nparam)
        self.dec_deconv_1 = torch.nn.ConvTranspose1d(in_channels=factor_3*self.edim, out_channels=factor_2*self.edim, kernel_size=ks, padding=self.pad)
        self.dec_deconv_2 = torch.nn.ConvTranspose1d(in_channels=factor_2*self.edim, out_channels=  self.edim, kernel_size=ks, padding=self.pad)
        self.nembed = self.embed.num_embeddings
        self.rev_embed = torch.nn.Linear(self.edim,self.nembed)
        
        # save hyperparameters for logging 
        self.save_hyperparameters()
        
    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1), device=self.device)
        
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
    
    def encoder(self,x):
        x = self.embed(x)
        x = x.permute(0,2,1) # swap length and channel dims
        x = self.enc_conv_1(x)
        x = F.leaky_relu(x)
        x = self.enc_conv_2(x)
        x = F.leaky_relu(x)
        x = x.view(-1,self.nparam) # flatten
        x = self.linear_postConv(x)
        x = F.relu(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, encoded

    def decoder(self, encoded):
        x = self.dec_linear_1(encoded)
        x = F.relu(x)
        x = self.dec_linear_2(x)
        x = x.view(-1,factor_3*self.edim,self.slen + 2*(- self.ks + 2*self.pad + 1))
        x = self.dec_deconv_1(x)
        x = F.leaky_relu(x)
        x = self.dec_deconv_2(x)
        x = F.leaky_relu(x)
        x = x.permute(0,2,1) # swap channel and length dims
        x = self.rev_embed(x)
        decoded = x.permute(0,2,1) # need to permute back
        return decoded
    
    def forward(self, x):
        z_mean, z_log_var, encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return z_mean, z_log_var, encoded, decoded
    
    ################################################################################################
    def kl_weight(self):
        # Cyclical annealing schedule
        return self.kl_weights[self.current_ep]
    ################################################################################################
        
    def training_step(self, batch, batch_idx):
        # pass through network 
        z_mean, z_log_var, encoded, decoded = self(batch)

        # cost = reconstruction loss + Kullback-Leibler divergence
        kl_divergence = (0.5 * (z_mean**2 + torch.exp(z_log_var) - z_log_var - 1)).sum()
        ce_loss = F.cross_entropy(decoded,batch,reduction='sum')
        cost = self.kl_weight() * kl_divergence + ce_loss
        # cost = kl_divergence + ce_loss
        
        # log 
        self.log("train_ce_loss", ce_loss, prog_bar=True, logger=True, on_step = False, on_epoch=True)
        self.log("val_kl_divergence", kl_divergence, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return cost

    def validation_step(self, batch, batch_idx):
        # pass through network 
        z_mean, z_log_var, encoded, decoded = self(batch)

        # cost = reconstruction loss + Kullback-Leibler divergence
        kl_divergence = (0.5 * (z_mean**2 + torch.exp(z_log_var) - z_log_var - 1)).sum()
        ce_loss = F.cross_entropy(decoded,batch,reduction='sum')
        cost = self.kl_weight() * kl_divergence + ce_loss 
        # cost = kl_divergence + ce_loss
        
        # log 
        self.log("val_ce_loss", ce_loss, prog_bar=True, logger=True, on_step = False, on_epoch=True)
        self.log("val_kl_divergence", kl_divergence, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return cost
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def on_validation_epoch_end(self):
#         # Print KL weight
#         print(f"Epoch: {self.current_ep}, KL Weight: {self.kl_weight()}")
        
        # Update current epoch
        self.current_ep += 1

# # Load Data for Model Training, Validation, and Testing
# MSA = get_msa_from_fasta("CreiLOV_MSA_UniRef90_2.fasta")
# weights = (np.load('CreiLOV_MSA_UniRef90_2_reweights.npy'))

####################################################################################################################
# define hyperparameters: changing these can sometimes change model performance or training time
batch_size = 32
ks = 17
nlatent = 64
epochs = 1000
learning_rate = 0.0001
slen = len(WT)
n_cycle = 1
factor_2 = 16
factor_3 = 1
dim_4 = 400
#####################################################################################################################





