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


# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
df = pd.read_pickle("./data/for_reward_model/CreiLOV_4cluster_df.pkl") # load preprocessed CreiLOV data
df.head()


# In[3]:


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


# In[4]:


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


# In[5]:


# dm = ProtDataModule(df, 32)
# dm.save_splits('./data/for_reward_model/CreiLOV_EnsMLP_splits_df.pkl')
# train_indices, val_indices, test_indices = dm.load_splits('./data/for_reward_model/CreiLOV_EnsMLP_splits_df.pkl')
# train_df, val_df, test_df = dm.get_split_dataframes()

# print("Training DataFrame:")
# print(train_df.head())  # Display first few rows of the training DataFrame
# print(len(train_df))

# # # # print("\nValidation DataFrame:")
# # # # print(val_df.head())  # Display first few rows of the validation DataFrame
# # # # print(len(val_df))

# # # # print("\nTest DataFrame:")
# # # # print(test_df.head())  # Display first few rows of the test DataFrame
# # # # print(len(test_df))

# # # Checking the values for MutationCount in each DataFrame
# # print("MutationCount values in Training DataFrame:")
# # print(train_df['MutationCount'].value_counts())

# # # # print("\nMutationCount values in Validation DataFrame:")
# # # # print(val_df['MutationCount'].value_counts())

# # # # print("\nMutationCount values in Test DataFrame:")
# # # # print(test_df['MutationCount'].value_counts())


# In[6]:


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
        x = self.embed(x)
        x = x.view(-1,self.ndim*self.slen)
        x = self.linear_1(x)
        x = self.dropout(x)  # Add dropout after the first fully connected layer
        x = F.relu(x) # Activation function
        x = self.linear_2(x)
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
        return optimizer
    
    def predict(self, sequence):
        ind = torch.tensor(aa2ind(list(sequence))) # Convert the amino acid sequence to a tensor of indices
        x = ind.view(1,-1) # Add a batch dimension to the tensor (put here instead of forward function)
        pred = self(x) # Apply the model to the tensor to get the prediction
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

# Split up data
dm = ProtDataModule(df,batch_size,'./data/for_reward_model/CreiLOV_EnsMLP_splits_df.pkl') # dm an instance of the class defined above, see notes above for its purpose

# Train 100 models
for i in range(num_models):
    # Resubstantiate the model for each training iteration
    model = MLP(learning_rate, batch_size, epochs, slen)
    logger_name = f'MLP'
    logger = CSVLogger('logs', name=logger_name)
    checkpoint_callback = ModelCheckpoint(dirpath='./trained_models/MLP_reward_models',filename=f'best_model_v{i}',monitor='val_loss',mode='min',save_top_k=1) # Define the model checkpoint callback with version number in the filename
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min') # Use early stopping
    trainer = pl.Trainer(logger=logger, max_epochs=epochs, callbacks=[early_stopping, checkpoint_callback], enable_progress_bar=False) # Trainer with early stopping and checkpointing
    trainer.fit(model, dm) # Train the model
    metrics_file_name = f'metrics.csv'
    pt_metrics = pd.read_csv(f'logs/MLP/version_{i}/metrics.csv')


# In[8]:

############################### Plot loss curves ###############################
train_losses = []
val_losses = []

# Loop over the models and collect the loss curves
for i in range(num_models):
    try:
        pt_metrics = pd.read_csv(f'logs/MLP/version_{i}/metrics.csv')
        train = pt_metrics[~pt_metrics.train_loss.isna()]
        val = pt_metrics[~pt_metrics.val_loss.isna()]
        max_epochs = max(len(train), len(val))  # Update the maximum number of epochs
        train_losses.append(train.train_loss.tolist())
        val_losses.append(val.val_loss.tolist())
    except FileNotFoundError:
        print(f"Metrics file for version {i} not found.")
        continue

# Find the maximum length of any training/validation loss array
max_length = max(max(len(loss_array) for loss_array in train_losses),
                 max(len(loss_array) for loss_array in val_losses))

# Pad each loss array in train_losses and val_losses to max_length with np.nan
train_losses_padded = [np.pad(loss_array, (0, max_length - len(loss_array)), constant_values=np.nan) for loss_array in train_losses]
val_losses_padded = [np.pad(loss_array, (0, max_length - len(loss_array)), constant_values=np.nan) for loss_array in val_losses]

# Calculate the mean and standard deviation
train_mean = np.nanmean(train_losses_padded, axis=0)
val_mean = np.nanmean(val_losses_padded, axis=0)
train_std = np.nanstd(train_losses_padded, axis=0)
val_std = np.nanstd(val_losses_padded, axis=0)
epochs = np.arange(max_length)

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_mean, label='Training Loss')
plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2, label='±1 Standard Deviation')
plt.plot(epochs, val_mean, label='Validation Loss')
plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2, label='±1 Standard Deviation')
plt.ylabel('Loss')
plt.xlabel('Epoch')
ax = plt.gca()  # Get the current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()

# Save the plot to a file
file_path = os.path.join('./trained_models/MLP_reward_models', 'EnsMLP_LossCurve.png')
plt.savefig(file_path, bbox_inches='tight')  # bbox_inches='tight' is used to fit the plot neatly
# plt.show()


# In[9]:

############################### Plot test results ###############################
all_Y_values = [[] for _ in range(len(df.iloc[dm.test_idx]))]  # List of lists to store predictions for each sequence

# Scores Test Sequences for Models
for i in range(num_models):
    model = MLP(learning_rate, batch_size, epochs, slen)
    model_path = f'./trained_models/MLP_reward_models/best_model_v{i}.ckpt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    test_data_frame = df.iloc[list(dm.test_idx)].copy()  # Use the test index from DataModule

    # Predict a single score per sequence
    Y = [model.predict(j).item() for j in test_data_frame['Sequence']]

    # Store predictions for each sequence
    for j, score in enumerate(Y):
        all_Y_values[j].append(score)

# Calculate median, mean, and variance for each sequence
medians = [np.median(scores) for scores in all_Y_values]
means = [np.mean(scores) for scores in all_Y_values]
variances = [np.var(scores) for scores in all_Y_values]


# In[10]:


# Calculating metrics
actual_scores = test_data_frame['log_mean'].tolist()
mse = metrics.mean_squared_error(actual_scores, medians)
r = np.corrcoef(actual_scores, medians)[0][1]
rho, _ = spearmanr(actual_scores, medians)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(actual_scores, medians, color='red', s=5)
plt.plot([min(min(actual_scores),min(medians)), max(max(actual_scores),max(medians))], [min(min(actual_scores),min(medians)), max(max(actual_scores),max(medians))], color='black')  # Diagonal line for reference
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Model Test Results")
# plt.text(3.0, 4.1, f"MSE = {mse:.5f}")
# plt.text(3.0, 4.05, f"R = {r :.3f}")
# plt.text(3.0, 4.0, f"Rho = {rho :.3f}")
ax = plt.gca()  # Get the current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the plot
plot_file_path = './trained_models/MLP_reward_models/EnsMLP_Test_Results.png'
plt.savefig(plot_file_path)

# Save the metrics
metrics_data = [['MSE', 'Pearson R', "Spearman's Rho"], [mse, r, rho]]
csv_file_path = plot_file_path.replace('.png', '.csv')
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(metrics_data)


# In[ ]:




