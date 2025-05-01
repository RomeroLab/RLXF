#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
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
from matplotlib.colors import LinearSegmentedColormap


# In[5]:


# load scripts to load ESM2 and functions
from transformers import AutoModelForMaskedLM, AutoTokenizer
from functions import (count_mutations, log_likelihood)
from tqdm import tqdm  # For progress tracking

# In[3]:


# define key parameters

# Define amino acid dictionary for tokenization, define WT for length of context window
AAs = 'ACDEFGHIKLMNPQRSTVWY' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
sequence_length = len(WT)


# In[4]:


# load training split of CreiLOC dataset previously used to train reward model that guided PPO into pandas dataframe
df = pd.read_pickle("CreiLOV_4cluster_df.pkl") # load preprocessed CreiLOV data

# add the number of mutations to dataframe to obtain reward model training set
df['Num_Muts'] = df['Mutations'].apply(count_mutations)
df = df[df['Num_Muts'] <= 4]

# Keep relevant columns
df = df[['Sequence', 'log_mean', 'Mutations']]
df.tail()


# In[28]:


# score sequences with ESM2 and add to pandas dataframe

# Load ESM2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_identifier ='esm2_t33_650M_UR50D' # esm2_t6_8M_UR50D # esm2_t12_35M_UR50D # esm2_t30_150M_UR50D # esm2_t33_650M_UR50D
ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}").to(device)
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_identifier}")


# Ensure sequences are in list format
sequences = df["Sequence"].tolist()

# Compute ESM2 scores
df["ESM2_score"] = log_likelihood(sequences, device, ESM2, tokenizer).cpu().numpy()


# In[25]:


# process dataset to contain Sequence (Prompt), ESM2 (log-likelihood), and (log mean fluorescence)
df = df[['Sequence', 'ESM2_score', 'log_mean', 'Mutations']]


# In[26]:


# save dataset to .pkl file
df.to_pickle("DPO_CreiLOV_DMS_dataset_w_ESM2_log_likelihood.pkl")


# In[27]:


df.head()


# In[ ]:




