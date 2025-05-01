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
from torch_ema import ExponentialMovingAverage

# load scripts to load ESM2
from transformers import AutoModelForMaskedLM, AutoTokenizer
from functions import (count_mutations, log_likelihood)
from DPO_EMS2 import (SeqFcnDataset, ProtDataModule, finetuning_ESM2_with_DPO)

# load data
data = 'DPO_CreiLOV_DMS_dataset_w_ESM2_log_likelihood'
df = pd.read_pickle(f"{data}.pkl")
print(f'Using dataset from {data}')
# df.head()

######################################## Hyperparameters that can be altered ########################################

# load ESM2 model
huggingface_identifier ='esm2_t33_650M_UR50D' # esm2_t6_8M_UR50D # esm2_t12_35M_UR50D # esm2_t30_150M_UR50D # esm2_t33_650M_UR50D
ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")
model_identifier = huggingface_identifier
token_format = 'ESM2'
using_sft_model = True

if using_sft_model:
    # load SFT ESM2 model
    sft_logger_version = 6
    sft_model_path = f'../RLXF_SFT_from_pretrained_ESM2_GPU/logs/SFT_pretrained_ESM2_models/version_{sft_logger_version}/SFT_{model_identifier}_v{sft_logger_version}.pt'
    state_dict = torch.load(sft_model_path)
    ESM2.load_state_dict(state_dict)
    print(f'training {huggingface_identifier} with weights from {sft_model_path}')
else:
    print(f'training pretrained {huggingface_identifier}')

# Define hyperparameters

# Model training hyperparameters
num_unfrozen_layers = 27
num_layers_unfreeze_each_epoch = 69
max_num_layers_unfreeze_each_epoch = 82

# Learning hyperparameters
epochs = 30 # 5
patience = 4
seed = 3
using_EMA = 1
decay = 0.8

# DPO hyperparameters from ProteinDPO (Widatalla1, et al., 2024 preprint)
learning_rate = 1e-7
batch_size = 16 # 5
beta = 0.1 # 0.01
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_eps = 1e-8
WD = 0.1

# Data hyperparameters
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
slen = len(WT) # length of protein

# saving parameters
filepath = f'DPO_{model_identifier}_with_{data}'
models_saved = 1

# Determine if we're running on a GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Determine if we're running on a GPU
if device == "cuda":
    # Make models reproducible on GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # Set the PYTHONHASHSEED environment variable to the chosen seed to make hash-based operations predictable
    np.random.seed(seed) # Set NumPy's random seed to ensure reproducibility of operations using NumPy's random number generator
    random.seed(seed) # Set Python's built-in random module's seed to ensure reproducibility of random operations using Python's random functions
    np.random.seed(seed)
    torch.manual_seed(seed) # Set the seed for generating random numbers in PyTorch to ensure reproducibility on the CPU
    torch.cuda.manual_seed(seed) # Set the seed for generating random numbers in PyTorch to ensure reproducibility on the GPU
    torch.cuda.manual_seed_all(seed) # Ensure reproducibility for all GPUs by setting the seed for generating random numbers for all CUDA devices
    torch.backends.cudnn.deterministic = True # Force cuDNN to use only deterministic convolutional algorithms (can slow down computations but guarantees reproducibility)
    torch.backends.cudnn.benchmark = False # Prevent cuDnn from using any algorithms that are nondeterministic
    torch.set_float32_matmul_precision('medium')
    print('Training model on GPU')
else:
    # fix random seeds for reproducibility on CPU
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print('Training model on CPU')
    
######################################## Hyperparameters that can be altered ########################################


# In[ ]:


# setup datamodule
dm = ProtDataModule(df, batch_size, seed) # Use same splits for each model


# In[ ]:


# initilize pytorch model
logger = CSVLogger('logs', name=f"{filepath}") # logger is a class instance that stores performance data to a csv after each epoch
logger_version = logger.version
model = finetuning_ESM2_with_DPO(
                ESM2, huggingface_identifier, tokenizer, num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch,
                 epochs, batch_size, seed, patience,
                 beta, adam_beta_1, adam_beta_2, adam_eps, WD,
                 learning_rate,
                 WT, slen,
                 using_EMA, decay,
                 data, logger_version,
                 using_sft_model
                )


# In[ ]:


# setup model checkpointing
checkpoint_callback = ModelCheckpoint(
        dirpath=f"./logs/{filepath}/",
        filename=f"{filepath}",
        monitor="DPO_loss", # ! Change this
        mode="min",
        save_top_k=models_saved)
early_stopping = EarlyStopping(monitor="DPO_loss", patience=patience, mode="min") # ! Change this


# In[ ]:


# Dynamically set up Trainer based on available device
trainer = pl.Trainer(
    logger=logger,
    max_epochs=epochs,
    callbacks=[checkpoint_callback, early_stopping],
    enable_progress_bar=True,
    accelerator=device,  # Automatically chooses between "cpu" and "gpu"
    devices=1 if device == "cuda" else None,  # Use 1 GPU if available, else default to CPU
    deterministic=True  # Ensure reproducibility
)
try:
    trainer.fit(model, dm)
except Exception as e:
    print(f"Training stopped due to an error: {e}")


# In[ ]:


# Save the model
non_ema_path = f'./logs/{filepath}/version_{logger.version}/multitask_ESM2.pt'
ema_path = f'./logs/{filepath}/version_{logger.version}/multitask_ESM2_w_EMA.pt'
model.save_model(non_ema_path, ema_path)


# In[ ]:


################################################################################################################################################

# make learning curves
version = logger.version  # Replace `logger.version` with the specific version number if needed
train_losses = []
# val_losses = []

# Load the metrics for the specified version
try:
    # Read metrics.csv for the specified version
    pt_metrics = pd.read_csv(f'./logs/{filepath}/version_{version}/metrics.csv')
    
    # Extract training and validation losses
    train = pt_metrics[~pt_metrics.DPO_loss_epoch.isna()]
    # val = pt_metrics[~pt_metrics.val_reg_loss.isna()]
    train_losses = train.DPO_loss_epoch.values
    # val_losses = val.val_reg_loss.values
except FileNotFoundError:
    print(f"Metrics file for version {version} not found.")
    train_losses = []
    # val_losses = []

# Check if losses are available
if len(train_losses) > 0: # and len(val_losses) > 0:
    # Ensure losses have the same length by padding if necessary
    max_length = len(train_losses) # max(  ) # , len(val_losses))
    train_losses = np.pad(train_losses, (0, max_length - len(train_losses)), 'constant', constant_values=np.nan)
    # val_losses = np.pad(val_losses, (0, max_length - len(val_losses)), 'constant', constant_values=np.nan)

    # Compute epochs
    epochs = np.arange(1, max_length + 1)

    # Plot the loss curves
    plt.plot(epochs, train_losses, label='training loss')
    # plt.plot(epochs, val_losses, label='validation loss')
    plt.title('Loss vs. Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Save the loss curves
    file_path_svg = os.path.join(f'./logs/{filepath}/version_{logger.version}', 'Loss_Curves.svg')
    file_path_png = os.path.join(f'./logs/{filepath}/version_{logger.version}', 'Loss_Curves.png')
    plt.savefig(file_path_svg)
    plt.savefig(file_path_png)
    
    print(f"Loss curves saved to {file_path_svg} and {file_path_png}")
else:
    print("No loss data found for this model version.")


# In[ ]:





# In[ ]:




