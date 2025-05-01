# Import packages
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
from torchtext import vocab
from pytorch_lightning.loggers import CSVLogger
from random import choice
import seaborn as sns
import random
from random import choice
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics
import torchmetrics
import enum
import csv
import os
import pickle
import math
import pathlib
import itertools
from functions import (load_vae_model, load_reward_model, find_closest_average_hd, generate_and_evaluate_mutants, generate_and_evaluate_mutants_vae_training,
    decoding, Using_VAE, hamming_distance_vae_training, ProtDataModule, ProtRepDataset, adjust_designs, SeqDataset, convert_and_score_sequences,
    save_metrics_to_csv, identify_mutations, save_sorted_designs_to_csv)
from conv_vae_model import ConvVAE
from MLP import MLP
import warnings
from RLXF_PPO_VAE_EMA import RLXF_PPO_vae
from torch_ema import ExponentialMovingAverage
warnings.filterwarnings("ignore", category=UserWarning)

# Send models to device
# torch.set_num_threads(18)

# Make models reproducible
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# Define amino acid dictionary for tokenization, define WT for length of context window
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
WT = "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA" # CreiLOV
Best_Single_Mutant = "MAGLRHSFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
slen = len(WT)

################################################## This variables will be constant during Optuna Sweeps ##################################################
num_EnsMLPs = 100 # We have 100 reward models
target_hd = 5 # We want the vae to generate variants with an average of 5 mutations during PPO
max_batch_size = 64 # max batch size
dkl_scale = 1e-7
dkl_scale_init = 1e-8
warm_restart = 1 # with warm restart
use_scheduler = 1 # with scheduler
clip_type = 1
max_num_layers_unfreeze_each_epoch = 15 # </= 21
dkl_backpropagation = 0

# Choose parameter ranges for Optuna sweep
WD = 2.9919702684397275e-06
average_type = 2
average_type_loss = 0
batch_size = 29
inc_batch_size = 9
epochs = 27
epsilon = 0.18264242707540573
grad_clip_threshold = 2.7782468182686686
grad_clip_threshold_factor = 2
iterations = 4
learning_rate = 0.0006516159286525216
lr_mult = 0.8821115071370329
lr_mult_factor = 0.9717580963668324
num_unfrozen_layers = 6
num_layers_unfreeze_each_epoch = 0
pairwise_hd_aver_factor = 89.82182093473786
sampling_max = 0
rel_to_WT = 1

# Load reward models
reward_models = []
for i in range(num_EnsMLPs):
    model_name = f"best_model_v{i}.ckpt"
    checkpoint_path = f"./MLP_Reward_Models/{model_name}"
    reward_model = load_reward_model(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_models.append(reward_model)

# Load pre-trained VAE
checkpoint_path = "./Best_ConvVAE.ckpt"
vae_fixed_model = load_vae_model(checkpoint_path)
for param in vae_fixed_model.parameters():
    param.requires_grad = False

# Load model to be updated
rl_updated_vae = load_vae_model(checkpoint_path)

# Define logger for storing model metrics
logger_name = 'Best_aligned_vae'
logger = CSVLogger('logs', name=logger_name, version=None)
version = logger.version # Retrieve the version number from the logger

# Initialize model with hyperparameters
dm = ProtDataModule(rl_updated_vae, WT, AAs, batch_size, target_hd, None, 0, version) # New dataset each epoch
model = RLXF_PPO_vae(reward_models, rl_updated_vae, vae_fixed_model, version, seed,
                     learning_rate, lr_mult, lr_mult_factor, use_scheduler, warm_restart,
                     WD, clip_type, grad_clip_threshold, grad_clip_threshold_factor, epsilon,
                     epochs, iterations,
                     dkl_scale, dkl_scale_init, dkl_backpropagation,
                     num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch,
                     batch_size, inc_batch_size, max_batch_size,
                     target_hd,
                     pairwise_hd_aver_factor, sampling_max, rel_to_WT, average_type, average_type_loss,
                     num_EnsMLPs)


# Train model
trainer = pl.Trainer(logger=logger, max_epochs=epochs, enable_progress_bar=False, log_every_n_steps=1)
trainer.fit(model, dm)

# Save the rl_updated_vae model when training is done, appending the version number to the filename
model.save_rl_updated_vae(f'Best_aligned_vae_{version}.pt')

# # Plot learning curves
# pt_metrics = pd.read_csv(f'./logs/Best_aligned_vae/version_{version}/metrics.csv')

# # Define the metrics you want to plot
# metrics_to_plot = [
#     ['kl_divergence'],
#     ['mean_ratio_initial_iter', 'mean_ratio_final_iter'],
#     ['median_ratio_initial_iter', 'median_ratio_final_iter'],
#     ['ppo_loss_initial_iter', 'ppo_loss_final_iter'],
#     ['fitness_advantage'],
#     ['rel_WT_fitness'],
#     ['pairwise_hd_aver'],
#     ['mean_hd_from_CreiLOV'],
#     ['total_reward'],
#     ['batch_size'],
#     ['max_norm']]

# # Calculate the number of rows for subplots, assuming 1 column
# num_rows = len(metrics_to_plot)

# # Create subplots
# fig, axs = plt.subplots(num_rows, 1, figsize=(10, num_rows * 3))  # Adjust the size as needed

# # In case there is only one metric, axs won't be an array, so we make it one for consistency
# if num_rows == 1:
#     axs = [axs]

# # Define ratio metrics for which legends will be added
# ratio_metrics = {'mean_ratio_initial_iter', 'mean_ratio_final_iter', 'median_ratio_initial_iter', 'median_ratio_final_iter','ppo_loss_initial_iter','ppo_loss_final_iter'}

# # Loop through each group of metrics and create a plot
# for i, metric_group in enumerate(metrics_to_plot):
#     for metric in metric_group:
#         if metric in pt_metrics.columns:
#             data = pt_metrics[~pt_metrics[metric].isna()][metric]
#             steps = pt_metrics[~pt_metrics[metric].isna()]['step']
#             axs[i].plot(steps, data, label=metric.title())
    
#     # Check if the current metric group contains any ratio metrics for adding legends
#     if any(metric in ratio_metrics for metric in metric_group):
#         axs[i].legend()

#     axs[i].set_xlabel('Epoch/Step')
#     axs[i].set_ylabel(', '.join(metric_group).replace('_initial_iter', '').replace(', mean_ratio_final_iter', '').replace(', median_ratio_final_iter', '').replace(', ppo_loss_final_iter', '').title())
#     axs[i].spines['top'].set_visible(False)
#     axs[i].spines['right'].set_visible(False)
    
# # Adjust the layout
# fig.tight_layout()

# # Save the plot to a file
# plt.savefig(f'./logs/Best_aligned_vae/version_{version}/metrics_vs_steps.svg')

# # Display the plots
# plt.show()







