#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Import packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from collections import OrderedDict
from torchtext import vocab # This package can give problems sometimes, it may be necessary to downgrade to a specific version
import seaborn as sns
import random
from random import choice
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from sklearn import metrics
import os
import pickle
from transformers import AutoModelForMaskedLM, AutoTokenizer
import itertools
import copy
import warnings
import optuna
import logging
import sys
from torch_ema import ExponentialMovingAverage
import gc

# Import helper scripts
from functions import (load_reward_model, hamming_distance, generate_df)
from functions import (mask_sequence, get_logits_for_all_positions, generate_heatmap, get_mutations, generate_high_confidence_mutant_sequences, generate_mutated_sequences, generate_and_evaluate_mutants_p_sampling)
from functions import (load_reward_model, identify_mutations_and_count, generate_df, generate_and_evaluate_mutants, mutate_sequences_after_training, mutate_sequences_after_training_esm2_max_sampling)
from PPO_with_psampling_and_model_saving import RLXF_PPO_ESM2
from MLP import MLP


# Define amino acid dictionary for tokenization, define WT for length of context window
AAs = 'ACDEFGHIKLMNPQRSTVWY' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap

################################################## hyperparameters ##################################################
# Model selections
sft_model_mapping = {
    0: "8M,105",
    1: "8M,25",
    2: "35M,201",
    3: "35M,25",
    4: "150M,25",
    5: "150M,82",
    6: "650M,25",
    7: "650M,82"
}

# ! Change this
# Model size
sft_logger_version = 7
model_identifier ='esm2_t33_650M_UR50D' # esm2_t6_8M_UR50D # esm2_t12_35M_UR50D # esm2_t30_150M_UR50D # esm2_t33_650M_UR50D

tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_identifier}")
sft_model_path = f'../RLXF_SFT_from_pretrained_ESM2_GPU/logs/SFT_pretrained_ESM2_models/version_{sft_logger_version}/SFT_{model_identifier}_v{sft_logger_version}.pt'
sft_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
rl_updated_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
num_reward_models = 100 # We have an ensemble of 100 MLP reward models
# Figure out number of layers in ESM2 model
layer_counter = 0
for idx, (name, param) in enumerate(sft_model.named_parameters()):
    if "contact_head" in name:
        continue  # Skip layers associated with the contact head
    layer_counter += 1
print(f"There are {layer_counter} layers in {model_identifier}.")

# ! Change these
max_num_layers_unfreeze_each_epoch = 82 # The max number of layers in ESM2 (650M) that will be aligned cannot exceed 82 -> We can go to at least 71 with bs = 10 on our GPU's @ Duke
num_unfrozen_layers = 27 # 82 # initial number of layers of ESM2 unlocked

# learning rate 
learning_rate = 0.008656618973037239
lr_mult = 0.8847762860054206
lr_mult_factor = 1
warm_restart = 1 # with warm restart
use_scheduler = 1 # with scheduler

# optimizer hyperparameters
WD = 0.009951801658490985
clip_type = 1 # with gradient clipping
grad_clip_threshold = 6.824466143373183
grad_clip_threshold_factor = 1.2

# training hyperparameters
seed = 2549
batch_size = 1 # Loading WT to dataloader, we generate variant designs each batch so only load WT initially to models
epochs = 2
iterations = 1
num_updates = max(1, int((epochs / 100) * iterations)) # First restart occurs at 10 epochs (backprop will have occured 10*iterations times)

# generating design hyperparameters
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA'
# WT =   'MAGLDHSFVVADATLPDCPLVYASETFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # ! R5D, T7S, G26T
num_sequences = 2 # initial batch size
inc_batch_size = 1 # increasing batch size each epoch until max_batch_size reached
max_batch_size = 10 # max batch size (dependent on GPU memory)
num_mutations = 15
high_conf_threshold = 0.9 # initial probability threshold to be considered high confidence mutation
cum_prob_threshold = 0.22164310879955906 # initial cumulative probability threshold of non-WT resides to be considered candidate position to explore mutating

# model dependent hyperparameters
num_layers_unfreeze_each_epoch = 69 # numbers of layers of ESM2 to unlock each epoch until max_num_layers_unfreeze_each_epoch reached
training_pos_emb = 0 # do not train positional embeddings
epoch_threshold_to_unlock_ESM2 = 1

# important PPO hyperparameters
average_type = 2
average_type_loss = 0
rel_to_WT = 1
epsilon = 0.17377598245568548 # clipping parameter for PPO loss

# total reward hyperparameters
pairwise_hd_aver_factor = 1.0e-06 # weight for pairwise hamming distance between generated designs each epoch
dkl_scale_init = 1e-8 # initial weight for Dkl
dkl_scale = 1e-7 # weight term for Dkl after 1st epoch

# hyparameters regarding model saving
reduce_EMA_impact = 1
decay = 0.8
saving_models_threshold = 1.012 # previous best -0.51171875 # do not save models if at 10 # 1.01812135525 # 4.225/4.1498 = generated design fitness / predicted WT fitness
filepath = 'Aligning_SFT_ESM2s_wpPPO'

################################################## hyperparameters ##################################################

if sft_model_path is not None:
    # Begin PPO with 2 copies of supervised fine-tuned models
    state_dict = torch.load(sft_model_path)
    sft_model.load_state_dict(state_dict)
    rl_updated_model.load_state_dict(state_dict)
    for param in sft_model.parameters():
        param.requires_grad = False
    print(f'Aligning supervised fine-tuned model from {sft_model_path}')
else:
    # Begin PPO with 2 copies of pretrained models
    for param in sft_model.parameters():
        param.requires_grad = False
    print(f'Aligning {model_identifier} model from huggingface')

# Load models
reward_models = []
for i in range(num_reward_models):
    model_name = f"best_model_v{i}.ckpt"
    checkpoint_path = f"./MLP_Reward_Models/{model_name}"
    reward_model = load_reward_model(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_models.append(reward_model)

# Determine if we're training on a GPU or CPU
if torch.cuda.is_available():
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
    accelerator = "gpu"
    num_devices = torch.cuda.device_count()  # Use all available GPUs
    strategy = "ddp" if num_devices > 1 else None  # Use DDP if multiple GPUs
    if num_devices > 1:
        from dataloading_RLXF_ESM2_DDP import (ProtDataModuleESM2, ProtRepDatasetESM2)
    else:
        from dataloading_RLXF_ESM2 import (ProtDataModuleESM2, ProtRepDatasetESM2)
        print('Running on single GPU, using alternative dataloader')
    print(f"Accelerator: {accelerator}, Number of devices: {num_devices}, Strategy: {strategy}")
else:
    # fix random seeds for reproducibility on CPU
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    accelerator = "cpu"
    max_threads = 16
    num_threads = min(os.cpu_count(), max_threads)  # Use all available CPUs up to a maximum of 16
    torch.set_num_threads(num_threads)  # Set the number of threads for PyTorch
    num_devices = 1  # Use the CPU
    strategy = None
    from dataloading_RLXF_ESM2_DDP import (ProtDataModuleESM2, ProtRepDatasetESM2)
    print(f"Accelerator: {accelerator}, Number of threads: {num_threads}, Strategy: {strategy}")

# Define logger for storing model metrics
logger = CSVLogger('logs', name=f"{filepath}")
version = logger.version

# Initialize the RLXF model
dm = ProtDataModuleESM2(WT, batch_size, seed)
model = RLXF_PPO_ESM2(model_identifier, sft_model, rl_updated_model, reward_models, tokenizer, num_reward_models, sft_model_path, # model selections
                num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch, training_pos_emb, # model dependent hyperparameters
                seed, batch_size, epochs, iterations, num_updates, # training hyperparameters
                learning_rate, lr_mult, lr_mult_factor, use_scheduler, warm_restart, # learning rate hyperparameters
                WD, clip_type, grad_clip_threshold, grad_clip_threshold_factor, # optimizer hyperparameters
                WT, num_sequences, inc_batch_size, max_batch_size, num_mutations, high_conf_threshold, cum_prob_threshold, # generating design hyperparameters
                average_type_loss, average_type, rel_to_WT, epsilon, # important PPO hyperparameters
                pairwise_hd_aver_factor, dkl_scale, dkl_scale_init, # total reward hyperparameters
                reduce_EMA_impact, decay, saving_models_threshold, filepath, version, # hyparameters regarding model saving
                epoch_threshold_to_unlock_ESM2
                     )

# Trainer setup in PyTorch Lightning
if strategy == "ddp":
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        precision=16 if accelerator == "gpu" else 32,  # Mixed precision only on GPU
        enable_progress_bar=True,
        log_every_n_steps=1,
        accelerator=accelerator,
        num_nodes=1,
        devices=num_devices,
        strategy=strategy
    )
else:
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        precision=16 if accelerator == "gpu" else 32,  # Mixed precision only on GPU
        enable_progress_bar=True,
        log_every_n_steps=1,
        accelerator=accelerator,
        num_nodes=1,
        devices=num_devices
        )


trainer.fit(model, dm)

############################################################################################################################################################

save_filepath = f'./logs/{filepath}'

############################################################################################################################################################

# Plot metrics
pt_metrics = pd.read_csv(f'{save_filepath}/version_{version}/metrics.csv')
# Define the metrics you want to plot
metrics_to_plot = [
    ['kl_divergence'],
    ['mean_ratio_initial_iter', 'mean_ratio_final_iter'],
    ['median_ratio_initial_iter', 'median_ratio_final_iter'],
    ['ppo_loss_initial_iter', 'ppo_loss_final_iter'],
    ['fitness_advantage'],
    ['rel_WT_fitness'],
    ['pairwise_hd_aver'],
    ['mean_hd_from_CreiLOV'],
    ['total_reward'],
    ['batch_size'],
    ['num_masks'],
    ['max_norm']]
# Calculate the number of rows for subplots, assuming 1 column
num_rows = len(metrics_to_plot)
# Create subplots
fig, axs = plt.subplots(num_rows, 1, figsize=(10, num_rows * 3))  # Adjust the size as needed
# In case there is only one metric, axs won't be an array, so we make it one for consistency
if num_rows == 1:
    axs = [axs]
# Define ratio metrics for which legends will be added
ratio_metrics = {'mean_ratio_initial_iter', 'mean_ratio_final_iter', 'median_ratio_initial_iter', 'median_ratio_final_iter', 'ppo_loss_initial_iter', 'ppo_loss_final_iter'}
# Loop through each group of metrics and create a plot
for i, metric_group in enumerate(metrics_to_plot):
    for metric in metric_group:
        if metric in pt_metrics.columns:
            data = pt_metrics[~pt_metrics[metric].isna()][metric]
            steps = pt_metrics[~pt_metrics[metric].isna()]['step']
            axs[i].plot(steps, data, label=metric.title())
    
    # Check if the current metric group contains any ratio metrics for adding legends
    if any(metric in ratio_metrics for metric in metric_group):
        axs[i].legend()
    axs[i].set_xlabel('Epoch/Step')
    axs[i].set_ylabel(', '.join(metric_group).replace('_initial_iter', '').replace(', mean_ratio_final_iter', '').replace(', median_ratio_final_iter', '').replace(', ppo_loss_final_iter', '').title())
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
# Adjust the layout and display the plot
fig.tight_layout()
# Save figure
plt.savefig(f'{save_filepath}/version_{version}/metrics_vs_steps.svg')
plt.savefig(f'{save_filepath}/version_{version}/metrics_vs_steps.png')
print('saved learning curves from aligned model')

############################################################################################################################################################

# Generate designs
num_designs = 1000
num_muts = 5
high_conf_threshold = 0.9
cum_prob_threshold = 0.25
seed = 7028
ep = 0

############################################################################################################################################################

# Load pretrained models
fixed_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")

# Generate and evaluate 1000 designs with 5 mutants
fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, fixed_model, model_identifier, tokenizer, save_filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
print(f"Status: finished generating sequences with fixed {model_identifier}")

# Save mutants from ESM2
base_path = f'./logs/{filepath}/version_{version}/'
np.save(base_path + f'fixed_{model_identifier}_scores.npy', fixed_scores_np)
with open(base_path + f'fixed_{model_identifier}_mutated_seqs.txt', 'w') as file:
    for seq in fixed_mutated_seqs:
        file.write(seq + '\n')

############################################################################################################################################################

# Load sft model
sft_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
state_dict = torch.load(f'{sft_model_path}')
sft_model.load_state_dict(state_dict)

# Generate and evaluate 1000 designs with 5 mutants from both models
sft_mutated_seqs, sft_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, sft_model, model_identifier, tokenizer, save_filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
print(f"Status: finished generating sequences with sft {model_identifier}")

# Save mutants from ESM2
base_path = f'./logs/{filepath}/version_{version}/'
np.save(base_path + f'sft_{model_identifier}_scores.npy', sft_scores_np)
with open(base_path + f'sft_{model_identifier}_mutated_seqs.txt', 'w') as file:
    for seq in sft_mutated_seqs:
        file.write(seq + '\n')

############################################################################################################################################################

# Load mutants
fixed_scores_np = np.load(f'./logs/{filepath}/version_{version}/fixed_{model_identifier}_scores.npy')
fixed_mutated_seqs = []
with open(f'./logs/{filepath}/version_{version}/fixed_{model_identifier}_mutated_seqs.txt', 'r') as file:
    fixed_mutated_seqs = file.read().splitlines()

sft_scores_np = np.load(f'./logs/{filepath}/version_{version}/sft_{model_identifier}_scores.npy')
sft_mutated_seqs = []
with open(f'./logs/{filepath}/version_{version}/sft_{model_identifier}_mutated_seqs.txt', 'r') as file:
    sft_mutated_seqs = file.read().splitlines()

# Generate DataFrames
df_sft = generate_df(sft_mutated_seqs, np.median(sft_scores_np, axis=0))
df_fixed = generate_df(fixed_mutated_seqs, np.median(fixed_scores_np, axis=0))

# Save to CSV
df_sft.to_csv(f'./logs/{filepath}/version_{version}/{model_identifier}_sft_mutated_designs_scores.csv', index=False)
df_fixed.to_csv(f'./logs/{filepath}/version_{version}/{model_identifier}_fixed_mutated_designs_scores.csv', index=False)

# Load mutants
fixed_scores_np = np.load(f'./logs/{filepath}/version_{version}/fixed_{model_identifier}_scores.npy')
fixed_mutated_seqs = []
with open(f'./logs/{filepath}/version_{version}/fixed_{model_identifier}_mutated_seqs.txt', 'r') as file:
    fixed_mutated_seqs = file.read().splitlines()

sft_scores_np = np.load(f'./logs/{filepath}/version_{version}/sft_{model_identifier}_scores.npy')
sft_mutated_seqs = []
with open(f'./logs/{filepath}/version_{version}/sft_{model_identifier}_mutated_seqs.txt', 'r') as file:
    sft_mutated_seqs = file.read().splitlines()

############################################################################################################################################################
epochs_models_saved = [0,1]

for ep in epochs_models_saved:
    # Load rl models
    rl_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
    state_dict = torch.load(f'./logs/{filepath}/version_{version}/ema_aligned_{model_identifier}_v{version}_ep{ep}.pt')
    rl_model.load_state_dict(state_dict)

    # Generate and evaluate 1000 designs with 5 mutants from both models
    rl_mutated_seqs, rl_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, rl_model, model_identifier, tokenizer, save_filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
    print(f"Status: finished generating sequences with sft {model_identifier}")

    # Save mutants from ESM2
    base_path = f'./logs/{filepath}/version_{version}/'
    np.save(base_path + f'ema_aligned_{model_identifier}_scores.npy', rl_scores_np)
    with open(base_path + f'ema_aligned_{model_identifier}_mutated_seqs.txt', 'w') as file:
        for seq in rl_mutated_seqs:
            file.write(seq + '\n')

    ############################################################################################################################################################

    # Load mutants
    rl_scores_np = np.load(f'./logs/{filepath}/version_{version}/ema_aligned_{model_identifier}_scores.npy')
    rl_mutated_seqs = []
    with open(f'./logs/{filepath}/version_{version}/ema_aligned_{model_identifier}_mutated_seqs.txt', 'r') as file:
        rl_mutated_seqs = file.read().splitlines()

    # Generate DataFrames
    df_rl = generate_df(rl_mutated_seqs, np.median(rl_scores_np, axis=0))

    # Save to CSV
    df_rl.to_csv(f'./logs/{filepath}/version_{version}/ema_aligned_{model_identifier}_mutated_designs_scores_ep{ep}.csv', index=False)

    ###############################################################################################################################################

    # Plot histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Constants for the mean and standard deviation
    log_mean_wt_score = 4.094413241

    # Plot histograms for the models
    sns.histplot(np.median(fixed_scores_np, axis=0), bins=25, alpha=0.4, color='grey', edgecolor='black', stat='density', ax=ax1, label='Pre-trained ESM2')
    sns.histplot(np.median(sft_scores_np, axis=0), bins=25, alpha=0.6, color='orange', edgecolor='black', stat='density', ax=ax1, label='SFT ESM2')
    sns.histplot(np.median(rl_scores_np, axis=0), bins=25, alpha=0.6, color='blue', edgecolor='black', stat='density', ax=ax1, label='Aligned ESM2')
    ax1.set_xlabel('Predicted Fluorescence', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.axvline(log_mean_wt_score, color='orange', linestyle='--', linewidth=3)
    ax1.axvspan(min(min(np.median(fixed_scores_np, axis=0))-0.05, min(np.median(sft_scores_np, axis=0))-0.05), log_mean_wt_score, color='red', alpha=0.1, zorder=-1)
    ax1.axvspan(log_mean_wt_score, max(max(np.median(fixed_scores_np, axis=0)) + 0.05, max(np.median(sft_scores_np, axis=0)) + 0.05), color='green', alpha=0.1, zorder=-1)
    ax1.legend()

    # Plot the cumulative density plot on the second subplot for all models
    sns.ecdfplot(np.median(fixed_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="grey", linestyle='-')
    sns.ecdfplot(np.median(sft_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="orange", linestyle='-')
    sns.ecdfplot(np.median(rl_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="blue", linestyle='-')
    ax2.set_xlabel('Predicted Fluorescence', fontsize=12)
    ax2.set_ylabel('Cumulative Density', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.axvline(log_mean_wt_score, color='orange', linestyle='--', linewidth=3)
    ax2.axvspan(min(min(np.median(fixed_scores_np, axis=0))-0.05, min(np.median(sft_scores_np, axis=0))-0.05), log_mean_wt_score, color='red', alpha=0.1, zorder=-1)
    ax2.axvspan(log_mean_wt_score, max(max(np.median(fixed_scores_np, axis=0)) + 0.05, max(np.median(sft_scores_np, axis=0)) + 0.05), color='green', alpha=0.1, label='Better than WT Fluorescence', zorder=-1)
    less_wt_patch = mpatches.Patch(color='red', alpha=0.8, label='Less than WT Log Fluorescence')
    wt_line = mpatches.Patch(color='orange', alpha=0.8, label='Mean WT Log Fluorescence')
    better_wt_patch = mpatches.Patch(color='green', alpha=0.8, label='Greater than WT Log Fluorescence')
    legend = ax2.legend(handles=[less_wt_patch, wt_line, better_wt_patch], frameon=True, edgecolor='black')
    plt.setp(legend.get_texts(), color='black', fontsize=10)
    plt.setp(legend.get_frame(), facecolor='white')

    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{save_filepath}/version_{version}/ema_{model_identifier}_design_scores_ep{ep}.svg')
    plt.savefig(f'{save_filepath}/version_{version}/ema_{model_identifier}_design_scores_ep{ep}.png')
    print('Saved design histograms')

    ############################################################################################################################################################

    # Load rl models
    rl_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
    state_dict = torch.load(f'./logs/{filepath}/version_{version}/non_ema_aligned_{model_identifier}_v{version}_ep{ep}.pt')
    rl_model.load_state_dict(state_dict)

    # Generate and evaluate 1000 designs with 5 mutants from both models
    rl_mutated_seqs, rl_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, rl_model, model_identifier, tokenizer, save_filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
    print(f"Status: finished generating sequences with sft {model_identifier}")

    # Save mutants from ESM2
    base_path = f'./logs/{filepath}/version_{version}/'
    np.save(base_path + f'non_ema_aligned_{model_identifier}_scores.npy', rl_scores_np)
    with open(base_path + f'non_ema_aligned_{model_identifier}_mutated_seqs.txt', 'w') as file:
        for seq in rl_mutated_seqs:
            file.write(seq + '\n')

    ############################################################################################################################################################

    # Load mutants
    rl_scores_np = np.load(f'./logs/{filepath}/version_{version}/non_ema_aligned_{model_identifier}_scores.npy')
    rl_mutated_seqs = []
    with open(f'./logs/{filepath}/version_{version}/non_ema_aligned_{model_identifier}_mutated_seqs.txt', 'r') as file:
        rl_mutated_seqs = file.read().splitlines()

    # Generate DataFrames
    df_rl = generate_df(rl_mutated_seqs, np.median(rl_scores_np, axis=0))

    # Save to CSV
    df_rl.to_csv(f'./logs/{filepath}/version_{version}/non_ema_aligned_{model_identifier}_mutated_designs_scores_ep{ep}.csv', index=False)

    ###############################################################################################################################################

    # Plot histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Constants for the mean and standard deviation
    log_mean_wt_score = 4.094413241

    # Plot histograms for the models
    sns.histplot(np.median(fixed_scores_np, axis=0), bins=25, alpha=0.4, color='grey', edgecolor='black', stat='density', ax=ax1, label='Pre-trained ESM2')
    sns.histplot(np.median(sft_scores_np, axis=0), bins=25, alpha=0.6, color='orange', edgecolor='black', stat='density', ax=ax1, label='SFT ESM2')
    sns.histplot(np.median(rl_scores_np, axis=0), bins=25, alpha=0.6, color='blue', edgecolor='black', stat='density', ax=ax1, label='Aligned ESM2')
    ax1.set_xlabel('Predicted Fluorescence', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.axvline(log_mean_wt_score, color='orange', linestyle='--', linewidth=3)
    ax1.axvspan(min(min(np.median(fixed_scores_np, axis=0))-0.05, min(np.median(sft_scores_np, axis=0))-0.05), log_mean_wt_score, color='red', alpha=0.1, zorder=-1)
    ax1.axvspan(log_mean_wt_score, max(max(np.median(fixed_scores_np, axis=0)) + 0.05, max(np.median(sft_scores_np, axis=0)) + 0.05), color='green', alpha=0.1, zorder=-1)
    ax1.legend()

    # Plot the cumulative density plot on the second subplot for all models
    sns.ecdfplot(np.median(fixed_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="grey", linestyle='-')
    sns.ecdfplot(np.median(sft_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="orange", linestyle='-')
    sns.ecdfplot(np.median(rl_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="blue", linestyle='-')
    ax2.set_xlabel('Predicted Fluorescence', fontsize=12)
    ax2.set_ylabel('Cumulative Density', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.axvline(log_mean_wt_score, color='orange', linestyle='--', linewidth=3)
    ax2.axvspan(min(min(np.median(fixed_scores_np, axis=0))-0.05, min(np.median(sft_scores_np, axis=0))-0.05), log_mean_wt_score, color='red', alpha=0.1, zorder=-1)
    ax2.axvspan(log_mean_wt_score, max(max(np.median(fixed_scores_np, axis=0)) + 0.05, max(np.median(sft_scores_np, axis=0)) + 0.05), color='green', alpha=0.1, label='Better than WT Fluorescence', zorder=-1)
    less_wt_patch = mpatches.Patch(color='red', alpha=0.8, label='Less than WT Log Fluorescence')
    wt_line = mpatches.Patch(color='orange', alpha=0.8, label='Mean WT Log Fluorescence')
    better_wt_patch = mpatches.Patch(color='green', alpha=0.8, label='Greater than WT Log Fluorescence')
    legend = ax2.legend(handles=[less_wt_patch, wt_line, better_wt_patch], frameon=True, edgecolor='black')
    plt.setp(legend.get_texts(), color='black', fontsize=10)
    plt.setp(legend.get_frame(), facecolor='white')

    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{save_filepath}/version_{version}/non_ema_{model_identifier}_design_scores_ep{ep}.svg')
    plt.savefig(f'{save_filepath}/version_{version}/non_ema_{model_identifier}_design_scores_ep{ep}.png')
    print('Saved design histograms')







