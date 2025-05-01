# Import packages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from collections import OrderedDict
from torchtext import vocab # This package can give problems sometimes, it may be necessary to downgrade to a specific version
import seaborn as sns
import random
from random import choice
from sklearn import metrics
import os
import pickle
from transformers import AutoModelForMaskedLM, AutoTokenizer
from MLP import MLP
import itertools
import copy
import optuna
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics

# Import helper scripts
from functions import (load_reward_model, identify_mutations_and_count, generate_df, generate_and_evaluate_mutants,
    mutate_sequences_after_training, mutate_sequences_after_training_esm2_max_sampling, get_sft_version_file)
from functions import (load_reward_model, hamming_distance, generate_df)
from functions import (mask_sequence, get_logits_for_all_positions, generate_heatmap, get_mutations, generate_high_confidence_mutant_sequences, generate_mutated_sequences, generate_and_evaluate_mutants_p_sampling)

from SFT_ESM2_curated_data_from_SA import (SFT_ESM2, SeqSeqDataset, SFTDataModule)

# Send models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make models reproducible
seed=42
os.environ['PYTHONHASHSEED'] = str(seed) # Set the PYTHONHASHSEED environment variable to the chosen seed to make hash-based operations predictable
np.random.seed(seed) # Set NumPy's random seed to ensure reproducibility of operations using NumPy's random number generator
random.seed(seed) # Set Python's built-in random module's seed to ensure reproducibility of random operations using Python's random functions
torch.manual_seed(seed) # Set the seed for generating random numbers in PyTorch to ensure reproducibility on the CPU
torch.cuda.manual_seed(seed) # Set the seed for generating random numbers in PyTorch to ensure reproducibility on the GPU
torch.cuda.manual_seed_all(seed) # Ensure reproducibility for all GPUs by setting the seed for generating random numbers for all CUDA devices
torch.backends.cudnn.deterministic = True # Force cuDNN to use only deterministic convolutional algorithms (can slow down computations but guarantees reproducibility)
torch.backends.cudnn.benchmark = False # Prevent cuDnn from using any algorithms that are nondeterministic
torch.set_float32_matmul_precision('medium')

# Define amino acid dictionary for tokenization, define WT for length of context window
AAs = 'ACDEFGHIKLMNPQRSTVWY' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
sequence_length = len(WT)

################################################## This variables will be constant during Optuna Sweeps ##################################################
num_EnsMLPs = 100 # We have 100 reward models
dataset = 1
max_num_layers_unfreeze_each_epoch = 82 # ! Will change depending on model size
warm_restart = 1
use_scheduler = 1
reinit_optimizer = 0
training_pos_emb = 0
batch_size = 8

# load reward models
reward_models = []
for i in range(num_EnsMLPs):
    model_name = f"best_model_v{i}.ckpt"
    checkpoint_path = f"/home/nlb51/RLXF_PPO_from_pretrained_ESM2_GPU/MLP_Reward_Models/{model_name}"
    reward_model = load_reward_model(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_models.append(reward_model)
    
# Load ESM2
model_identifier ='esm2_t33_650M_UR50D' # esm2_t6_8M_UR50D # esm2_t12_35M_UR50D # esm2_t30_150M_UR50D # esm2_t33_650M_UR50D
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_identifier}")
ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
ESM2 = ESM2.to(device)

# Figure out number of layers in ESM2 model
layer_counter = 0
for idx, (name, param) in enumerate(ESM2.named_parameters()):
    if "contact_head" in name:
        continue  # Skip layers associated with the contact head
    layer_counter += 1
print(f"There are {layer_counter} layers in {model_identifier}.")

# Define logger for storing model metrics
logger_name = 'SFT_ESM2_650M_with_SA_data'
logger = CSVLogger('logs', name=logger_name, version=None)
version = logger.version # Retrieve the version number from the logger

# Parameters from Optuna
epochs = 1
learning_rate = 0.0051114990195524
use_weights = 1
WD = 0.003506385543831778
grad_clip_threshold = 3
lr_mult_factor = 1.5704059582871683
num_layers_unfreeze_each_epoch = 1
lr_mult = 0.8897135219977205
num_unfrozen_layers = 25 # max_num_layers_unfreeze_each_epoch # ! Will change depending on model size, try 25 for all, 82 for all, try full for 8M and 35M
dataset_version = 5 # 3 # 4 # 5 # 4
random_masking =  0 # 0 # 1 # 1

# Load preference data
if dataset_version == 0:
    df = pd.read_pickle("./SFT_dataset_df_all_mutation_counts.pkl") # load preprocessed CreiLOV data
elif dataset_version == 1:
    df = pd.read_pickle("./SFT_dataset_df_single_mutants.pkl") # load preprocessed CreiLOV data
elif dataset_version == 2:
    df = pd.read_pickle("./SFT_dataset_all_unique_designs_from_SA.pkl") # load preprocessed CreiLOV data
elif dataset_version == 3:
    df = pd.read_pickle("./SFT_dataset_df_1000_unique_designs_from_SA.pkl") # load preprocessed CreiLOV data
elif dataset_version == 4:
    df = pd.read_pickle("./SFT_dataset_df_best_1000_unique_designs_from_SA.pkl") # load preprocessed CreiLOV data
elif dataset_version == 5:
    df = pd.read_pickle("./SFT_dataset_df_max_fitness_unique_designs_from_SA.pkl") # load preprocessed CreiLOV data

############################################################################################################################################################

# Initialize model with hyperparameters
dm = SFTDataModule(df, batch_size, seed)
model = SFT_ESM2(ESM2, reward_models, seed, learning_rate, lr_mult, lr_mult_factor, use_scheduler, warm_restart, reinit_optimizer, WD, grad_clip_threshold, epochs, num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch, training_pos_emb, batch_size, dataset, dataset_version, use_weights, random_masking, model_identifier)

# Train model
trainer = pl.Trainer(logger=logger, max_epochs=epochs, enable_progress_bar=False, log_every_n_steps=1, accelerator = "gpu", devices = 1)
trainer.fit(model,dm)

# Save the sft_updated_esm2 model when training is done, appending the version number to the filename
model.save_sft_updated_esm2(f'./logs/{logger_name}/version_{version}/SFT_{model_identifier}_v{version}.pt')

############################################################################################################################################################

# Plotting metrics
pt_metrics = pd.read_csv(f'./logs/{logger_name}/version_{version}/metrics.csv')

# Define the metrics you want to plot
metrics_to_plot = [['train_loss']]

# Calculate the number of rows for subplots, assuming 1 column
num_rows = len(metrics_to_plot)

# Create subplots
fig, axs = plt.subplots(num_rows, 1, figsize=(10, num_rows * 3))  # Adjust the size as needed

# In case there is only one metric, axs won't be an array, so we make it one for consistency
if num_rows == 1:
    axs = [axs]

# Loop through each group of metrics and create a plot
for i, metric_group in enumerate(metrics_to_plot):
    for metric in metric_group:
        if metric in pt_metrics.columns:
            data = pt_metrics[~pt_metrics[metric].isna()][metric]
            steps = pt_metrics[~pt_metrics[metric].isna()]['step']
            axs[i].plot(steps, data, label=metric.title())

    axs[i].set_xlabel('Epoch/Step')
    axs[i].set_ylabel(', '.join(metric_group).replace('_initial_iter', '').replace(', mean_ratio_final_iter', '').replace(', median_ratio_final_iter', '').replace(', ppo_loss_final_iter', '').title())
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)

# Adjust the layout and display the plot
fig.tight_layout()

# Save figure
plt.savefig(f'./logs/{logger_name}/version_{version}/{model_identifier}_metrics_vs_steps.svg')
plt.savefig(f'./logs/{logger_name}/version_{version}/{model_identifier}_metrics_vs_steps.png')
print('Saved learning curves')

############################################################################################################################################################

with torch.no_grad():

    # Generate designs
    num_designs = 1000
    num_muts = 5
    high_conf_threshold = 0.9
    cum_prob_threshold = 0.01 # 0.01 # 0.25 # 0.25
    seed = 7028
    filepath = f'./logs/{logger_name}'

    # # Load fixed and sft models
    # fixed_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")

    # # Generate and evaluate 1000 designs with 5 mutants
    # fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, fixed_model, model_identifier, tokenizer, filepath, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
    # print(f"Status: finished generating sequences with fixed {model_identifier}")

    # Save mutants from ESM2
    # base_path = f'./logs/{logger_name}/version_{version}/'
    # np.save(base_path + f'fixed_{model_identifier}_scores.npy', fixed_scores_np)
    # with open(base_path + f'fixed_{model_identifier}_mutated_seqs.txt', 'w') as file:
    #     for seq in fixed_mutated_seqs:
    #         file.write(seq + '\n')

    ############################################################################################################################################################

    # Load fixed and sft models
    sft_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
    state_dict = torch.load(f'./logs/{logger_name}/version_{version}/SFT_{model_identifier}_v{version}.pt')
    sft_model.load_state_dict(state_dict)

    # Generate and evaluate 1000 designs with 5 mutants from both models
    sft_mutated_seqs, sft_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, sft_model, model_identifier, tokenizer, filepath, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
    print(f"Status: finished generating sequences with sft {model_identifier}")

    # Save mutants from ESM2
    base_path = f'./logs/{logger_name}/version_{version}/'
    np.save(base_path + f'sft_{model_identifier}_scores.npy', sft_scores_np)
    with open(base_path + f'sft_{model_identifier}_mutated_seqs.txt', 'w') as file:
        for seq in sft_mutated_seqs:
            file.write(seq + '\n')

    ############################################################################################################################################################

    # Load mutants
    fixed_scores_np = np.load(f'./logs/{logger_name}/version_0/fixed_{model_identifier}_scores.npy')
    # fixed_scores_np = np.load(f'./logs/{logger_name}/version_{version}/fixed_{model_identifier}_scores.npy')
    fixed_mutated_seqs = []
    # with open(f'./logs/{logger_name}/version_{version}/fixed_{model_identifier}_mutated_seqs.txt', 'r') as file:
    with open(f'./logs/{logger_name}/version_0/fixed_{model_identifier}_mutated_seqs.txt', 'r') as file:
        fixed_mutated_seqs = file.read().splitlines()

    sft_scores_np = np.load(f'./logs/{logger_name}/version_{version}/sft_{model_identifier}_scores.npy')
    sft_mutated_seqs = []
    with open(f'./logs/{logger_name}/version_{version}/sft_{model_identifier}_mutated_seqs.txt', 'r') as file:
        sft_mutated_seqs = file.read().splitlines()

    # Generate DataFrames
    df_sft = generate_df(sft_mutated_seqs, np.median(sft_scores_np, axis=0))
    df_fixed = generate_df(fixed_mutated_seqs, np.median(fixed_scores_np, axis=0))

    # Save to CSV
    df_sft.to_csv(f'./logs/{logger_name}/version_{version}/{model_identifier}_sft_mutated_designs_scores.csv', index=False)
    df_fixed.to_csv(f'./logs/{logger_name}/version_{version}/{model_identifier}_fixed_mutated_designs_scores.csv', index=False)

    ###############################################################################################################################################

    # Plot histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Constants for the mean and standard deviation
    log_mean_wt_score = 4.094413241

    # Plot histograms for the models
    sns.histplot(np.median(fixed_scores_np, axis=0), bins=25, alpha=0.4, color='grey', edgecolor='black', stat='density', ax=ax1, label='Pre-trained ESM2')
    sns.histplot(np.median(sft_scores_np, axis=0), bins=25, alpha=0.6, color='blue', edgecolor='black', stat='density', ax=ax1, label='Aligned ESM2')
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
    sns.ecdfplot(np.median(sft_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="blue", linestyle='-')
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
    plt.savefig(f'./logs/{logger_name}/version_{version}/{model_identifier}_design_scores.svg')
    plt.savefig(f'./logs/{logger_name}/version_{version}/{model_identifier}_design_scores.png')

    print('Saved design histograms')
















