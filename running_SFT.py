# Import packages
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import seaborn as sns
import random
from random import choice
import os
import pickle
from transformers import AutoModelForMaskedLM, AutoTokenizer
from MLP import MLP
import itertools
import copy
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# import helper scripts
from functions import (generate_df, generate_and_evaluate_mutants_p_sampling)
from SFT_ESM2 import (SFT_ESM2, SFTDataModule)

# Parameters to update
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
sequence_length = len(WT)
num_reward_models = 100

# model parameters
model_identifier ='esm2_t33_650M_UR50D' # esm2_t6_8M_UR50D # esm2_t12_35M_UR50D # esm2_t30_150M_UR50D # esm2_t33_650M_UR50D
max_num_layers_unfreeze_each_epoch = 82 # max number of layers in ESM2 (650M) that will be trained
num_unfrozen_layers = 25 # initial number of layers of ESM2 unlocked
num_layers_unfreeze_each_epoch = 1 # numbers of layers of ESM2 to unlock each epoch until max_num_layers_unfreeze_each_epoch reached

# SFT parameters
seed = 42
batch_size = 8
epochs = 1
random_masking =  0 # adding random masks (1) or not (0) to sequence dataset

# optimizer hyperparameters
learning_rate = 0.0051114990195524
lr_mult_factor = 1.5704059582871683
lr_mult = 0.8897135219977205
WD = 0.003506385543831778
grad_clip_threshold = 3

# parameters for generating designs after alignment
num_designs = 100
num_muts = 5
high_conf_threshold = 0.9
cum_prob_threshold = 0.25
ep = epochs - 1
generation_seed = 7028
predicted_wt_score = 1.1498 # predicted wildtype score as reference for evaluations

# send models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make models reproducible
os.environ['PYTHONHASHSEED'] = str(seed) # Set the PYTHONHASHSEED environment variable to the chosen seed to make hash-based operations predictable
np.random.seed(seed) # Set NumPy's random seed to ensure reproducibility of operations using NumPy's random number generator
random.seed(seed) # Set Python's built-in random module's seed to ensure reproducibility of random operations using Python's random functions
torch.manual_seed(seed) # Set the seed for generating random numbers in PyTorch to ensure reproducibility on the CPU
torch.cuda.manual_seed(seed) # Set the seed for generating random numbers in PyTorch to ensure reproducibility on the GPU
torch.cuda.manual_seed_all(seed) # Ensure reproducibility for all GPUs by setting the seed for generating random numbers for all CUDA devices
torch.backends.cudnn.deterministic = True # Force cuDNN to use only deterministic convolutional algorithms (can slow down computations but guarantees reproducibility)
torch.backends.cudnn.benchmark = False # Prevent cuDnn from using any algorithms that are nondeterministic
torch.set_float32_matmul_precision('medium')

# load ensemble of reward models
models = []
for i in range(num_reward_models):
    model_name = f"reward_model_v{i}.ckpt"
    checkpoint_path = f"./reward_models/{model_name}"
    reward_model = MLP.load_from_checkpoint(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    models.append(reward_model)
    
# load ESM2
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_identifier}")
ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
ESM2 = ESM2.to(device)

# Load preference data
df = pd.read_pickle("./unique_optimized_designs_from_simulated_annealing.csv") # load preprocessed CreiLOV data

############################################################################################################################################################

# SFT ESM2
logger_name = f'SFT_{model_identifier}'
logger = CSVLogger('logs', name=logger_name, version=None)
version = logger.version # Retrieve the version number from the logger
dm = SFTDataModule(df, batch_size, seed)
model = SFT_ESM2(ESM2, reward_models, seed, learning_rate, lr_mult, lr_mult_factor, WD, grad_clip_threshold, epochs, num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch, batch_size, random_masking, model_identifier)
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

    # Load fixed and sft models
    fixed_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")

    # Generate and evaluate 1000 designs with 5 mutants
    fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, fixed_model, model_identifier, tokenizer, save_filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, generation_seed)
    print(f"Status: finished generating sequences with fixed {model_identifier}")

    Save mutants from ESM2
    base_path = f'./logs/{logger_name}/version_{version}/'
    np.save(base_path + f'fixed_{model_identifier}_scores.npy', fixed_scores_np)
    with open(base_path + f'fixed_{model_identifier}_mutated_seqs.txt', 'w') as file:
        for seq in fixed_mutated_seqs:
            file.write(seq + '\n')

    ############################################################################################################################################################

    # Load fixed and sft models
    sft_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
    state_dict = torch.load(f'./logs/{logger_name}/version_{version}/SFT_{model_identifier}_v{version}.pt')
    sft_model.load_state_dict(state_dict)

    # Generate and evaluate 1000 designs with 5 mutants from both models
    sft_mutated_seqs, sft_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, sft_model, model_identifier, tokenizer, save_filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, generation_seed)
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

    # Plot histograms for the models
    sns.histplot(np.median(fixed_scores_np, axis=0), bins=25, alpha=0.4, color='grey', edgecolor='black', stat='density', ax=ax1, label='Pre-trained ESM2')
    sns.histplot(np.median(sft_scores_np, axis=0), bins=25, alpha=0.6, color='yellow', edgecolor='black', stat='density', ax=ax1, label='SFT ESM2')
    ax1.set_xlabel('Predicted Fluorescence', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.axvline(predicted_wt_score, color='orange', linestyle='--', linewidth=3)
    ax1.axvspan(min(min(np.median(fixed_scores_np, axis=0))-0.05, min(np.median(sft_scores_np, axis=0))-0.05), predicted_wt_score, color='red', alpha=0.1, zorder=-1)
    ax1.axvspan(predicted_wt_score, max(max(np.median(fixed_scores_np, axis=0)) + 0.05, max(np.median(sft_scores_np, axis=0)) + 0.05), color='green', alpha=0.1, zorder=-1)
    ax1.legend()

    # Plot the cumulative density plot on the second subplot for all models
    sns.ecdfplot(np.median(fixed_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="grey", linestyle='-')
    sns.ecdfplot(np.median(sft_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="yellow", linestyle='-')
    ax2.set_xlabel('Predicted Fluorescence', fontsize=12)
    ax2.set_ylabel('Cumulative Density', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.axvline(predicted_wt_score, color='orange', linestyle='--', linewidth=3)
    ax2.axvspan(min(min(np.median(fixed_scores_np, axis=0))-0.05, min(np.median(sft_scores_np, axis=0))-0.05), predicted_wt_score, color='red', alpha=0.1, zorder=-1)
    ax2.axvspan(predicted_wt_score, max(max(np.median(fixed_scores_np, axis=0)) + 0.05, max(np.median(sft_scores_np, axis=0)) + 0.05), color='green', alpha=0.1, label='Better than Predicted WT Fluorescence', zorder=-1)
    less_wt_patch = mpatches.Patch(color='red', alpha=0.8, label='Less than Predicted WT Log Fluorescence')
    wt_line = mpatches.Patch(color='orange', alpha=0.8, label='Predicted WT Log Fluorescence')
    better_wt_patch = mpatches.Patch(color='green', alpha=0.8, label='Greater than Predicted WT Log Fluorescence')
    legend = ax2.legend(handles=[less_wt_patch, wt_line, better_wt_patch], frameon=True, edgecolor='black')
    plt.setp(legend.get_texts(), color='black', fontsize=10)
    plt.setp(legend.get_frame(), facecolor='white')
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'./logs/{logger_name}/version_{version}/{model_identifier}_design_scores.svg')
    plt.savefig(f'./logs/{logger_name}/version_{version}/{model_identifier}_design_scores.png')

    print('Saved design histograms')
















