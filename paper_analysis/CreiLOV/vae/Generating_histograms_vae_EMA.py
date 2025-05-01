
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

# Define amino acid dictionary for tokenization, define WT for length of context window
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
WT = "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA" # CreiLOV
Best_Single_Mutant = "MAGLRHSFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
slen = len(WT)

# Make designs from both models
num_designs = 7500
version = 2
seed = 2459
orig_scale = 0.9111111111111111
# final_rl_scale = 1.8953457515384882
final_rl_scale = orig_scale
num_EnsMLPs = 100

# load models
rl_updated_vae = ConvVAE()
model_path = f'rl_updated_vae_EMA.pt' # load rl_updated_vae
state_dict = torch.load(model_path)
rl_updated_vae.load_state_dict(state_dict)
rl_updated_vae.eval()

# Load pre-trained VAE
checkpoint_path = "./Best_ConvVAE.ckpt"
vae_fixed_model = load_vae_model(checkpoint_path)
for param in vae_fixed_model.parameters():
    param.requires_grad = False

# Load reward models
reward_models = []
for i in range(num_EnsMLPs):
    model_name = f"best_model_v{i}.ckpt"
    checkpoint_path = f"./MLP_Reward_Models/{model_name}"
    reward_model = load_reward_model(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_models.append(reward_model)

mutant_representations, mutant_metrics, mutant_designs  = generate_and_evaluate_mutants(seed, rl_updated_vae, WT, AAs, final_rl_scale, num_designs)
orig_representations, orig_metrics, orig_designs  = generate_and_evaluate_mutants(seed, vae_fixed_model, WT, AAs, orig_scale, num_designs)
# Original scale: 0.9111111111111111

adjusted_mutant_designs = adjust_designs(mutant_designs)
adjusted_orig_designs = adjust_designs(orig_designs)
# print(adjusted_mutant_designs)

scores = convert_and_score_sequences(adjusted_mutant_designs, reward_models)
orig_scores = convert_and_score_sequences(adjusted_orig_designs, reward_models)
# print(scores)

# Save data for mutant designs
EMA = True
save_metrics_to_csv(version, mutant_metrics,EMA)  # Save metrics to a separate file
save_sorted_designs_to_csv(version, WT, adjusted_mutant_designs, scores,EMA)  # Save sorted designs and scores to another file

plt.xlabel('Design Hamming Distances from CreiLOV')
plt.ylabel('Probability')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
annotation_str = f'Sequences Decoded: {num_designs}\nUnique Sequences: {mutant_metrics["number_of_unique_sequences"]}\nUnique Positions: {mutant_metrics["position_diversity"]}\nUnique Mutations: {mutant_metrics["mutation_diversity"]}'
plt.annotate(annotation_str, xy=(0.65, 0.6), xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round", fc="w"))
annotation_str = f'Sequences Decoded: {num_designs}\nUnique Sequences: {orig_metrics["number_of_unique_sequences"]}\nUnique Positions: {orig_metrics["position_diversity"]}\nUnique Mutations: {orig_metrics["mutation_diversity"]}'
plt.annotate(annotation_str, xy=(0.65, 0.3), xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round", fc="w"))
# Show

plt.savefig(f'./logs/Best_aligned_vae/version_{version}/data_set_metrics.png')


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Constants for the mean and standard deviation
log_mean_wt_score = 4.094413241

# Plot histograms
sns.histplot(orig_scores, bins=25, alpha=0.3, color='grey', edgecolor='black', stat='density', ax=ax1, label='Pre-trained VAE')
sns.histplot(scores, bins=25, alpha=0.6, color='blue', edgecolor='black', stat='density', ax=ax1, label='RLXF-updated VAE')
ax1.set_xlabel('Predicted Fluorescence', fontsize=12)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_xlim(min(min(orig_scores)-0.05, min(scores)-0.05), max(max(orig_scores) + 0.05, max(scores) + 0.05))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.axvline(log_mean_wt_score, color='orange', linestyle='--', linewidth=3)
ax1.axvspan(min(min(orig_scores)-0.05, min(scores)-0.05), log_mean_wt_score, color='red', alpha=0.1, zorder=-1)
ax1.axvspan(log_mean_wt_score, max(max(orig_scores) + 0.05, max(scores) + 0.05), color='green', alpha=0.1, zorder=-1)
ax1.legend()

# Plot the cumulative density plot on the second subplot
sns.ecdfplot(orig_scores, stat="proportion", complementary=True, ax=ax2, color="grey", linestyle='-')
sns.ecdfplot(scores, stat="proportion", complementary=True, ax=ax2, color="blue", linestyle='-')
ax2.set_xlabel('Predicted Fluorescence', fontsize=12)
ax2.set_ylabel('Cumulative Density', fontsize=12)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axvline(log_mean_wt_score, color='orange', linestyle='--', linewidth=3)
ax2.axvspan(min(min(orig_scores)-0.05, min(scores)-0.05), log_mean_wt_score, color='red', alpha=0.1, label='Less than WT Fluorescence', zorder=-1)
ax2.axvspan(log_mean_wt_score, max(max(orig_scores) + 0.05, max(scores) + 0.05), color='green', alpha=0.1, label='Better than WT Fluorescence', zorder=-1)
less_wt_patch = mpatches.Patch(color='red', alpha=0.8, label='Less than WT Log Fluorescence')
wt_line = mpatches.Patch(color='orange', alpha=0.8, label='Mean WT Log Fluorescence')
better_wt_patch = mpatches.Patch(color='green', alpha=0.8, label='Greater than WT Log Fluorescence')
legend = ax2.legend(handles=[less_wt_patch, wt_line, better_wt_patch], frameon=True, edgecolor='black')
plt.setp(legend.get_texts(), color='black', fontsize=10)
plt.setp(legend.get_frame(), facecolor='white')

plt.tight_layout()

# Save the plot
plt.savefig(f'./logs/Best_aligned_vae/version_{version}/design_scores_EMA.svg')
plt.show()



