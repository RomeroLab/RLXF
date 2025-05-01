# Import packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from collections import OrderedDict
from torchtext import vocab # This package can give problems sometimes, it may be necessary to downgrade to a specific version
import seaborn as sns
import random
from random import choice
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics
import os
import pickle
from functions import (load_reward_model, identify_mutations_and_count, generate_df, generate_and_evaluate_mutants, generate_and_evaluate_mutants_max_sampling,
    mutate_sequences_after_training, mutate_sequences_after_training_esm2_max_sampling, get_sft_version_file)
from dataloading_RLXF_ESM2 import (ProtDataModuleESM2, ProtRepDatasetESM2)
from PPO_ESM2_650M_with_model_saving_DDP import RLXF_PPO_ESM2
from transformers import AutoModelForMaskedLM, AutoTokenizer
from MLP import MLP
import itertools
import copy
import warnings
import optuna
import logging
import sys
from optuna.exceptions import TrialPruned
from pytorch_lightning.callbacks import Callback
# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig
# from esm.sdk.forge import ESM3ForgeInferenceClient


# Define amino acid dictionary for tokenization, define WT for length of context window
AAs = 'ACDEFGHIKLMNPQRSTVWY' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
sequence_length = len(WT)

# Parameters
num_EnsMLPs = 100  # We have 100 reward models
num_designs = 1000
seed = 7028

# Load reward models
reward_models = []
for i in range(num_EnsMLPs):
    model_name = f"best_model_v{i}.ckpt"
    checkpoint_path = f"./MLP_Reward_Models/{model_name}"
    reward_model = load_reward_model(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_models.append(reward_model)

# ################################################################################################################

# # Load pretrained model
# model_identifier ='esmc-6b-2024-12' # esmc_600m
# ESMC_model = ESM3ForgeInferenceClient(model=model_identifier, url="https://forge.evolutionaryscale.ai", token="7cebzsdq955rf3p2LlRHsz")

# # Generate and evaluate 1000 designs with 5 mutants
# fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
#                                                                     num_muts=5,
#                                                                     WT=WT,
#                                                                     reward_models=reward_models,
#                                                                     model=ESMC_model,
#                                                                     seed=seed,
#                                                                     model_identifier=model_identifier)

# print(f"Status: finished generating sequences with {model_identifier}")

# # Save mutants from ESMC
# base_path = f'./logs/'
# np.save(base_path + f'fixed_scores_{model_identifier}.npy', fixed_scores_np)
# with open(base_path + f'fixed_mutated_seqs_{model_identifier}.txt', 'w') as file:
#     for seq in fixed_mutated_seqs:
#         file.write(seq + '\n')

# ################################################################################################################

# # Load pretrained model
# model_identifier ='esmc_300m' # esmc_600m
# ESMC_model = ESMC.from_pretrained(model_identifier)

# # Generate and evaluate 1000 designs with 5 mutants
# fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
#                                                                     num_muts=5,
#                                                                     WT=WT,
#                                                                     reward_models=reward_models,
#                                                                     model=ESMC_model,
#                                                                     seed=seed,
#                                                                     model_identifier=model_identifier)

# print(f"Status: finished generating sequences with {model_identifier}")

# # Save mutants from ESMC
# base_path = f'./logs/'
# np.save(base_path + f'fixed_scores_{model_identifier}.npy', fixed_scores_np)
# with open(base_path + f'fixed_mutated_seqs_{model_identifier}.txt', 'w') as file:
#     for seq in fixed_mutated_seqs:
#         file.write(seq + '\n')

# ################################################################################################################

# # Load pretrained model
# model_identifier ='esmc_600m' # esmc_600m
# ESMC_model = ESMC.from_pretrained(model_identifier)

# # Generate and evaluate 1000 designs with 5 mutants
# fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
#                                                                     num_muts=5,
#                                                                     WT=WT,
#                                                                     reward_models=reward_models,
#                                                                     model=ESMC_model,
#                                                                     seed=seed,
#                                                                     model_identifier=model_identifier)

# print(f"Status: finished generating sequences with {model_identifier}")

# # Save mutants from ESMC
# base_path = f'./logs/'
# np.save(base_path + f'fixed_scores_{model_identifier}.npy', fixed_scores_np)
# with open(base_path + f'fixed_mutated_seqs_{model_identifier}.txt', 'w') as file:
#     for seq in fixed_mutated_seqs:
#         file.write(seq + '\n')

# ################################################################################################################

# # Load pretrained model
# huggingface_identifier ='esm2_t6_8M_UR50D' # esm2_t36_3B_UR50D # esm2_t33_650M_UR50D # esm2_t30_150M_UR50D # esm2_t12_35M_UR50D  # esm2_t6_8M_UR50D
# fixed_model_8M = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
# model_identifier = huggingface_identifier
# tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")

# # Generate and evaluate 1000 designs with 5 mutants
# fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
#                                                                     num_muts=5,
#                                                                     WT=WT,
#                                                                     reward_models=reward_models,
#                                                                     model=fixed_model_8M,
#                                                                     seed=seed)

# print(f"Status: finished generating sequences with {model_identifier}")

# # Save mutants from ESM2
# base_path = f'./logs/'
# np.save(base_path + f'fixed_scores_{model_identifier}.npy', fixed_scores_np)
# with open(base_path + f'fixed_mutated_seqs_{model_identifier}.txt', 'w') as file:
#     for seq in fixed_mutated_seqs:
#         file.write(seq + '\n')

# ################################################################################################################

# # Load pretrained model
# huggingface_identifier ='esm2_t12_35M_UR50D' # esm2_t36_3B_UR50D # esm2_t33_650M_UR50D # esm2_t30_150M_UR50D # esm2_t12_35M_UR50D  # esm2_t6_8M_UR50D
# fixed_model_8M = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
# model_identifier = huggingface_identifier
# tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")

# # Generate and evaluate 1000 designs with 5 mutants
# fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
#                                                                     num_muts=5,
#                                                                     WT=WT,
#                                                                     reward_models=reward_models,
#                                                                     model=fixed_model_8M,
#                                                                     seed=seed)

# print(f"Status: finished generating sequences with {model_identifier}")

# # Save mutants from ESM2
# base_path = f'./logs/'
# np.save(base_path + f'fixed_scores_{model_identifier}.npy', fixed_scores_np)
# with open(base_path + f'fixed_mutated_seqs_{model_identifier}.txt', 'w') as file:
#     for seq in fixed_mutated_seqs:
#         file.write(seq + '\n')

# ################################################################################################################

# # Load pretrained model
# huggingface_identifier ='esm2_t30_150M_UR50D' # esm2_t36_3B_UR50D # esm2_t33_650M_UR50D # esm2_t30_150M_UR50D # esm2_t12_35M_UR50D  # esm2_t6_8M_UR50D
# fixed_model_8M = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
# model_identifier = huggingface_identifier
# tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")

# # Generate and evaluate 1000 designs with 5 mutants
# fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
#                                                                     num_muts=5,
#                                                                     WT=WT,
#                                                                     reward_models=reward_models,
#                                                                     model=fixed_model_8M,
#                                                                     seed=seed)

# print(f"Status: finished generating sequences with {model_identifier}")

# # Save mutants from ESM2
# base_path = f'./logs/'
# np.save(base_path + f'fixed_scores_{model_identifier}.npy', fixed_scores_np)
# with open(base_path + f'fixed_mutated_seqs_{model_identifier}.txt', 'w') as file:
#     for seq in fixed_mutated_seqs:
#         file.write(seq + '\n')

# ################################################################################################################

# # Load pretrained model
# huggingface_identifier ='esm2_t33_650M_UR50D' # esm2_t36_3B_UR50D # esm2_t33_650M_UR50D # esm2_t30_150M_UR50D # esm2_t12_35M_UR50D  # esm2_t6_8M_UR50D
# fixed_model_8M = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
# model_identifier = huggingface_identifier
# tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")

# # Generate and evaluate 1000 designs with 5 mutants
# fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
#                                                                     num_muts=5,
#                                                                     WT=WT,
#                                                                     reward_models=reward_models,
#                                                                     model=fixed_model_8M,
#                                                                     seed=seed)

# print(f"Status: finished generating sequences with {model_identifier}")

# # Save mutants from ESM2
# base_path = f'./logs/'
# np.save(base_path + f'fixed_scores_{model_identifier}.npy', fixed_scores_np)
# with open(base_path + f'fixed_mutated_seqs_{model_identifier}.txt', 'w') as file:
#     for seq in fixed_mutated_seqs:
#         file.write(seq + '\n')

################################################################################################################

# Load pretrained model
huggingface_identifier ='esm2_t33_650M_UR50D'
sft_model_650M = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
state_dict = torch.load(f'../RLXF_SFT_from_pretrained_ESM2_GPU/logs/SFT_pretrained_ESM2_models/version_6/SFT_esm2_t33_650M_UR50D_v6.pt')
sft_model_650M.load_state_dict(state_dict)

model_identifier = f'sft_{huggingface_identifier}'
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")

# Generate and evaluate 1000 designs with 5 mutants
fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
                                                                    num_muts=5,
                                                                    WT=WT,
                                                                    reward_models=reward_models,
                                                                    model=sft_model_650M,
                                                                    seed=seed)

print(f"Status: finished generating sequences with {model_identifier}")

# Save mutants from ESM2
base_path = f'./logs/'
np.save(base_path + f'fixed_scores_{model_identifier}.npy', fixed_scores_np)
with open(base_path + f'fixed_mutated_seqs_{model_identifier}.txt', 'w') as file:
    for seq in fixed_mutated_seqs:
        file.write(seq + '\n')

################################################################################################################

# Load pretrained model
huggingface_identifier ='esm2_t33_650M_UR50D'
rl_model_650M = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
state_dict = torch.load(f'./logs/Aligning_SFT_ESM2s_wpPPO/version_12/ema_aligned_esm2_t33_650M_UR50D_v12_ep1.pt')
rl_model_650M.load_state_dict(state_dict)
model_identifier = f'rl_{huggingface_identifier}'
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")

# Generate and evaluate 1000 designs with 5 mutants
fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
                                                                    num_muts=5,
                                                                    WT=WT,
                                                                    reward_models=reward_models,
                                                                    model=rl_model_650M,
                                                                    seed=seed)

print(f"Status: finished generating sequences with {model_identifier}")

# Save mutants from ESM2
base_path = f'./logs/'
np.save(base_path + f'fixed_scores_{model_identifier}.npy', fixed_scores_np)
with open(base_path + f'fixed_mutated_seqs_{model_identifier}.txt', 'w') as file:
    for seq in fixed_mutated_seqs:
        file.write(seq + '\n')

# ################################################################################################################

# # Load pretrained model
# huggingface_identifier ='esm2_t36_3B_UR50D' # esm2_t36_3B_UR50D # esm2_t33_650M_UR50D # esm2_t30_150M_UR50D # esm2_t12_35M_UR50D  # esm2_t6_8M_UR50D
# fixed_model_8M = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
# model_identifier = huggingface_identifier
# tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")

# # Generate and evaluate 1000 designs with 5 mutants
# fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
#                                                                     num_muts=5,
#                                                                     WT=WT,
#                                                                     reward_models=reward_models,
#                                                                     model=fixed_model_8M,
#                                                                     seed=seed)

# print(f"Status: finished generating sequences with {model_identifier}")

# # Save mutants from ESM2
# base_path = f'./logs/'
# np.save(base_path + f'fixed_scores_{model_identifier}.npy', fixed_scores_np)
# with open(base_path + f'fixed_mutated_seqs_{model_identifier}.txt', 'w') as file:
#     for seq in fixed_mutated_seqs:
#         file.write(seq + '\n')

# ################################################################################################################

# Create histogram
model_identifiers = ['esm2_t36_3B_UR50D', 'esm2_t33_650M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t6_8M_UR50D', 'esmc_300m', 'esmc_600m', 'esmc-6b-2024-12', 'sft_esm2_t33_650M_UR50D', 'rl_esm2_t33_650M_UR50D']
all_fixed_scores = {}  # Dictionary to store scores for each model

for model_identifier in model_identifiers:
    # Load mutants and scores
    fixed_scores_np = np.load(f'./logs/fixed_scores_{model_identifier}.npy')
    fixed_mutated_seqs = []
    with open(f'./logs/fixed_mutated_seqs_{model_identifier}.txt', 'r') as file:
        fixed_mutated_seqs = file.read().splitlines()
    
    # Store scores in dictionary
    all_fixed_scores[model_identifier] = np.median(fixed_scores_np, axis=0)
    
    # Save to CSV
    df_fixed = generate_df(fixed_mutated_seqs, np.median(fixed_scores_np, axis=0))
    df_fixed.to_csv(f'./logs/fixed_mutated_designs_scores_mutations_version_{model_identifier}.csv', index=False)

# Plot histogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Constants for the mean and standard deviation
predicted_wt_score = 4.1498 # this is predicted WT score # mean log exp score: 4.094413241

# Plot histograms and cumulative density for all models
colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00']  # Colors for models
for model_identifier, color in zip(model_identifiers, colors):
    sns.kdeplot(
        all_fixed_scores[model_identifier],
        color=color,
        ax=ax1,
        label=model_identifier,
    )
    sns.ecdfplot(
        all_fixed_scores[model_identifier],
        stat="proportion",
        complementary=True,
        ax=ax2,
        color=color,
        linestyle='-',
        label=model_identifier,
    )

# Plot histograms for the models
ax1.set_xlabel('Predicted Fluorescence', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.axvline(predicted_wt_score, color='orange', linestyle='--', linewidth=3, label='Predicted CreiLOV Log Fluorescence')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend()

# Plot the cumulative density plot on the second subplot for all models
ax2.set_xlabel('Predicted Fluorescence', fontsize=12)
ax2.set_ylabel('Cumulative Density', fontsize=12)
ax2.axvline(predicted_wt_score, color='orange', linestyle='--', linewidth=3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('./logs/design_scores_all_models.svg')
plt.savefig('./logs/design_scores_all_models.png') 






