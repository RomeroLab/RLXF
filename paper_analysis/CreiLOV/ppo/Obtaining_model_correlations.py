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
from functions import (load_vae_model, load_reward_model, identify_mutations_and_count, generate_df, generate_and_evaluate_mutants, mutate_sequences_after_training, mutate_sequences_after_training_esm2_max_sampling)
from dataloading_RLXF_ESM2_DDP import (ProtDataModuleESM2, ProtRepDatasetESM2)
from PPO_with_psampling_and_model_saving import RLXF_PPO_ESM2
from MLP import MLP, ProtDataModule
from conv_vae_model import ConvVAE

##############################################################################################################################

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap

##############################################################################################################################

# Load reward models
sequence_length = len(WT)
num_models = 100
learning_rate = 1e-6 # important to optimize this
batch_size = 128 # typically powers of 2: 32, 64, 128, 256, ...
epochs = 2000 # rounds of training
slen = len(WT) # length of protein
num_models = 100 # number of models in ensemble

models = []
for i in range(num_models):
    model = MLP(learning_rate, batch_size, epochs, slen)
    model_path = f'./MLP_Reward_Models/best_model_v{i}.ckpt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    models.append(model)

# Load pre-trained VAE
checkpoint_path = "./Best_ConvVAE.ckpt" # TODO: check if on server
fixed_vae = load_vae_model(checkpoint_path)
fixed_vae.eval()

# Load aligned VAE
version = 507
aligned_vae = ConvVAE()
model_path = f'./rl_updated_vae_version_{version}.pt' # TODO: check if on server
state_dict = torch.load(model_path)
aligned_vae.load_state_dict(state_dict)
aligned_vae.eval()

# Load pre-trained ESM-2
fixed_ESM2 = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
fixed_ESM2.eval()

# Load SFT ESM-2
sft_version = 1260
sft_ESM2 = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
state_dict = torch.load(f'./SFT_ESM2_650M_with_data_v5_and_random_masking_v0.pt')
sft_ESM2.load_state_dict(state_dict)
sft_ESM2.eval()

# Load aligned ESM-2
rl_version = 30
ep = 158
EMA = True
aligned_ESM2 = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
state_dict = torch.load(f'./logs/Aligning_SFT_ESM2s_wpPPO/version_12/ema_aligned_esm2_t33_650M_UR50D_v12_ep1.pt')
aligned_ESM2.load_state_dict(state_dict)
aligned_ESM2.eval()

# Load reward model test split data
reward_model_df = pd.read_pickle("./CreiLOV_4cluster_df.pkl") # TODO: check if on server
dm = ProtDataModule(reward_model_df,32,'./CreiLOV_EnsMLP_splits_df.pkl') # TODO: check if on server
reward_model_train_df, reward_model_val_df, reward_model_test_df = dm.get_split_dataframes()
# reward_model_test_df.head(5)

# Load SFT preference data
SFT_df = pd.read_pickle("../RLXF_SFT_from_pretrained_ESM2_GPU/SFT_dataset_df_max_fitness_unique_designs_from_SA.pkl") # TODO: check if on server
# SFT_df.head(5)

# Add source to each dataframe
reward_model_train_df["Source"] = "reward_model_train_df"
reward_model_val_df["Source"] = "reward_model_val_df"
reward_model_test_df["Source"] = "reward_model_test_df"
SFT_df["Source"] = "SFT_df"

# Combine all dataframes into a single dataframe
df_combined = pd.concat([reward_model_test_df, SFT_df], ignore_index=True)
df_combined = df_combined[['Sequence', 'log_mean', 'Source']]

# Check that there is no overlap between reward_model_test_df and SFT_df
duplicates = df_combined[df_combined['Sequence'].duplicated()]
print('duplicates', duplicates)

##############################################################################################################################

# Calculate reward model predictions
for j in range(len(df_combined)): # len(df_combined)
    scores = []
    sequence = df_combined['Sequence'].iloc[j]
    sequence = torch.tensor(aa2ind(list(sequence)))

    for model in models:
        score = model.predict(sequence).item()
        scores.append(score)

    # Calculate the scores from num_models scores
    # median_score = np.median(scores)
    scores_tensor = torch.tensor(scores)
    fifth_percentile_score = torch.quantile(scores_tensor, 0.05).item()

    # Add scores to df_combined
    # df_combined.at[j, 'EnsMLP_Score'] = median_score
    df_combined.at[j, 'Conservative_EnsMLP_Score'] = fifth_percentile_score

    # Print progress every 1000 sequences
    if (j + 1) % 1000 == 0:
        print(f'{j + 1} sequences have been scored.')

# Save the updated DataFrame to a CSV file
df_combined.to_csv('conservative_correlations.csv', index=False)
print('Finished scoring sequences with reward models')

##############################################################################################################################

# Calculate scores from pre-trained VAE and aligned VAE using function from Riesselman et al, 2018

# Calculate logits for WT
tokenized_WT = torch.tensor(aa2ind(list(WT))).unsqueeze(0) # Adds a batch dimension

# Score test set and SFT sequences
for j in range(len(df_combined)):
    sequence = df_combined['Sequence'].iloc[j]
    sequence = torch.tensor(aa2ind(list(sequence)))
    sequence = sequence.unsqueeze(0)  # Adds a batch dimension

    with torch.no_grad():
        # Calculate score with fixed vae
        z_mean, z_log_var, encoded, decoded = fixed_vae(sequence)
        fixed_logits = fixed_vae.decoder(z_mean)
        WT_fixed_vae_score = F.cross_entropy(fixed_logits, tokenized_WT, reduction='none')
        mutant_fixed_vae_score = F.cross_entropy(fixed_logits, sequence, reduction='none')
        fixed_vae_score = (mutant_fixed_vae_score - WT_fixed_vae_score).sum().numpy()
        df_combined.at[j, 'fixed_vae_score'] = fixed_vae_score

        # Calculate score with aligned vae
        z_mean, z_log_var, encoded, decoded = aligned_vae(sequence)
        aligned_logits = aligned_vae.decoder(z_mean)
        WT_aligned_vae_score = F.cross_entropy(aligned_logits, tokenized_WT, reduction='none')
        mutant_aligned_vae_score = F.cross_entropy(aligned_logits, sequence, reduction='none')
        aligned_vae_score = (mutant_aligned_vae_score - WT_aligned_vae_score).sum().numpy()
        df_combined.at[j, 'aligned_vae_score'] = aligned_vae_score

    # Print progress every 1000 sequences
    if (j + 1) % 1000 == 0:
        print(f'{j + 1} sequences have been scored.')

# Save the updated DataFrame to a CSV file
df_combined.to_csv('correlations.csv', index=False)
print('Finished scoring sequences using function from Riesselman et al, 2018')

##############################################################################################################################

def ESM2_mutant_marginal(model, tokenizer, sequence, WT):
    '''
    Masked marginal probability (1 forward pass per mutation per sequence)
    from https://proceedings.neurips.cc/paper_files/paper/2021/file/f51338d736f95dd42427296047067694-Supplemental.pdf
    
    Score sequences by masking every mutated position and computing the log odds ratio between the mutated and wild-type
    residues at each mutated position, assuming an additive model when a sequence contains multiple mutations
    '''
    # Tokenize WT and mutated sequence for ESM2
    WT_inputs = tokenizer(WT, return_tensors='pt', padding=True, truncation=True)
    inputs = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True)

    # Determine mutated positions
    mutated_positions = [i for i, (wt, mt) in enumerate(zip(WT, sequence)) if wt != mt]

    # Get input_ids and prepare for masked operation
    input_ids = inputs['input_ids'].clone()
    scores = []

    with torch.no_grad():
        # Iterate only over mutated positions
        for index in mutated_positions:
            masked_input_ids = input_ids.clone()
            masked_index = index + 1  # Adjust index for tokenizer specifics (CLS token at the start)
            masked_input_ids[0, masked_index] = tokenizer.mask_token_id
            
            # Get model output for masked input
            outputs = model(masked_input_ids)
            logits = outputs.logits

            # Calculate log probabilities at the masked position
            log_probs = F.log_softmax(logits, dim=-1)

            # Get the log probabilities of the actual wildtype and mutant amino acids at this position
            wt_log_prob = log_probs[0, masked_index, WT_inputs['input_ids'][0, masked_index]]
            mutant_log_prob = log_probs[0, masked_index, input_ids[0, masked_index]]

            # Compute the score for this position (mutant - WT)
            score = (mutant_log_prob - wt_log_prob).item()
            scores.append(score)

    # Sum scores for all mutated positions
    ESM2_score = sum(scores)
    return ESM2_score

# Calculate scores from ESM-2, SFT ESM-2, aligned ESM-2 using ESM2_mutant_marginal function from Meier et al, 2018

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

for j in range(len(df_combined)): # len(df_combined)
    sequence = df_combined['Sequence'].iloc[j]
    
    fixed_ESM2_score = ESM2_mutant_marginal(fixed_ESM2, tokenizer,sequence, WT)
    df_combined.at[j, 'fixed_ESM2_mutant_marginal_score'] = fixed_ESM2_score

    sft_ESM2_score = ESM2_mutant_marginal(sft_ESM2, tokenizer,sequence, WT)
    df_combined.at[j, 'sft_ESM2_mutant_marginal_score'] = sft_ESM2_score

    aligned_ESM2_score = ESM2_mutant_marginal(aligned_ESM2, tokenizer,sequence, WT)
    df_combined.at[j, 'aligned_mutant_marginal_ESM2_score'] = aligned_ESM2_score

    # Print progress every 100 sequences
    if (j + 1) % 50 == 0:
        print(f'{j + 1} sequences have been scored.')


# Save the updated DataFrame to a CSV file
df_combined.to_csv('correlations.csv', index=False)
print('Finished scoring sequences using function from ESM2_mutant_marginal')

##############################################################################################################################

def ESM2_masked_marginal(model, tokenizer, sequence, WT):
    '''
    Masked marginal probability (1 forward passes / sequence)
    from https://proceedings.neurips.cc/paper_files/paper/2021/file/f51338d736f95dd42427296047067694-Supplemental.pdf
    
    We score mutations using the log odds ratio at the mutated position, assuming an additive model when multiple mutations
    T exist in the same sequence. Here the sum is over the mutated positions, and the sequence input to the model is masked
    at every mutated position.
    '''
    # Tokenize WT and mutated sequence for ESM2
    WT_inputs = tokenizer(WT, return_tensors='pt', padding=True, truncation=True)
    inputs = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True)

    # Determine mutated positions
    mutated_positions = [i for i, (wt, mt) in enumerate(zip(WT, sequence)) if wt != mt]

    # Get input_ids and prepare for masked operation
    input_ids = inputs['input_ids'].clone()
    scores = []

    with torch.no_grad():
        # Mask the mutated positions
        masked_input_ids = input_ids.clone()
        for index in mutated_positions:
            masked_index = index + 1  # Adjust index for tokenizer specifics (CLS token at the start)
            masked_input_ids[0, masked_index] = tokenizer.mask_token_id
        
        # Get model output for masked input
        outputs = model(masked_input_ids)
        logits = outputs.logits

        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Iterate only over mutated positions
        for index in mutated_positions:
            masked_index = index + 1  # Adjust index for tokenizer specifics (CLS token at the start)
            
            # Get the log probabilities of the actual wildtype and mutant amino acids at this position
            wt_log_prob = log_probs[0, masked_index, WT_inputs['input_ids'][0, masked_index]]
            mutant_log_prob = log_probs[0, masked_index, input_ids[0, masked_index]]

            # Compute the score for this position (mutant - WT)
            score = (mutant_log_prob - wt_log_prob).item()
            scores.append(score)

    # Sum scores for all mutated positions
    ESM2_score = sum(scores)
    return ESM2_score

# Calculate scores from ESM-2, SFT ESM-2, aligned ESM-2 using function from Meier et al, 2018

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

for j in range(len(df_combined)): # len(df_combined)
    sequence = df_combined['Sequence'].iloc[j]
    
    fixed_ESM2_score = ESM2_masked_marginal(fixed_ESM2,tokenizer,sequence, WT)
    df_combined.at[j, 'fixed_ESM2_masked_marginal_score'] = fixed_ESM2_score

    sft_ESM2_score = ESM2_masked_marginal(sft_ESM2,tokenizer,sequence, WT)
    df_combined.at[j, 'sft_ESM2_masked_marginal_score'] = sft_ESM2_score

    aligned_ESM2_score = ESM2_masked_marginal(aligned_ESM2,tokenizer,sequence, WT)
    df_combined.at[j, 'aligned_ESM2_masked_marginal_score'] = aligned_ESM2_score

    # Print progress every 100 sequences
    if (j + 1) % 50 == 0:
        print(f'{j + 1} sequences have been scored.')

# Save the updated DataFrame to a CSV file
df_combined.to_csv('correlations.csv', index=False)
print('Finished scoring sequences using function from ESM2_masked_marginal')

##############################################################################################################################

def ESM2_pseudo_perplexity(model, tokenizer, sequence, WT):
    '''
    Pseudo perplexity (len(WT) forward passes)
    from https://arxiv.org/pdf/1910.14659
    
    Score sequences by masking every position in each sequence one at a time and summing the log likelihoods for each residue
    '''
    # Tokenize the sequence
    inputs = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True)

    # Get input_ids and prepare for masked operation
    input_ids = inputs['input_ids'].clone()
    log_likelihoods = []

    with torch.no_grad():
        # Iterate over each position in the sequence, ignoring special tokens such as CLS and SEP
        for index in range(1, input_ids.size(1) - 1):  # Assuming first and last tokens are special tokens
            # Mask the current position
            masked_input_ids = input_ids.clone()
            masked_input_ids[0, index] = tokenizer.mask_token_id
            
            # Get model output for masked input
            outputs = model(masked_input_ids)
            logits = outputs.logits

            # Calculate log probabilities at the masked position
            log_probs = F.log_softmax(logits, dim=-1)

            # Get the log probability of the actual amino acid at this position
            log_prob = log_probs[0, index, input_ids[0, index]]

            # Append the log probability for this position to the list
            log_likelihoods.append(log_prob)

    # Sum log probabilities across all positions for the final score
    pseudo_perplexity_score = sum(log_likelihoods).item()

    return pseudo_perplexity_score

# Calculate pseudo-perplexity scores from ESM-2, SFT ESM-2, and aligned ESM-2 for datasets

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

for j in range(len(df_combined)): # len(df_combined)
    sequence = df_combined['Sequence'].iloc[j]
    
    fixed_ESM2_pseudo_perplexity = ESM2_pseudo_perplexity(fixed_ESM2,tokenizer,sequence, WT)
    df_combined.at[j, 'fixed_ESM2_pseudo_perplexity'] = fixed_ESM2_pseudo_perplexity

    sft_ESM2_pseudo_perplexity = ESM2_pseudo_perplexity(sft_ESM2,tokenizer,sequence, WT)
    df_combined.at[j, 'sft_ESM2_pseudo_perplexity'] = sft_ESM2_pseudo_perplexity

    aligned_ESM2_pseudo_perplexity = ESM2_pseudo_perplexity(aligned_ESM2,tokenizer,sequence, WT)
    df_combined.at[j, 'aligned_ESM2_pseudo_perplexity'] = aligned_ESM2_pseudo_perplexity

    # Print progress every 100 sequences
    if (j + 1) % 50 == 0:
        print(f'{j + 1} sequences have been scored.')

# Save the updated DataFrame to a CSV file
df_combined.to_csv('correlations.csv', index=False)
print('Finished scoring sequences using function from ESM2_pseudo_perplexity')

##############################################################################################################################

# Assuming df_combined is your existing DataFrame
# Specified columns to be copied
columns_to_copy = [
    'Sequence', 'log_mean', 'Source', 'EnsMLP_Score', 'Conservative_EnsMLP_Score', 'fixed_vae_score', 
    'aligned_vae_score', 'fixed_ESM2_pseudo_perplexity',
    'sft_ESM2_pseudo_perplexity', 'aligned_ESM2_pseudo_perplexity',
    'fixed_ESM2_mutant_marginal_score', 'sft_ESM2_mutant_marginal_score',
    'aligned_mutant_marginal_ESM2_score',
    'fixed_ESM2_masked_marginal_score', 'sft_ESM2_masked_marginal_score',
    'aligned_ESM2_masked_marginal_score'
]

# Creating a new DataFrame with the specified columns
final_df = df_combined[columns_to_copy]

WT_length = len(WT)

# Columns to be processed
columns_to_process = [
    'fixed_ESM2_pseudo_perplexity',
    'sft_ESM2_pseudo_perplexity',
    'aligned_ESM2_pseudo_perplexity'
]

# Applying the transformation to the specified columns
for column in columns_to_process:
    final_df[column] = -1*np.exp(-final_df[column] / WT_length) # smaller is better

# Save the updated DataFrame to a CSV file
df_combined.to_csv('correlations.csv', index=False)
final_df.to_csv('finalized_correlations.csv', index=False)
print('Finished scoring sequences using all function and process scores')

##############################################################################################################################

# Calculate spearmann correlations and top 10% recall with reward model test split data and SFT preference data

from sklearn.metrics import recall_score

def top_k_recall(true_scores, model_scores, percent_cutoff=10):
    ''' Adapted from
    https://github.com/OATML-Markslab/ProteinGym/blob/495cc305135767b53478dda1e12039c30d7f82ce/proteingym/performance_DMS_benchmarks.py#L72
    '''
    top_true = (true_scores >= np.percentile(true_scores, 100-percent_cutoff))
    top_model = (model_scores >= np.percentile(model_scores, 100-percent_cutoff))
    TP = (top_true) & (top_model)
    recall = TP.sum() / (top_true.sum())
    return recall

test_df = final_df[final_df['Source'] == 'reward_model_test_df']
SFT_df = final_df[final_df['Source'] == 'SFT_df']

# List to hold correlation results
correlation_results = []

# Columns to calculate correlations for
columns_with_metrics = ['EnsMLP_Score', 'Conservative_EnsMLP_Score','fixed_vae_score',
       'aligned_vae_score', 'fixed_ESM2_pseudo_perplexity',
       'sft_ESM2_pseudo_perplexity', 'aligned_ESM2_pseudo_perplexity',
       'fixed_ESM2_mutant_marginal_score', 'sft_ESM2_mutant_marginal_score',
       'aligned_mutant_marginal_ESM2_score',
       'fixed_ESM2_masked_marginal_score', 'sft_ESM2_masked_marginal_score',
       'aligned_ESM2_masked_marginal_score']

# DataFrames for correlation calculations
dataframes = {
    'reward_model_test_df': test_df,
    'SFT_df': SFT_df
}

# Calculate Spearman correlations for each metric in each DataFrame
for df_name, df in dataframes.items():
    for metric in columns_with_metrics:
        if metric in df.columns:
            spearman_corr, p_value = spearmanr(df[metric], df['log_mean'])
            accuracy = top_k_recall(df['log_mean'], df[metric])
            
            # Append the results to the list
            correlation_results.append({
                'Source': df_name,
                'Metric': metric,
                'Spearman_Correlation': spearman_corr,
                'Top 10% Recall': accuracy
            })
        else:
            print(f"Column {metric} not found in {df_name} DataFrame.")

# Convert the list of results into a DataFrame
correlation_df = pd.DataFrame(correlation_results)

correlation_df.head()


# Save the updated DataFrame to a CSV file
correlation_df.to_csv('calculated_correlations.csv', index=False)
print('Finished calculating correlations between model scores')

# Filter rows where Source is 'reward_model_test_df'
filtered_df = correlation_df[correlation_df['Source'] == 'reward_model_test_df']
filtered_df.to_csv('reward_model_calculated_correlations.csv', index=False)
print('Finished calculating correlations for reward model test set')


# Filter rows where Source is 'reward_model_test_df'
filtered_df = correlation_df[correlation_df['Source'] == 'SFT_df']
filtered_df.to_csv('SFT_calculated_correlations.csv', index=False)
print('Finished calculating correlations for synthetic SFT dataset')







