#!/usr/bin/env python
# coding: utf-8

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
from functions import (load_reward_model, hamming_distance, generate_df, generate_and_evaluate_mutants_max_sampling)
from functions import (mask_sequence, get_logits_for_all_positions, generate_heatmap, get_mutations, generate_high_confidence_mutant_sequences, generate_mutated_sequences, generate_and_evaluate_mutants_p_sampling)
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
from matplotlib.colors import LinearSegmentedColormap

################################################################################################################

# Define amino acid dictionary for tokenization, define WT for length of context window
AAs = 'ACDEFGHIKLMNPQRSTVWY' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
sequence_length = len(WT)
num_EnsMLPs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(20)

################################################################################################################

# Load reward models
reward_models = []
for i in range(num_EnsMLPs):
    model_name = f"best_model_v{i}.ckpt"
    checkpoint_path = f"../RLXF_PPO_from_pretrained_ESM2_GPU/MLP_Reward_Models/{model_name}"
    reward_model = load_reward_model(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_models.append(reward_model)

################################################################################################################

versions = [10, 11, 12, 13, 14, 15]
for version in versions:

    # Shared parameters for generating designs from pretrained, sft, and aligned models
    huggingface_identifier ='esm2_t33_650M_UR50D' # esm2_t33_650M_UR50D # esm2_t30_150M_UR50D # esm2_t12_35M_UR50D	# esm2_t6_8M_UR50D
    pretrained_ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")
    model_identifier = huggingface_identifier
    num_designs = 1000
    num_muts = 5
    # cum_prob_threshold = 0.25
    seed = 7028

    # Where to save figures and data
    data = 'DPO_CreiLOV_DMS_dataset_w_ESM2_log_likelihood'
    filepath = f'./logs/DPO_{huggingface_identifier}_with_{data}'

    # version = 4
    ep = 30 # 5

    # high_conf_threshold = 0.9

    # Generate loss curves
    create_loss_curves = False

    # Generate histogram
    generate_histogram = True

    # Generate designs from pretrained model
    generate_pretrained_designs = True
    if generate_pretrained_designs == False:
        saved_fixed_mutants_version = 4 # May need to be changed if you already have designs from pretrained model in a different version file than 0
        saved_fixed_mutants_path = f'{filepath}/version_{saved_fixed_mutants_version}'

    # Generate designs from sft model
    generate_sft_designs = True
    sft_model_exists = True
    saved_sft_mutants_version = 6 # May need to be changed if you already have designs from pretrained model in a different version file than 0
    sft_model_filepath = f'../RLXF_SFT_from_pretrained_ESM2_GPU/logs/SFT_pretrained_ESM2_models/version_{saved_sft_mutants_version}'
    sft_model_name = f'SFT_esm2_t33_650M_UR50D_v6' # ! this is the same model as ../RLXF_SFT_from_pretrained_ESM2_GPU/logs/SFT_ESM2_650M_with_SA_data/version_3/SFT_ESM2_650M_with_data_v5_and_random_masking_v0.pt

    # Generate designs from aligned model
    generate_aligned_designs = True
    rl_model_exists = True
    rl_model_filepath = f'{filepath}/version_{version}'
    rl_model_name = f'multitask_ESM2'

    ################################################################################################################

    if generate_pretrained_designs:
        saved_fixed_mutants_version = version
        
        # Generate designs with pretrained ESM2
        # fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, sft_ESM2, model_identifier, tokenizer, filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
        fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
                                                                    num_muts=num_muts,
                                                                    WT=WT,
                                                                    reward_models=reward_models,
                                                                    model=pretrained_ESM2,
                                                                    seed=seed,
                                                                    model_identifier=model_identifier)
        print("Status: finished generating sequences with ESM2")

        # Save mutants from ESM2
        base_path = f'{filepath}/version_{version}/'
        np.save(base_path + f'fixed_scores_max_sampling_{num_muts}.npy', fixed_scores_np)
        with open(base_path + f'fixed_mutated_seqs_max_sampling_{num_muts}.txt', 'w') as file:
            for seq in fixed_mutated_seqs:
                file.write(seq + '\n')

    else:
        print('Skipping generating sequences from pretrained model')

    if generate_histogram:
        # Load mutants from pretrained model if generating histograms
        fixed_scores_np = np.load(f'{filepath}/version_{saved_fixed_mutants_version}/fixed_scores_max_sampling_{num_muts}.npy')
        fixed_mutated_seqs = []
        with open(f'{filepath}/version_{saved_fixed_mutants_version}/fixed_mutated_seqs_max_sampling_{num_muts}.txt', 'r') as file:
            fixed_mutated_seqs = file.read().splitlines()
        df_fixed = generate_df(fixed_mutated_seqs, np.median(fixed_scores_np, axis=0))
        df_fixed.to_csv(f'{filepath}/version_{version}/fixed_mutated_designs_scores_mutations_max_sampling_{num_muts}.csv', index=False)

    ################################################################################################################

    if generate_sft_designs:
        model_identifier = f"sft_{model_identifier}"
        sft_ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
        state_dict = torch.load(f'{sft_model_filepath}/{sft_model_name}.pt', map_location=torch.device('cpu'))
        sft_ESM2.load_state_dict(state_dict)
        
        # Generate designs with sft ESM2
        # sft_mutated_seqs, sft_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, sft_ESM2, model_identifier, tokenizer, filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
        sft_mutated_seqs, sft_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
                                                                    num_muts=num_muts,
                                                                    WT=WT,
                                                                    reward_models=reward_models,
                                                                    model=sft_ESM2,
                                                                    seed=seed,
                                                                    model_identifier=model_identifier)
        print("Status: finished generating sequences with sft ESM2")
        
        # Save mutants from ESM2
        base_path = f'{filepath}/version_{version}/'
        np.save(base_path + f'sft_scores_max_sampling_{num_muts}.npy', sft_scores_np)
        with open(base_path + f'sft_mutated_seqs_max_sampling_{num_muts}.txt', 'w') as file:
            for seq in sft_mutated_seqs:
                file.write(seq + '\n')

    else:
        print('Skipping generating sequences from sft model')

    if generate_histogram:
        # Load mutants from pretrained model if generating histograms
        sft_scores_np = np.load(f'{filepath}/version_{saved_fixed_mutants_version}/sft_scores_max_sampling_{num_muts}.npy')
        sft_mutated_seqs = []
        with open(f'{filepath}/version_{saved_fixed_mutants_version}/sft_mutated_seqs_max_sampling_{num_muts}.txt', 'r') as file:
            sft_mutated_seqs = file.read().splitlines()
        df_sft = generate_df(sft_mutated_seqs, np.median(sft_scores_np, axis=0))
        df_sft.to_csv(f'{filepath}/version_{version}/sft_mutated_designs_scores_mutations_max_sampling_{num_muts}.csv', index=False)
    
    ################################################################################################################

    if generate_aligned_designs:
        model_identifier = f"aligned_{model_identifier}_non_EMA"
        rl_ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
        checkpoint = torch.load(f'{rl_model_filepath}/{rl_model_name}.pt', map_location=torch.device('cpu'))
        state_dict = {k.replace("ESM2.", ""): v for k, v in checkpoint.items()}
        rl_ESM2.load_state_dict(state_dict)
        
        # Generate designs with rl ESM2
        # rl_mutated_seqs, rl_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, rl_ESM2, model_identifier, tokenizer, filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
        rl_mutated_seqs, rl_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
                                                                    num_muts=num_muts,
                                                                    WT=WT,
                                                                    reward_models=reward_models,
                                                                    model=rl_ESM2,
                                                                    seed=seed,
                                                                    model_identifier=model_identifier)
        print("Status: finished generating sequences with aligned ESM2")
        
        # Save mutants from ESM2
        base_path = f'{filepath}/version_{version}/'
        np.save(base_path + f'non_EMA_rl_scores_max_sampling_{num_muts}.npy', rl_scores_np)
        with open(base_path + f'non_EMA_rl_mutated_seqs_max_sampling_{num_muts}.txt', 'w') as file:
            for seq in rl_mutated_seqs:
                file.write(seq + '\n')

    else:
        print('Skipping generating sequences from aligned model')

    if generate_histogram:
        if generate_aligned_designs == False:
            model_identifier = f"aligned_{model_identifier}"
        
        # Load mutants
        rl_scores_np = np.load(f'{filepath}/version_{version}/non_EMA_rl_scores_max_sampling_{num_muts}.npy')
        rl_mutated_seqs = []
        with open(f'{filepath}/version_{version}/non_EMA_rl_mutated_seqs_max_sampling_{num_muts}.txt', 'r') as file:
            rl_mutated_seqs = file.read().splitlines()
        df_rl = generate_df(rl_mutated_seqs, np.median(rl_scores_np, axis=0))
        df_rl.to_csv(f'{filepath}/version_{version}/non_EMA_rl_mutated_designs_scores_mutations_max_sampling_{num_muts}.csv', index=False)
    
    ################################################################################################################

    if generate_histogram:
        # Plot histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # Constants for the mean and standard deviation
        predicted_wt_score = 4.1498 # this is predicted WT score # mean log exp score: 4.094413241
        if generate_histogram and sft_model_exists and rl_model_exists:
            # Plot histograms for the models
            sns.histplot(np.median(fixed_scores_np, axis=0), bins=25, alpha=0.4, color='grey', edgecolor='black', stat='density', ax=ax1, label='Pre-trained ESM2')
            sns.histplot(np.median(sft_scores_np, axis=0), bins=25, alpha=0.6, color='yellow', edgecolor='black', stat='density', ax=ax1, label='SFT ESM2')
            sns.histplot(np.median(rl_scores_np, axis=0), bins=25, alpha=0.6, color='blue', edgecolor='black', stat='density', ax=ax1, label='Aligned ESM2')
            ax1.set_xlabel('Predicted Fluorescence', fontsize=12)
            ax1.set_ylabel('Probability Density', fontsize=12)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.axvline(predicted_wt_score, color='orange', linestyle='--', linewidth=3)
            ax1.axvspan(min(min(np.median(fixed_scores_np, axis=0))-0.05, min(np.median(rl_scores_np, axis=0))-0.05), predicted_wt_score, color='red', alpha=0.1, zorder=-1)
            ax1.axvspan(predicted_wt_score, max(max(np.median(fixed_scores_np, axis=0)) + 0.05, max(np.median(rl_scores_np, axis=0)) + 0.05), color='green', alpha=0.1, zorder=-1)
            ax1.legend()
            # Plot the cumulative density plot on the second subplot for all models
            sns.ecdfplot(np.median(fixed_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="grey", linestyle='-')
            sns.ecdfplot(np.median(sft_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="yellow", linestyle='-')
            sns.ecdfplot(np.median(rl_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="blue", linestyle='-')
            ax2.set_xlabel('Predicted Fluorescence', fontsize=12)
            ax2.set_ylabel('Cumulative Density', fontsize=12)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.axvline(predicted_wt_score, color='orange', linestyle='--', linewidth=3)
            ax2.axvspan(min(min(np.median(fixed_scores_np, axis=0))-0.05, min(np.median(rl_scores_np, axis=0))-0.05), predicted_wt_score, color='red', alpha=0.1, zorder=-1)
            ax2.axvspan(predicted_wt_score, max(max(np.median(fixed_scores_np, axis=0)) + 0.05, max(np.median(rl_scores_np, axis=0)) + 0.05), color='green', alpha=0.1, label='Better than Predicted WT Fluorescence', zorder=-1)
            less_wt_patch = mpatches.Patch(color='red', alpha=0.8, label='Less than Predicted WT Log Fluorescence')
            wt_line = mpatches.Patch(color='orange', alpha=0.8, label='Predicted WT Log Fluorescence')
            better_wt_patch = mpatches.Patch(color='green', alpha=0.8, label='Greater than Predicted WT Log Fluorescence')
            legend = ax2.legend(handles=[less_wt_patch, wt_line, better_wt_patch], frameon=True, edgecolor='black')
            plt.setp(legend.get_texts(), color='black', fontsize=10)
            plt.setp(legend.get_frame(), facecolor='white')
            plt.tight_layout()
            # Save the plot
            plt.savefig(f'{filepath}/version_{version}/design_scores_{model_identifier}_ep{ep}_max_sampling_{num_muts}.svg')
            plt.savefig(f'{filepath}/version_{version}/design_scores_{model_identifier}_ep{ep}_max_sampling_{num_muts}.png')

        elif generate_histogram and rl_model_exists:
            # Plot histograms for the models
            sns.histplot(np.median(fixed_scores_np, axis=0), bins=25, alpha=0.4, color='grey', edgecolor='black', stat='density', ax=ax1, label='Pre-trained ESM2')
            sns.histplot(np.median(rl_scores_np, axis=0), bins=25, alpha=0.6, color='blue', edgecolor='black', stat='density', ax=ax1, label='Aligned ESM2')
            ax1.set_xlabel('Predicted Fluorescence', fontsize=12)
            ax1.set_ylabel('Probability Density', fontsize=12)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.axvline(predicted_wt_score, color='orange', linestyle='--', linewidth=3)
            ax1.axvspan(min(min(np.median(fixed_scores_np, axis=0))-0.05, min(np.median(rl_scores_np, axis=0))-0.05), predicted_wt_score, color='red', alpha=0.1, zorder=-1)
            ax1.axvspan(predicted_wt_score, max(max(np.median(fixed_scores_np, axis=0)) + 0.05, max(np.median(rl_scores_np, axis=0)) + 0.05), color='green', alpha=0.1, zorder=-1)
            ax1.legend()
            # Plot the cumulative density plot on the second subplot for all models
            sns.ecdfplot(np.median(fixed_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="grey", linestyle='-')
            sns.ecdfplot(np.median(rl_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="blue", linestyle='-')
            ax2.set_xlabel('Predicted Fluorescence', fontsize=12)
            ax2.set_ylabel('Cumulative Density', fontsize=12)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.axvline(predicted_wt_score, color='orange', linestyle='--', linewidth=3)
            ax2.axvspan(min(min(np.median(fixed_scores_np, axis=0))-0.05, min(np.median(rl_scores_np, axis=0))-0.05), predicted_wt_score, color='red', alpha=0.1, zorder=-1)
            ax2.axvspan(predicted_wt_score, max(max(np.median(fixed_scores_np, axis=0)) + 0.05, max(np.median(rl_scores_np, axis=0)) + 0.05), color='green', alpha=0.1, label='Better than Predicted WT Fluorescence', zorder=-1)
            less_wt_patch = mpatches.Patch(color='red', alpha=0.8, label='Less than Predicted WT Log Fluorescence')
            wt_line = mpatches.Patch(color='orange', alpha=0.8, label='Predicted WT Log Fluorescence')
            better_wt_patch = mpatches.Patch(color='green', alpha=0.8, label='Greater than Predicted WT Log Fluorescence')
            legend = ax2.legend(handles=[less_wt_patch, wt_line, better_wt_patch], frameon=True, edgecolor='black')
            plt.setp(legend.get_texts(), color='black', fontsize=10)
            plt.setp(legend.get_frame(), facecolor='white')
            plt.tight_layout()
            # Save the plot
            plt.savefig(f'{filepath}/version_{version}/design_scores_{model_identifier}_ep{ep}_max_sampling_{num_muts}.svg')
            plt.savefig(f'{filepath}/version_{version}/design_scores_{model_identifier}_ep{ep}_max_sampling_{num_muts}.png')
        
        else:
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
            plt.savefig(f'{filepath}/version_{version}/design_scores_{model_identifier}_max_sampling_{num_muts}.svg')
            plt.savefig(f'{filepath}/version_{version}/design_scores_{model_identifier}_max_sampling_{num_muts}.png')

    ################################################################################################################
    if create_loss_curves and rl_model_exists:
        # Plotting metrics
        pt_metrics = pd.read_csv(f'{filepath}/version_{version}/metrics.csv')
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
        plt.savefig(f'{filepath}/version_{version}/metrics_vs_steps.svg')
        plt.savefig(f'{filepath}/version_{version}/metrics_vs_steps.png')
        print('saved learning curves from aligned model')

    elif create_loss_curves:
        # Plotting metrics
        pt_metrics = pd.read_csv(f'{filepath}/version_{version}/metrics.csv')
        # Define the metrics you want to plot
        metrics_to_plot = [
            ['train_loss']]
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
        plt.savefig(f'{filepath}/version_{version}/metrics_vs_steps.svg')
        plt.savefig(f'{filepath}/version_{version}/metrics_vs_steps.png')
        print('saved learning curves from sft model')

    else:
        print('Skipping generating loss cruves')
    ################################################################################################################



