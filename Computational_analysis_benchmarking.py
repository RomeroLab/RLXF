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
from functions import (hamming_distance, generate_df)
from functions import (mask_sequence, get_logits_for_all_positions, generate_heatmap, get_mutations, generate_high_confidence_mutant_sequences, generate_mutated_sequences, generate_and_evaluate_mutants_p_sampling)
#from dataloading_RLXF_ESM2 import (ProtDataModuleESM2, ProtRepDatasetESM2)
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
import scipy
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(20)

# load mutants for kde plot
esm2_models = ['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D']
num_reward_models = 10
version = 8
sft_version = 0
ep = 5

# Define avGFP sequence
WT_name = "avGFP"
WT = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
slen = len(WT)

# generating designs
num_designs = 50
num_muts_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
high_conf_threshold = 0.9
cum_prob_threshold = 0.25
seed = 7028
filepath = './logs'

# create folder structure it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

if not os.path.exists('logs/figures'):
    os.makedirs('./logs/figures')
    os.makedirs('./logs/figures/8M')
    os.makedirs('./logs/figures/35M')
    os.makedirs('./logs/figures/150M')
    os.makedirs('./logs/figures/650M')

# load ensemble of reward models
reward_models = []
for i in range(num_reward_models):
    model_name = f"reward_model_v{i}.ckpt"
    checkpoint_path = f"./reward_models/{model_name}"
    reward_model = MLP.load_from_checkpoint(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_models.append(reward_model)

for huggingface_identifier in esm2_models:
    dir_filepath = f'./logs/PPO_{huggingface_identifier}' # ! update
    model_size = huggingface_identifier.split('_')[2]

    # Load mutants from pretrained ESM2 650M, sft, and aligned models
    fixed_scores_np = np.load(f'{dir_filepath}/version_{version}/fixed_{huggingface_identifier}_scores.npy') # ! update
    sft_scores_np = np.load(f'{dir_filepath}/version_{version}/sft_{huggingface_identifier}_scores.npy') # ! update
    rl_scores_np = np.load(f'{dir_filepath}/version_{version}/ema_aligned_{huggingface_identifier}_scores.npy') # ! update

    # Constants for the mean and standard deviation
    scores = []
    for reward_model in reward_models:
        reward_model.to(device)
        reward_model.eval()
        with torch.no_grad():
            score = reward_model.predict(WT)[0][0].cpu()  # Assuming predict returns a nested list/array
            scores.append(score)
    predicted_wt_score = np.median(np.array(scores, dtype=np.float32))

    # Plot histogram
    fig, ax = plt.subplots(figsize=(6, 6))
    alpha = 0.5

    # Plot histograms for the models7
    sns.kdeplot(np.median(fixed_scores_np, axis=0), color='#bdbdbd', ax=ax, linewidth=2.5, fill=True, alpha=alpha, label=f'Pre-trained ESM2 ({model_size})')
    sns.kdeplot(np.median(sft_scores_np, axis=0), color='#92c5de', ax=ax, linewidth=2.5, fill=True, alpha=alpha, label=f'SFT ESM2 ({model_size})')
    sns.kdeplot(np.median(rl_scores_np, axis=0), color='#2166ac', ax=ax, linewidth=2.5, fill=True, alpha=alpha, label=f'Aligned ESM2 ({model_size})')
    ax.axvline(predicted_wt_score, color='black', linestyle='--', linewidth=1, label=f'Predicted {WT_name} score')
    ax.set_xlabel('Predicted Fluorescence', fontsize=12)
    ax.set_ylabel('Density', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'./logs/figures/{model_size}/ppo_sft_pretrained_esm2_design_scores.svg')
    plt.savefig(f'./logs/figures/{model_size}/ppo_sft_pretrained_esm2_design_scores.png')

    # Load the data
    ema_filepath = f"./logs/PPO_{huggingface_identifier}/version_{version}/ema_aligned_{huggingface_identifier}_mutated_designs_scores_ep{ep}.csv"
    fixed_filepath = f"./logs/PPO_{huggingface_identifier}/version_{version}/{huggingface_identifier}_fixed_mutated_designs_scores.csv"
    ema_df = pd.read_csv(ema_filepath)[["Sequence"]].head(30)
    ema_df["Model"] = f"Aligned_ESM2_{model_size}"
    fixed_df = pd.read_csv(fixed_filepath)[["Sequence"]].head(30)
    fixed_df['Model'] = f"Pretrained_ESM2_{model_size}"
    df = pd.concat([ema_df, fixed_df], ignore_index=True)
    df = df.rename(columns={'Sequence': 'AA_sequence'})
    # df.head()

    # Initialize a dictionary to store mutation counts for each model
    mutation_counts = {model: np.zeros(slen) for model in df['Model'].unique()}

    # Count mutations for each model
    for _, row in df.iterrows():
        model = row['Model']
        seq = row['AA_sequence']
        for i in range(slen):
            if seq[i] != WT[i]:
                mutation_counts[model][i] += 1

    # Create a dataframe for the heatmap
    mutation_df = pd.DataFrame(mutation_counts).T

    # Create a new DataFrame by subtracting Pre-trained from Aligned for each model
    mutation_difference_df = pd.DataFrame()
    mutation_difference_df['ESM2_Aligned-Pretrained'] = mutation_df.iloc[0] - mutation_df.iloc[1]

    # Create a custom colormap, setting white as the center
    min_score_1 = np.min(mutation_difference_df)
    max_score_1 = np.max(mutation_difference_df)
    midpoint = abs(min_score_1) / (max_score_1 - min_score_1)
    midpoint = midpoint.values[0]
    colors = [(0, '#B2182B'), (midpoint, 'white'), (1, '#2166AC')]
    cmap_name = 'custom'
    custom_cmap_1 = LinearSegmentedColormap.from_list(cmap_name, colors)

    # Create figure
    fig, ax = plt.subplots(figsize=(slen/4.5, 10))

    # Plot the heatmap
    sns.heatmap(mutation_difference_df.T, cmap=custom_cmap_1, vmin=min_score_1, vmax=max_score_1, square=True, cbar=True, 
                yticklabels=mutation_difference_df.T.index, ax=ax, linewidths=0.5, linecolor='black')
    ax.set_xticks(np.arange(mutation_df.shape[1]) + 0.5)
    ax.set_xticklabels(list(WT))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', labeltop=True, labelbottom=False)
    plt.savefig(f'logs/figures/{model_size}/Mutational_Freq_All_Models.svg')
    plt.savefig(f'logs/figures/{model_size}/Mutational_Freq_All_Models.png')

    # Function to convert counts to probabilities
    def counts_to_probabilities(arr: np.ndarray):
        aa_dict = {aa:0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        aas, counts = np.unique(arr, return_counts=True)
        counts = counts / counts.sum()
        aa_dict.update({aa:count for aa, count in zip(aas, counts)})
        return np.array(list(aa_dict.values()))

    # Function to calculate sequence entropy
    def sequence_entropy(seq_list, axis=1):
        seq_arr = np.vstack([np.array(list(s)) for s in seq_list])
        probabilities = np.vstack([counts_to_probabilities(i).T for i in seq_arr.T])
        return scipy.stats.entropy(probabilities, qk=None, base=None, axis=axis)

    # Load the dataset
    designs_df = df
    models = designs_df['Model'].unique()

    # Calculate entropies for each model and store in the dictionary
    entropy_dict = {}
    for model in models:
        # Filter sequences by model
        seq_list = designs_df[designs_df['Model'] == model]['AA_sequence'].tolist()
        
        # Calculate entropies for the list of sequences
        entropies = sequence_entropy(seq_list)
        
        # Store entropies in the dictionary with the model name as the key
        entropy_dict[model] = entropies

    # Convert the dictionary to a DataFrame
    entropy_df = pd.DataFrame(entropy_dict, index=[f"Position_{i}" for i in range(len(entropies))]).T

    # Create a new DataFrame by subtracting Pre-trained from Aligned for each model
    entropy_difference_df = pd.DataFrame()

    # Create new row that is row 2 - row 3 of mutation_df for mutational frequency difference between aligned and pre-trained ESM2
    entropy_difference_df['ESM2_Aligned-Pretrained'] = entropy_df.iloc[0] - entropy_df.iloc[1]
    print(entropy_difference_df.T)

    # Create a custom colormap where white is centered at 0
    min_score_2 = np.min(entropy_difference_df)
    max_score_2 = np.max(entropy_difference_df)
    midpoint = abs(min_score_2) / (max_score_2 - min_score_2)
    midpoint = midpoint.values[0]
    colors = [(0.0, '#7b3294'), (midpoint, 'white'), (1.0, '#008837')]
    custom_cmap_2 = LinearSegmentedColormap.from_list(cmap_name, colors)

    # Create figure
    fig, ax = plt.subplots(figsize=(slen/4.5, 10))

    # Plot the heatmap
    sns.heatmap(entropy_difference_df.T, cmap=custom_cmap_2, vmin=min_score_2, vmax=max_score_2, square=True, cbar=True, 
                yticklabels=entropy_difference_df.T.index, ax=ax, linewidths=0.5, linecolor='black')

    # Set the x-axis labels (representing WT)
    ax.set_xticks(np.arange(mutation_df.shape[1]) + 0.5)
    ax.set_xticklabels(list(WT))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', labeltop=True, labelbottom=False)
    plt.savefig(f'./logs/figures/{model_size}/Shannon_Entropy_All_Models.svg')
    plt.savefig(f'./logs/figures/{model_size}/Shannon_Entropy_All_Models.png')

    def hex_to_rgb(hex_str):
        """Convert a hex color string to a list of normalized RGB values."""
        hex_str = hex_str.lstrip('#')
        # Slightly reduce the intensity of the red to avoid high intensity reflections
        if hex_str.upper() == 'EF3B2C':
            return [int(hex_str[i:i+2], 16) * 0.85 / 255.0 for i in (0, 2, 4)]
        return [int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

    def create_color_scale_svg(color_map, filename):
        """Create an SVG file showing a linear color scale."""
        fig, ax = plt.subplots(figsize=(6, 1))
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=color_map), cax=ax, orientation='horizontal')
        cb.set_label('Mutation Frequency')
        plt.savefig(filename, format='svg')
        plt.close()

    # Provided mutation frequency data
    mutation_data = {'Aligned_ESM2': mutation_df.iloc[0], 'Pre_trained_ESM2': mutation_df.iloc[1]}

    # Define colors as normalized RGB lists
    base_grey = hex_to_rgb('#f0f0f0')  # Base color for no mutation
    target_red = hex_to_rgb('#ef3b2c')  # Adjusted target red for high mutation
    fad_blue = hex_to_rgb('#f0f0f0')   # Same as base color for no mutation

    # Normalize the mutation frequencies for each model
    normalized_data = {model: mutations / np.max(mutations) for model, mutations in mutation_data.items()}
    # print(normalized_data)

    # Generate separate PyMOL scripts for each model
    for model, normalized_mutations in normalized_data.items():
        pymol_script = f"load {WT_name}_AF3.pdb, {WT_name}_AF3\n"

        # Color each residue by interpolating from base_grey (low mutation) to target_red (high mutation)
        for i, freq in enumerate(normalized_mutations):
            # Use a non-linear interpolation to enhance red visibility at lower frequencies
            scaled_freq = 1 - (1 - freq) ** (2)  # scaling
            
            interp_color = [
                base + (target - base) * scaled_freq
                for base, target in zip(base_grey, target_red)
            ]
            pymol_script += (
                f"set_color color_{WT_name}_AF3_{i}, "
                f"[{interp_color[0]:.3f}, {interp_color[1]:.3f}, {interp_color[2]:.3f}]\n"
            )
            pymol_script += f"color color_{WT_name}_AF3_{i}, /{WT_name}_AF3//A/{i+1}\n"

        # Color FAD in the structure using the blue color
        pymol_script += (
            f"set_color fad_color, [{fad_blue[0]:.3f}, {fad_blue[1]:.3f}, {fad_blue[2]:.3f}]\n"
        )
        pymol_script += "color fad_color, resn FAD\n"

        pymol_script += "set_view (\
        0.980802536,    0.071432516,   -0.181449398,\
        0.075355798,   -0.997043610,    0.014813880,\
        -0.179854885,   -0.028202031,   -0.983288884,\
        0.000000000,    0.000000000, -169.733245850,\
        2.112551689,   -1.722059250,   -1.260961533,\
        133.818984985,  205.647506714,  -20.000000000 )\n"

        # Rendering settings
        pymol_script += "set ray_opaque_background, off\n"
        pymol_script += "set specular, 0\n"  # Disable specular reflections to avoid cyan highlights
        pymol_script += "set ray_trace_fog, 0\n"  # Disable ray trace fog
        pymol_script += "ray\n"
        pymol_script += f"png ray_color_{WT_name}_{model}.png, dpi=300\n"

        # Save color scale as SVG
        color_map = LinearSegmentedColormap.from_list("mutation_scale", [base_grey, target_red])
        create_color_scale_svg(color_map, f"logs/figures/{model_size}/color_scale_{model}.svg")

        script_filename = f"./logs/figures/{model_size}/color_{WT_name}_{model}.pml"
        with open(script_filename, "w") as file:
            file.write(pymol_script)

        print(f"PyMOL script saved as '{script_filename}'.")

    # ---------- Overlay Script for Aligned_ESM2 vs Pre_trained_ESM2 ----------
    overlay_script = f"load {WT_name}_AF3.pdb, {WT_name}_AF3\n"

    aligned = normalized_data['Aligned_ESM2']
    pretrain = normalized_data['Pre_trained_ESM2']

    for i, (a, p) in enumerate(zip(aligned, pretrain)):
        if a == 0 and p == 0:
            continue  # skip unmutated

        if a >= p:
            scaled = 1 - (1 - a)**2
            color = [
                base + (aligned_red[i] - base) * scaled
                for base, aligned_red[i] in zip(base_grey, aligned_red)
            ]
        else:
            scaled = 1 - (1 - p)**2
            color = [
                base + (pretrain_blue[i] - base) * scaled
                for base, pretrain_blue[i] in zip(base_grey, pretrain_blue)
            ]

        overlay_script += (
            f"set_color overlay_{WT_name}_AF3_{i}, "
            f"[{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}]\n"
            f"color overlay_{WT_name}_AF3_{i}, /{WT_name}_AF3//A/{i+1}\n"
        )

    overlay_script += (
        f"set_color fad_color, [{fad_blue[0]:.3f}, {fad_blue[1]:.3f}, {fad_blue[2]:.3f}]\n"
        "color fad_color, resn FAD\n"
        "set_view (\
        0.980802536,    0.071432516,   -0.181449398,\
        0.075355798,   -0.997043610,    0.014813880,\
        -0.179854885,   -0.028202031,   -0.983288884,\
        0.000000000,    0.000000000, -169.733245850,\
        2.112551689,   -1.722059250,   -1.260961533,\
        133.818984985,  205.647506714,  -20.000000000 )\n"
        "set ray_opaque_background, off\nset specular, 0\nset ray_trace_fog, 0\nray\n"
        f"png ray_color_{WT_name}_overlay.png, dpi=300\n"
    )

    overlay_filename = f"./logs/figures/{model_size}/color_{WT_name}_overlay.pml"
    with open(overlay_filename, "w") as f:
        f.write(overlay_script)

    # Copy the PDB file
    shutil.copyfile(f"{WT_name}_AF3.pdb", f'./logs/figures/{model_size}/{WT_name}_AF3.pdb')

    print(f"Overlay PyMOL script saved as '{overlay_filename}'.")
    print(f"PDB file copied to '{overlay_dir}/{WT_name}_AF3.pdb'.")

    ################################################################################################################
    # Define amino acid dictionary for tokenization, define WT for length of context window
    AAs = 'ACDEFGHIKLMNPQRSTVWY' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
    aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
    aa2ind.set_default_index(20) # set unknown charcterers to gap

    # Shared parameters for generating designs from pretrained, sft, and aligned models
    pretrained_ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")
    model_identifier = huggingface_identifier

    for num_muts in num_muts_list:
        # Do not generate loss curves and histogram
        create_loss_curves = False
        generate_histogram = False

        # Generate designs from pretrained, sft, and aligned models
        generate_pretrained_designs = True
        generate_sft_designs = True
        sft_model_exists = True
        sft_model_filepath = f'{filepath}/SFT_{huggingface_identifier}/version_{sft_version}' # ! update
        sft_model_name = f'SFT_{huggingface_identifier}_v0' # ! update

        # Generate designs from aligned model
        generate_aligned_designs = True
        rl_model_exists = True
        rl_model_filepath = f'{filepath}/PPO_{huggingface_identifier}/version_{version}' # ! update
        rl_model_name = f'ema_aligned_{huggingface_identifier}_v{version}_ep{ep}' # ! update

        ################################################################################################################

        if generate_pretrained_designs:
            saved_fixed_mutants_version = version
            # Generate designs with pretrained ESM2
            fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, pretrained_ESM2, model_identifier, tokenizer, filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
            print("Status: finished generating sequences with ESM2")

            # Save mutants from ESM2
            base_path = f'{filepath}/figures/{model_size}/'
            np.save(base_path + f'fixed_scores_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.npy', fixed_scores_np)
            with open(base_path + f'fixed_mutated_seqs_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.txt', 'w') as file:
                for seq in fixed_mutated_seqs:
                    file.write(seq + '\n')
        else:
            print('Skipping generating sequences from pretrained model')

        ################################################################################################################

        if generate_sft_designs:
            model_identifier = f"sft_{huggingface_identifier}"
            sft_ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
            state_dict = torch.load(f'{sft_model_filepath}/{sft_model_name}.pt', map_location=torch.device('cpu'))
            sft_ESM2.load_state_dict(state_dict)

            # Generate designs with sft ESM2
            sft_mutated_seqs, sft_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, sft_ESM2, model_identifier, tokenizer, filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
            print("Status: finished generating sequences with sft ESM2")

            # Save mutants from ESM2
            base_path = f'{filepath}/figures/{model_size}/'
            np.save(base_path + f'sft_scores_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.npy', sft_scores_np)
            with open(base_path + f'sft_mutated_seqs_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.txt', 'w') as file:
                for seq in sft_mutated_seqs:
                    file.write(seq + '\n')

        else:
            print('Skipping generating sequences from sft model')

        ################################################################################################################

        if generate_aligned_designs:
            model_identifier = f"aligned_{huggingface_identifier}"
            rl_ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
            state_dict = torch.load(f'{rl_model_filepath}/{rl_model_name}.pt', map_location=torch.device('cpu'))
            rl_ESM2.load_state_dict(state_dict)

            # Generate designs with rl ESM2
            rl_mutated_seqs, rl_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, rl_ESM2, model_identifier, tokenizer, filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
            print("Status: finished generating sequences with aligned ESM2")

            # Save mutants from ESM2
            base_path = f'{filepath}/figures/{model_size}/'
            np.save(base_path + f'rl_scores_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.npy', rl_scores_np)
            with open(base_path + f'rl_mutated_seqs_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.txt', 'w') as file:
                for seq in rl_mutated_seqs:
                    file.write(seq + '\n')

        else:
            print('Skipping generating sequences from aligned model')
        
        ################################################################################################################

    ################################################################################################################
    # Generate mutational extrapolation plot
    model_prefixes = ['fixed_', 'sft_', 'rl_']
    model_labels = {'fixed_': 'Pre-trained', 'sft_': 'SFT', 'rl_': 'PPO'}
    dir_filepath = f'./logs/figures/{model_size}'

    # Initialize dictionary to store score lists for each model prefix
    score_dict = {prefix: [] for prefix in model_prefixes}
    for model_prefix in model_prefixes:
        for num_muts in num_muts_list:
            try:
                filename = f'{model_prefix}scores_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.npy'
                Ensemble_of_scores = np.load(f'{dir_filepath}/{filename}')
                scores = np.median(Ensemble_of_scores, axis=0)
                # print('Number of scores:', len(scores))
                score_dict[model_prefix].append(scores)
            except:
                print(f'File not found at {dir_filepath}/{filename}')

    # Constants
    alpha = 1
    WT_linewidth = 2
    linewidth = 1
    fill = False

    # Create Figure with Two Subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

    # --- Subplot 1: Stripplot of scores for sequences ---

    # Process data for stripplot
    strip_rows = []
    for prefix in model_prefixes:
        label = model_labels[prefix]
        for i, num in enumerate(num_muts_list):
            sampled_scores = score_dict[prefix][i]
            for s in sampled_scores:
                strip_rows.append({'Number of Mutations': num, 'Predicted Log Fluorescence': s, 'Model': label})
    df_strip = pd.DataFrame(strip_rows)

    # Add predicted wild-type score line
    ax1.axhline(predicted_wt_score, color='black', linestyle='--', linewidth=WT_linewidth, label=f'Predicted {WT_name} score')

    # plot scores
    sns.stripplot(x='Number of Mutations', y='Predicted Log Fluorescence', hue='Model', data=df_strip, ax=ax1, jitter=True, dodge=True)
    sns.pointplot(x='Number of Mutations', y='Predicted Log Fluorescence', hue='Model',
                data=df_strip, ax=ax1, estimator=np.mean, errorbar=None,
                markers='_', dodge=0.55, linestyle='none', palette='dark:black',
                zorder=10)
    # Label plot
    ax1.set_xlabel('Number of Mutations', fontsize=12)
    ax1.set_ylabel('Predicted Log Fluorescence', fontsize=12)
    ax1.legend(fontsize=10)

    # --- Subplot 2: Lineplot of Win Rate vs. Mutational Regime ---

    # Compute win rate relative to pretrained model
    win_rate_rows = []
    baseline = 'fixed_'
    for prefix in model_prefixes:
        label = model_labels[prefix]
        for i, num in enumerate(num_muts_list):
            if prefix == baseline:
                continue
            else:
                sft_scores = score_dict[baseline][i]
                curr_scores = score_dict[prefix][i]
                win_rate = 100 * np.mean(curr_scores > sft_scores)
            win_rate_rows.append({'Number of Mutations': num, 'WinRate': win_rate, 'Model': label})
    df_win = pd.DataFrame(win_rate_rows)

    # Draw the baseline as a black dashed line at 50% and capture its handle.
    baseline_handle = ax2.axhline(50, color='black', linestyle='--', linewidth=WT_linewidth, label='Baseline')

    # Plot lines
    sns.lineplot(x='Number of Mutations', y='WinRate', hue='Model', data=df_win, marker='o', ax=ax2)

    # Label plot
    ax2.set_xlabel('Number of Mutations', fontsize=12)
    ax2.set_ylabel(f'Win Rate (%) vs. SFT ESM-2 ({model_size})', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{dir_filepath}/mutational_extrapolation_vs_pretrained.svg')
    plt.savefig(f'{dir_filepath}/mutational_extrapolation_vs_pretrained.png')
    plt.show()
