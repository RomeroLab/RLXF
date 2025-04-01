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


# load mutants for kde plot
esm2_models = ['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D']
version = 4
WT_name = "avGFP"

# create folder structure it doesn't exist
if not os.path.exists('logs/Aligning_SFT_ESM2s_wpPPO'):
    os.makedirs('logs/Aligning_SFT_ESM2s_wpPPO')

if not os.path.exists('logs/Aligning_SFT_ESM2s_wpPPO/figures'):
    os.makedirs('logs/Aligning_SFT_ESM2s_wpPPO/figures')
    os.makedirs('logs/Aligning_SFT_ESM2s_wpPPO/figures/8M')
    os.makedirs('logs/Aligning_SFT_ESM2s_wpPPO/figures/35M')
    os.makedirs('logs/Aligning_SFT_ESM2s_wpPPO/figures/150M')
    os.makedirs('logs/Aligning_SFT_ESM2s_wpPPO/figures/650M')

if not os.path.exists(f'logs/Aligning_SFT_ESM2s_wpPPO/version_{version}'):
    os.makedirs(f'logs/Aligning_SFT_ESM2s_wpPPO/version_{version}')
    os.makedirs(f'logs/Aligning_SFT_ESM2s_wpPPO/version_{version}/8M')
    os.makedirs(f'logs/Aligning_SFT_ESM2s_wpPPO/version_{version}/35M')
    os.makedirs(f'logs/Aligning_SFT_ESM2s_wpPPO/version_{version}/150M')
    os.makedirs(f'logs/Aligning_SFT_ESM2s_wpPPO/version_{version}/650M')


for huggingface_identifier in esm2_models:
    dir_filepath = f'logs/PPO_{huggingface_identifier}' # ! update
    model_size = huggingface_identifier.split('_')[2]

    # Load mutants from pretrained ESM2 650M
    fixed_scores_np = np.load(f'{dir_filepath}/version_{version}/fixed_{huggingface_identifier}_scores.npy') # ! update

    # Load sft mutants
    sft_scores_np = np.load(f'{dir_filepath}/version_{version}/sft_{huggingface_identifier}_scores.npy') # ! update

    # Load rl mutants
    rl_scores_np = np.load(f'{dir_filepath}/version_{version}/ema_aligned_{huggingface_identifier}_scores.npy') # ! update


    # Constants for the mean and standard deviation
    predicted_log_mean_wt_score = 4.1498 # this is predicted WT score # mean log exp score: 4.094413241
    alpha = 0.5

    # Plot histogram
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot histograms for the models7
    sns.kdeplot(np.median(fixed_scores_np, axis=0), color='#bdbdbd', ax=ax, linewidth=2.5, fill=True, alpha=alpha, label=f'Pre-trained ESM2 ({model_size})')
    sns.kdeplot(np.median(sft_scores_np, axis=0), color='#92c5de', ax=ax, linewidth=2.5, fill=True, alpha=alpha, label=f'SFT ESM2 ({model_size})')
    sns.kdeplot(np.median(rl_scores_np, axis=0), color='#2166ac', ax=ax, linewidth=2.5, fill=True, alpha=alpha, label=f'Aligned ESM2 ({model_size})')

    ax.axvline(predicted_log_mean_wt_score, color='black', linestyle='--', linewidth=1, label=f'Predicted {WT_name} score')

    ax.set_xlabel('Predicted Fluorescence', fontsize=12)
    ax.set_ylabel('Density', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_xlim(3.6, 4.2)
    # ax.set_ylim(0, 80)
    ax.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'logs/Aligning_SFT_ESM2s_wpPPO/figures/{model_size}/ppo_sft_pretrained_esm2_design_scores.svg')
    plt.savefig(f'logs/Aligning_SFT_ESM2s_wpPPO/figures/{model_size}/ppo_sft_pretrained_esm2_design_scores.png')



    # Load the data
    ema_filepath = f"logs/PPO_{huggingface_identifier}/version_4/ema_aligned_{huggingface_identifier}_mutated_designs_scores_ep2.csv"
    fixed_filepath = f"logs/PPO_{huggingface_identifier}/version_4/{huggingface_identifier}_fixed_mutated_designs_scores.csv"

    ema_df = pd.read_csv(ema_filepath)[["Sequence"]].head(30)
    ema_df["Model"] = f"Aligned_ESM2_{model_size}"
    fixed_df = pd.read_csv(fixed_filepath)[["Sequence"]].head(30)
    fixed_df['Model'] = f"Pretrained_ESM2_{model_size}"

    df = pd.concat([ema_df, fixed_df], ignore_index=True)
    df = df.rename(columns={'Sequence': 'AA_sequence'})
    df.head()

    # Define avGFP sequence
    base_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    sequence_length = len(base_sequence)



    # Initialize a dictionary to store mutation counts for each model
    mutation_counts = {model: np.zeros(sequence_length) for model in df['Model'].unique()}

    # Count mutations for each model
    for _, row in df.iterrows():
        model = row['Model']
        seq = row['AA_sequence']
        for i in range(sequence_length):
            if seq[i] != base_sequence[i]:
                mutation_counts[model][i] += 1

    # Create a dataframe for the heatmap
    mutation_df = pd.DataFrame(mutation_counts).T

    # Split the data into two parts
    first_half = mutation_df.iloc[:, :60]
    second_half = mutation_df.iloc[:, 60:]

    mutation_df.head()



    # Create a new DataFrame by subtracting Pre-trained from Aligned for each model
    mutation_difference_df = pd.DataFrame()

    # # Create new row that is row 0 - row 1 of mutation_df for mutational frequency difference between aligned and pre-trained VAE
    # mutation_difference_df['VAE_Aligned-Pretrained'] = mutation_df.iloc[2] - mutation_df.iloc[3]

    # Create new row that is row 2 - row 3 of mutation_df for mutational frequency difference between aligned and pre-trained ESM2
    mutation_difference_df['ESM2_Aligned-Pretrained'] = mutation_df.iloc[0] - mutation_df.iloc[1]

    # Display the resulting DataFrame
    print(mutation_difference_df.T)

    # Find min and max scores to properly set the colormap range
    min_score_1 = np.min(mutation_difference_df)
    max_score_1 = np.max(mutation_difference_df)

    # Calculate the position of 0 in the colormap
    midpoint = abs(min_score_1) / (max_score_1 - min_score_1)
    midpoint = midpoint.values[0]


    # Create a custom colormap, setting white as the center
    colors = [(0, '#B2182B'), (midpoint, 'white'), (1, '#2166AC')]
    cmap_name = 'custom'
    custom_cmap_1 = LinearSegmentedColormap.from_list(cmap_name, colors)

    # Create figure
    fig, ax = plt.subplots(figsize=(sequence_length/4.5, 10))

    # Plot the heatmap
    sns.heatmap(mutation_difference_df.T, cmap=custom_cmap_1, vmin=min_score_1, vmax=max_score_1, square=True, cbar=True, 
                yticklabels=mutation_difference_df.T.index, ax=ax, linewidths=0.5, linecolor='black')

    # Set the x-axis labels (representing base_sequence)
    ax.set_xticks(np.arange(mutation_df.shape[1]) + 0.5)
    ax.set_xticklabels(list(base_sequence))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', labeltop=True, labelbottom=False)

    plt.savefig(f'logs/Aligning_SFT_ESM2s_wpPPO/figures/{model_size}/Mutational_Freq_All_Models.svg')
    plt.savefig(f'logs/Aligning_SFT_ESM2s_wpPPO/figures/{model_size}/Mutational_Freq_All_Models.png')


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


    # # Load the dataset
    designs_df = df

    # List of models
    models = designs_df['Model'].unique()

    # Dictionary to store entropies for each model
    entropy_dict = {}

    # Calculate entropies for each model and store in the dictionary
    for model in models:
        # Filter sequences by model
        seq_list = designs_df[designs_df['Model'] == model]['AA_sequence'].tolist()
        
        # Calculate entropies for the list of sequences
        entropies = sequence_entropy(seq_list)
        
        # Store entropies in the dictionary with the model name as the key
        entropy_dict[model] = entropies

    # Convert the dictionary to a DataFrame
    entropy_df = pd.DataFrame(entropy_dict, index=[f"Position_{i}" for i in range(len(entropies))]).T

    # Display the DataFrame
    print(entropy_df)


    # Create a new DataFrame by subtracting Pre-trained from Aligned for each model
    entropy_difference_df = pd.DataFrame()

    # Create new row that is row 0 - row 1 of mutation_df for mutational frequency difference between aligned and pre-trained VAE
    #entropy_difference_df['VAE_Aligned-Pretrained'] = entropy_df.iloc[2] - #entropy_df.iloc[3]

    # Create new row that is row 2 - row 3 of mutation_df for mutational frequency difference between aligned and pre-trained ESM2
    entropy_difference_df['ESM2_Aligned-Pretrained'] = entropy_df.iloc[0] - entropy_df.iloc[1]

    # Display the resulting DataFrame
    print(entropy_difference_df.T)


    # Find min and max scores to properly set the colormap range
    min_score_2 = np.min(entropy_difference_df)
    max_score_2 = np.max(entropy_difference_df)

    # Calculate the position of 0 in the colormap
    midpoint = abs(min_score_2) / (max_score_2 - min_score_2)
    midpoint = midpoint.values[0]

    # Create a custom colormap where white is centered at 0
    colors = [
        (0.0, '#7b3294'),  # Purple
        (midpoint, 'white'),  # White at midpoint
        (1.0, '#008837')  # Green
    ]

    custom_cmap_2 = LinearSegmentedColormap.from_list(cmap_name, colors)

    # Create figure
    fig, ax = plt.subplots(figsize=(sequence_length/4.5, 10))

    # Plot the heatmap
    sns.heatmap(entropy_difference_df.T, cmap=custom_cmap_2, vmin=min_score_2, vmax=max_score_2, square=True, cbar=True, 
                yticklabels=entropy_difference_df.T.index, ax=ax, linewidths=0.5, linecolor='black')

    # Set the x-axis labels (representing base_sequence)
    ax.set_xticks(np.arange(mutation_df.shape[1]) + 0.5)
    ax.set_xticklabels(list(base_sequence))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', labeltop=True, labelbottom=False)

    plt.savefig(f'logs/Aligning_SFT_ESM2s_wpPPO/figures/{model_size}/Shannon_Entropy_All_Models.svg')
    plt.savefig(f'logs/Aligning_SFT_ESM2s_wpPPO/figures/{model_size}/Shannon_Entropy_All_Models.png')


    mutation_df.head()

    # Provided mutation frequency data
    mutation_data = {
        # 'Aligned_VAE': mutation_df.iloc[2],
        # 'Pre_trained_VAE': mutation_df.iloc[3],
        'Aligned_ESM2': mutation_df.iloc[0],
        'Pre_trained_ESM2': mutation_df.iloc[1]}

    mutation_data



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

    # Define colors as normalized RGB lists
    base_grey = hex_to_rgb('#f0f0f0')  # Base color for no mutation
    target_red = hex_to_rgb('#ef3b2c')  # Adjusted target red for high mutation
    fad_blue = hex_to_rgb('#f0f0f0')   # Same as base color for no mutation

    # Normalize the mutation frequencies for each model
    normalized_data = {model: mutations / np.max(mutations) for model, mutations in mutation_data.items()}
    print(normalized_data)

    # Generate separate PyMOL scripts for each model
    for model, normalized_mutations in normalized_data.items():
        pymol_script = "load avGFP_AF3.pdb, avGFP_AF3\n"

        # Color each residue by interpolating from base_grey (low mutation) to target_red (high mutation)
        for i, freq in enumerate(normalized_mutations):
            # Use a non-linear interpolation to enhance red visibility at lower frequencies
            scaled_freq = 1 - (1 - freq) ** (2)  # scaling
            
            interp_color = [
                base + (target - base) * scaled_freq
                for base, target in zip(base_grey, target_red)
            ]
            pymol_script += (
                f"set_color color_avGFP_AF3_{i}, "
                f"[{interp_color[0]:.3f}, {interp_color[1]:.3f}, {interp_color[2]:.3f}]\n"
            )
            pymol_script += f"color color_avGFP_AF3_{i}, /avGFP_AF3//A/{i+1}\n"

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
        pymol_script += f"png ray_color_avGFP_{model}.png, dpi=300\n"

        # Save color scale as SVG
        color_map = LinearSegmentedColormap.from_list("mutation_scale", [base_grey, target_red])
        create_color_scale_svg(color_map, f"logs/Aligning_SFT_ESM2s_wpPPO/figures/{model_size}/color_scale_{model}.svg")

        script_filename = f"logs/Aligning_SFT_ESM2s_wpPPO/figures/{model_size}/color_avGFP_{model}.pml"
        with open(script_filename, "w") as file:
            file.write(pymol_script)

        print(f"PyMOL script saved as '{script_filename}'.")



    ################################################################################################################

    # Define amino acid dictionary for tokenization, define WT for length of context window
    AAs = 'ACDEFGHIKLMNPQRSTVWY' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
    WT = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
    aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
    aa2ind.set_default_index(20) # set unknown charcterers to gap
    sequence_length = len(WT)
    num_EnsMLPs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(20)

    ################################################################################################################

    # Load reward models
    reward_models = []
    for i in range(num_EnsMLPs):
        model_name = f"reward_model_v{i}.ckpt"
        checkpoint_path = f"./reward_models/{model_name}"
        reward_model = MLP.load_from_checkpoint(checkpoint_path)
        for param in reward_model.parameters():
            param.requires_grad = False
        reward_models.append(reward_model)

    ################################################################################################################

    # Shared parameters for generating designs from pretrained, sft, and aligned models
    # ! update all below
    pretrained_ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{huggingface_identifier}")
    model_identifier = huggingface_identifier
    num_designs = 50
    num_muts_list = [5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
    high_conf_threshold = 0.9
    cum_prob_threshold = 0.25
    seed = 7028

    # Where to save figures and data
    filepath = './logs/Aligning_SFT_ESM2s_wpPPO'
    version = 4
    ep = 2

    for num_muts in num_muts_list:
        # Generate loss curves
        create_loss_curves = False

        # Generate histogram
        generate_histogram = False

        # Generate designs from pretrained model
        generate_pretrained_designs = True
        if generate_pretrained_designs == False:
            saved_fixed_mutants_version = 4 # May need to be changed if you already have designs from pretrained model in a different version file than 0
            saved_fixed_mutants_path = f'./logs/PPO_{huggingface_identifier}/' # ! update

        # Generate designs from sft model
        generate_sft_designs = True
        sft_model_exists = True
        sft_model_filepath = f'./logs/Aligning_SFT_ESM2s_wpPPO/version_4/{model_size}' # ! update
        sft_model_name = f'SFT_{huggingface_identifier}_v0' # ! update
        if generate_sft_designs == False:
            saved_sft_mutants_version = 0 # May need to be changed if you already have designs from pretrained model in a different version file than 0
            saved_sft_mutants_path = f'./logs/PPO_{huggingface_identifier}/version_4/{model_size}' # ! update


        # Generate designs from aligned model
        generate_aligned_designs = True
        rl_model_exists = True
        rl_model_filepath = f'{filepath}/version_{version}/{model_size}' # ! update
        # rl_model_name = f'rl_updated_esm2_t33_650M_UR50D_cuda_rank0_ep{ep}_v{version}'
        rl_model_name = f'ema_aligned_{huggingface_identifier}_v{version}_ep{ep}' # ! update

        ################################################################################################################

        if generate_pretrained_designs:
            saved_fixed_mutants_version = version
            # Generate designs with pretrained ESM2
            fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, pretrained_ESM2, model_identifier, tokenizer, filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
            print("Status: finished generating sequences with ESM2")

            # Save mutants from ESM2
            base_path = f'{filepath}/version_{version}/{model_size}/'
            np.save(base_path + f'fixed_scores_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.npy', fixed_scores_np)
            with open(base_path + f'fixed_mutated_seqs_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.txt', 'w') as file:
                for seq in fixed_mutated_seqs:
                    file.write(seq + '\n')

        else:
            print('Skipping generating sequences from pretrained model')

        if generate_histogram:
            # Load mutants from pretrained model if generating histograms
            fixed_scores_np = np.load(f'{filepath}/version_{version}/{model_size}/fixed_scores_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.npy')
            fixed_mutated_seqs = []
            with open(f'{filepath}/version_{version}/{model_size}/fixed_mutated_seqs_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.txt', 'r') as file:
                fixed_mutated_seqs = file.read().splitlines()
            df_fixed = generate_df(fixed_mutated_seqs, np.median(fixed_scores_np, axis=0), WT)
            df_fixed.to_csv(f'{filepath}/version_{version}/{model_size}/fixed_mutated_designs_scores_mutations_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.csv', index=False)

        ################################################################################################################

        if generate_sft_designs:
            model_identifier = f"sft_{model_identifier}"
            sft_ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
            state_dict = torch.load(f'{sft_model_filepath}/{sft_model_name}.pt', map_location=torch.device('cpu'))
            sft_ESM2.load_state_dict(state_dict)

            # Generate designs with sft ESM2
            sft_mutated_seqs, sft_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, sft_ESM2, model_identifier, tokenizer, filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
            print("Status: finished generating sequences with sft ESM2")

            # Save mutants from ESM2
            base_path = f'{filepath}/version_{version}/{model_size}/'
            np.save(base_path + f'sft_scores_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.npy', sft_scores_np)
            with open(base_path + f'sft_mutated_seqs_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.txt', 'w') as file:
                for seq in sft_mutated_seqs:
                    file.write(seq + '\n')

        else:
            print('Skipping generating sequences from sft model')

        if generate_histogram:
            # Load mutants from pretrained model if generating histograms
            sft_scores_np = np.load(f'{filepath}/version_{version}/{model_size}/sft_scores_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.npy')
            sft_mutated_seqs = []
            with open(f'{filepath}/version_{version}/{model_size}/sft_mutated_seqs_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.txt', 'r') as file:
                sft_mutated_seqs = file.read().splitlines()
            df_sft = generate_df(sft_mutated_seqs, np.median(sft_scores_np, axis=0), WT)
            df_sft.to_csv(f'{filepath}/version_{version}/{model_size}/sft_mutated_designs_scores_mutations_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.csv', index=False)

        ################################################################################################################

        if generate_aligned_designs:
            model_identifier = f"aligned_{model_identifier}"
            rl_ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{huggingface_identifier}")
            state_dict = torch.load(f'{rl_model_filepath}/{rl_model_name}.pt', map_location=torch.device('cpu'))
            rl_ESM2.load_state_dict(state_dict)

            # Generate designs with rl ESM2
            rl_mutated_seqs, rl_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, rl_ESM2, model_identifier, tokenizer, filepath, ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, seed)
            print("Status: finished generating sequences with aligned ESM2")

            # Save mutants from ESM2
            base_path = f'{filepath}/version_{version}/{model_size}/'
            np.save(base_path + f'rl_scores_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.npy', rl_scores_np)
            with open(base_path + f'rl_mutated_seqs_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.txt', 'w') as file:
                for seq in rl_mutated_seqs:
                    file.write(seq + '\n')

        else:
            print('Skipping generating sequences from aligned model')

        if generate_histogram:
            model_identifier = f"aligned_{model_identifier}"

            # Load mutants
            rl_scores_np = np.load(f'{filepath}/version_{version}/{model_size}/rl_scores_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.npy')
            rl_mutated_seqs = []
            with open(f'{filepath}/version_{version}/{model_size}/rl_mutated_seqs_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.txt', 'r') as file:
                rl_mutated_seqs = file.read().splitlines()
            df_rl = generate_df(rl_mutated_seqs, np.median(rl_scores_np, axis=0), WT)
            df_rl.to_csv(f'{filepath}/version_{version}/{model_size}/rl_mutated_designs_scores_mutations_HCthreshold_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.csv', index=False)
        
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
                plt.savefig(f'{filepath}/version_{version}/{model_size}/design_scores_{model_identifier}_ep{ep}_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.svg')
                plt.savefig(f'{filepath}/version_{version}/{model_size}/design_scores_{model_identifier}_ep{ep}_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.png')

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
                plt.savefig(f'{filepath}/version_{version}/{model_size}/design_scores_{model_identifier}_ep{ep}_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.svg')
                plt.savefig(f'{filepath}/version_{version}/{model_size}/design_scores_{model_identifier}_ep{ep}_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.png')

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
                plt.savefig(f'{filepath}/version_{version}/{model_size}/design_scores_{model_identifier}_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.svg')
                plt.savefig(f'{filepath}/version_{version}/{model_size}/design_scores_{model_identifier}_HC{high_conf_threshold}_CP{cum_prob_threshold}_{num_muts}muts.png')

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



    # datatypes
    # list_of_num_muts = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,25,30,35,40,45,50,55,60]
    list_of_num_muts = num_muts_list
    model_prefixes = ['fixed_', 'sft_', 'rl_']
    model_labels = {'fixed_': 'Pre-trained', 'sft_': 'SFT', 'rl_': 'PPO'}
    dir_filepath = f'./logs/Aligning_SFT_ESM2s_wpPPO/version_4/{model_size}'

    # Initialize dictionary to store score lists for each model prefix
    score_dict = {prefix: [] for prefix in model_prefixes}

    for model_prefix in model_prefixes:
        for num_muts in list_of_num_muts:
            try:
                filename = f'{model_prefix}scores_HCthreshold_HC0.9_CP0.25_{num_muts}muts.npy'
                Ensemble_of_scores = np.load(f'{dir_filepath}/{filename}')
                scores = np.median(Ensemble_of_scores, axis=0)
                # print('Number of scores:', len(scores))
                score_dict[model_prefix].append(scores)
            except:
                print(f'File not found at {dir_filepath}/{filename}')

    # Constants
    alpha = 1
    predicted_wt_score = 4.1498  # predicted WT score
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
        for i, num in enumerate(list_of_num_muts):
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

    # Compute win rate relative to SFT
    win_rate_rows = []
    baseline = 'fixed_'
    for prefix in model_prefixes:
        label = model_labels[prefix]
        for i, num in enumerate(list_of_num_muts):
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
    plt.savefig(f'logs/Aligning_SFT_ESM2s_wpPPO/figures/{model_size}/mutational_extrapolation_vs_pretrained.svg')
    plt.savefig(f'logs/Aligning_SFT_ESM2s_wpPPO/figures/{model_size}/mutational_extrapolation_vs_pretrained.png')
    plt.show()


