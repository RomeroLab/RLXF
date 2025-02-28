#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Importing Packages
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
from torchtext import vocab # This package can give problems sometimes, it may be necessary to downgrade to a specific version
from pytorch_lightning.loggers import CSVLogger
from random import choice
import seaborn as sns
import random
from random import choice
import matplotlib.pyplot as plt
from sklearn import metrics
import torchmetrics
import enum
import csv
import os
import pickle
from sklearn.model_selection import train_test_split
from Bio import AlignIO
import math
import pathlib
import warnings
from reward_model import CNN
from MLP import MLP
from transformers import AutoModelForMaskedLM, AutoTokenizer
from matplotlib.colors import LinearSegmentedColormap
# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig
# from esm.sdk.forge import ESM3ForgeInferenceClient
from conv_vae_model import ConvVAE

# # Training on GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Set up Amino Acid Dictionary of Indices
# AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
# WT = "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
# aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
# aa2ind.set_default_index(20) # set unknown charcterers to gap
    
# #For Optuna Sweep
# def get_sft_version_file(sft_version):
#     mapping = {
#         0: "SFT_ESM2_NB_model_v1081_logger_v0.pt",
#         1: "SFT_ESM2_NB_model_v1163_logger_v0.pt",
#         2: "SFT_ESM2_NB_model_v1174_logger_v0.pt",
#         3: "SFT_ESM2_NB_model_v1209_logger_v0.pt",
#         4: "SFT_ESM2_NB_model_v1229_logger_v0.pt",
#         5: "SFT_ESM2_NB_model_v1260_logger_v0.pt",
#         6: "SFT_ESM2_NB_model_v1302_logger_v0.pt",
#         7: "SFT_ESM2_NB_model_v854_logger_v0.pt",

#     }
#     return mapping.get(sft_version, None)

# # Loading VAE
# def load_vae_model(checkpoint_path):
#     """
#     Load a ConvVAE model from a checkpoint.
#     Args:
#         checkpoint_path (str): Path to the saved checkpoint file.
#     Returns:
#         ConvVAE: Loaded model.
#     """
#     vae_model = ConvVAE.load_from_checkpoint(checkpoint_path)
#     return vae_model

# # Loading reward model
# def load_reward_model(checkpoint_path):
#     """
#     Load a reward model from a checkpoint.
#     Args:
#         checkpoint_path (str): Path to the saved checkpoint file.
#     Returns:
#         Reward Model (nn.Module)
#     """
#     reward_model = MLP.load_from_checkpoint(checkpoint_path)
#     return reward_model

# # Detect hamming distance between proteins
# def hamming_distance_vae_training(s1, s2):
#     """Calculate the Hamming distance between two strings, ignoring gaps,
#        and return the positions of mutations along with the original and new amino acids."""
#     if len(s1) != len(s2):
#         raise ValueError("Sequences must be of the same length")

#     distance = 0
#     mutation_info = []
    
#     # Filter out positions where either sequence has a gap
#     filtered_pairs = [(i, el1, el2) for i, (el1, el2) in enumerate(zip(s1, s2)) if el1 != '-' and el2 != '-']

#     for i, el1, el2 in filtered_pairs:
#         if el1 != el2:
#             distance += 1
#             mutation_detail = {
#                 'pos': i+1,
#                 'orig_aa': el1,
#                 'mut_aa': el2
#             }
#             mutation_info.append(mutation_detail)
            
#     return distance, mutation_info

# # Define function for passing in tensor proteins through VAE encoder and decoder
# def Using_VAE(VAE, batch):
#     """
#     Computes probabilities for a given batch of sequences using VAE
#     Args:
#         batch (torch.Tensor): A LongTensor of shape (b, L) with b = batch size and L = sequence length.
#     Returns:
#         probabilities (torch.FloatTensor):
#     """
#     with torch.no_grad():  # We do not want training to occur during scoring
#         VAE.eval()
#         torch.manual_seed(42)
#         # Pass the batch through the model to get the z_mean, z_log_var, encoded, and decoded tensors
#         z_mean, z_log_var, encoded, decoded = VAE(batch)
#         VAE.train()
        
#     return encoded, decoded

# # Creating initial state (Inference with Pre-trained VAE)
# def decoding(VAE, batch):
#     """
#     Computes probabilities matrices (states) for given batch of sequences using VAEs
#     Args:
#         batch (torch.Tensor): (b, latent_dim)
#             A LongTensor of shape (b, latent_dim) with b = batch size and latent_dim = dimensions of latent space in VAE
#         pre_trained_VAE (pytorch model): pretrained MSA VAE that remains frozen during training
#     Returns:
#         states (torch.FloatTensor): (b, dict, L)
#             batch of state matrices for each protein with column corresponding to amino acid index of protein normalized to 1 (probabilities)
#     """
#     with torch.no_grad():  # We do not want training to occur during scoring
#         VAE.eval()
#         batch = batch.to(next(VAE.parameters()).device)
#         logits = VAE.decoder(batch)  # Use z_mean instead of encoded to remove stochastic reparameterization
#         VAE.train()
#     return logits

# # Adding noise to CreiLOV representation in vae latent space to create dataset for RLXF
# def generate_and_evaluate_mutants_vae_training(vae_model, WT, AAs, scale=0.9111111111111111, num_samples=1000):
#     # torch.manual_seed(42)  # For reproducibility
    
#     # Load parameters
#     CreiLOV_representation = torch.load('./CreiLOV_representation.pt')
#     std_difference_from_best_single_mutant = np.load('./singlemutant_std_est.npy')
#     latent_dim = CreiLOV_representation.shape[0]

#     # Generate noise
#     mean = torch.tensor(0.0)
#     noise_std = torch.tensor(std_difference_from_best_single_mutant * scale)
#     noise = torch.normal(mean=mean,
#                          std=noise_std,
#                          size=(num_samples, latent_dim))

#     # Create new latent representations by adding the noise to the original WT representation
#     new_representations = CreiLOV_representation + noise

#     # Send sequence into VAE to find average hamming distance
#     decoded_SMs = decoding(vae_model, new_representations)

#     # Identify the max values for each amino acid positions
#     _, max_indices = torch.max(decoded_SMs, dim=1)

#     # Create a reverse mapping manually based on the AAs string
#     ind2aa = {i: aa for i, aa in enumerate(AAs)}

#     # Convert the tensor to a list of indices and decode the sequences
#     decoded_sequences = [''.join([ind2aa[idx] for idx in batch]) for batch in max_indices.tolist()]

#     # Calculate metrics
#     unique_sequences = set(decoded_sequences)
#     num_unique_sequences = len(unique_sequences)
#     hd_list = []
#     unique_mutations = set()
#     unique_positions = set()

#     for decoded_sequence in decoded_sequences:
#         hd, mutation_info = hamming_distance_vae_training(WT, decoded_sequence)
#         hd_list.append(hd)

#         for mut in mutation_info:
#             unique_mut = f"{mut['pos']}_{mut['orig_aa']}_{mut['mut_aa']}"
#             unique_mutations.add(unique_mut)
#             unique_positions.add(mut['pos'])

#     # Compile metrics
#     avg_hd = sum(hd_list) / len(hd_list) if hd_list else 0
#     max_hd = max(hd_list) if hd_list else 0
#     mutation_diversity = len(unique_mutations)
#     pos_diversity = len(unique_positions)

#     return new_representations, {
#         'average_hamming_distance': avg_hd,
#         'maximum_hamming_distance': max_hd,
#         'mutation_diversity': mutation_diversity,
#         'position_diversity': pos_diversity,
#         'number_of_unique_sequences': num_unique_sequences}, decoded_sequences, hd_list

# # ProtRepDataset is a data handling class for getting protein representations from .pt file
# class ProtRepDataset(torch.utils.data.Dataset):
#     """A custom PyTorch dataset for protein representations from pre-trained VAE"""
#     def __init__(self, rl_updated_vae, WT, AAs, num_samples=1000, target_hd=5, initial_scale=None, current_epoch=0, version=None):
#         # Store parameters as instance attributes
#         self.vae_model = rl_updated_vae
#         self.WT = WT
#         self.AAs = AAs
#         self.num_samples = num_samples
#         self.target_hd = target_hd
#         self.initial_scale = None
#         self.current_epoch = current_epoch
#         self.version = version

#         # Update the save_dir to include logger version information if available
#         if self.version is not None and self.version is not None:
#             version_path = f'version_{self.version}'
#         else:
#             version_path = 'version_unknown'
#         self.save_dir = os.path.join('./datavae', version_path)
#         os.makedirs(self.save_dir, exist_ok=True)  # Ensure the directory exists
        
#         # Initialize the dataset with data for the current epoch
#         self.data, self.metrics, self.scale = self.create_data_for_epoch()
#         self.save_data()

#     def create_data_for_epoch(self):
#         closest_dataset, closest_dataset_metrics, closest_scale = find_closest_average_hd(self.vae_model, self.WT, self.AAs, self.num_samples, self.target_hd, self.initial_scale)
#         # self.initial_scale = closest_scale
#         return closest_dataset, closest_dataset_metrics, closest_scale

#     def save_data(self):
#         # Where to save data
#         data_filename = f'data_epoch_{self.current_epoch}.pt'
#         metrics_and_scale_filename = f'metrics_and_scale_epoch_{self.current_epoch}.txt'
#         data_path = os.path.join(self.save_dir, data_filename)
#         metrics_and_scale_path = os.path.join(self.save_dir, metrics_and_scale_filename)

#         # Saving data, metrics, and scale
#         torch.save(self.data, data_path)

#         # Saving metrics and scale to the same text file
#         with open(metrics_and_scale_path, 'w') as file:
#             for key, value in self.metrics.items():
#                 file.write(f"{key}: {value}\n")
#             file.write(f"scale: {self.scale}\n")
        
#         print(f"Data: {data_path}, metrics and scale: {metrics_and_scale_path}.")

#     def __len__(self):
#         # Return the number of items in the dataset
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         # Return the item at the given index
#         return self.data[idx]

# # Dataloader for RLXF with VAE
# class ProtDataModule(pl.LightningDataModule):
#     """A PyTorch Lightning Data Module to handle data splitting for protein representations"""

#     def __init__(self, rl_updated_vae, WT, AAs, batch_size, seed, num_samples=1000, target_hd=5, initial_scale=None, current_epoch=0, version=None):
#         super().__init__()
#         self.rl_updated_vae = rl_updated_vae
#         self.WT = WT
#         self.AAs = AAs
#         self.batch_size = batch_size
#         self.num_samples = num_samples
#         self.target_hd = target_hd
#         self.initial_scale = None
#         self.current_epoch = current_epoch
#         self.version = version
#         self.seed = seed


#     # This can help with loading data
#     def prepare_data(self):
#         pass

#     def setup(self, stage=None):
#         # Assign train/val datasets for use in dataloaders
#         if stage == 'fit' or stage is None:
#             # Updated to pass the necessary arguments to ProtRepDataset
#             self.train_ds = ProtRepDataset(
#                 self.rl_updated_vae, 
#                 self.WT, 
#                 self.AAs, 
#                 self.num_samples, 
#                 self.target_hd, 
#                 self.initial_scale,
#                 self.current_epoch,
#                 self.version
#             )
            
#     def train_dataloader(self):
#         # def seed_worker(worker_id):
#         #     worker_seed = torch.initial_seed() % 2**32  # Compute a seed for the worker based on the initial seed of the torch Generator
#         #     np.random.seed(worker_seed)  # Set NumPy's random seed based on the worker seed
#         #     random.seed(worker_seed)  # Set Python's built-in random module's seed
        
#         # generator = torch.Generator()  # Create a new torch Generator
#         # generator.manual_seed(self.seed)  # Manually seed the generator with the predefined seed from the class

#         # return data_utils.DataLoader(
#         #     self.train_ds, # The dataset to load, in this case, the generated sequences
#         #     batch_size=self.batch_size, # The number of samples in each batch to load
#         #     shuffle=True, # Enable shuffling to randomize the order of data before each epoch, enhancing training effectiveness
#         #     worker_init_fn = seed_worker, # Function to initialize each worker's seed to ensure reproducibility across runs
#         #     generator = generator,  # Specify the generator used for random number generation in shuffling
#         #     num_workers=32, # The number of subprocesses to use for data loading. More workers can increase the speed of data loading
#         #     pin_memory=True
#         # )

#         return data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8)



# # ProtRepDataset is a data handling class for getting protein representations from .pt file
# class ProtRepDataset_0(torch.utils.data.Dataset):
#     """A custom PyTorch dataset for protein representations from pre-trained VAE"""

#     def __init__(self, protein_reps):
#         self.protein_reps = protein_reps

#     def __getitem__(self, idx):
#         rep = self.protein_reps[idx]  # Extract protein representation at index idx
#         return rep  # Return protein representation

#     def __len__(self):
#         return len(self.protein_reps)

# # ProtDataModule randomizes protein representation batches per epoch of PPO
# class ProtDataModule_0(pl.LightningDataModule):
#     """A PyTorch Lightning Data Module to handle data splitting"""

#     def __init__(self, protein_reps, batch_size):
#         # Call the __init__ method of the parent class
#         super().__init__()

#         # Store the batch size
#         self.protein_reps = protein_reps
#         self.batch_size = batch_size

#     # This can help with loading data to GPU
#     def prepare_data(self):
#         pass

#     # Assign datasets for use in dataloaders
#     def setup(self, stage=None):
#         if stage == 'fit' or stage is None:
#             self.train_ds = ProtRepDataset_0(self.protein_reps)
            
#     def train_dataloader(self):
#         return data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8)

# # Enforce MA- and -KALA to protein where MSA had high probability of gaps
# def adjust_designs(mutant_designs):
#     adjusted_sequences = []
    
#     for seq in mutant_designs:
#         # Convert string to list of characters for easier manipulation
#         seq_list = list(seq)
        
#         # Ensure the N terminus begins with Methionine (M) and Alanine (A)
#         # Assuming 'M' is represented by '10' and 'A' is represented by '0'
#         seq_list[0] = 'M'  # Replace with correct letter or code for Methionine
#         seq_list[1] = 'A'  # Replace with correct letter or code for Alanine

#         # Ensure the C terminus ends with KALA
#         # Assuming 'K' is '8', 'A' is '0', 'L' is '9', 'A' is '0'
#         seq_list[-4] = 'K'  # Replace with correct letter or code for Lysine
#         seq_list[-3] = 'A'  # Replace with correct letter or code for Alanine
#         seq_list[-2] = 'L'  # Replace with correct letter or code for Leucine
#         seq_list[-1] = 'A'  # Replace with correct letter or code for Alanine

#         # Revert the list of characters back to a string
#         adjusted_seq = ''.join(seq_list)
#         adjusted_sequences.append(adjusted_seq)
#     return adjusted_sequences

# # SeqDataset is a data handling class. I convert amino acid sequences to torch tensors for model input
# class SeqDataset(torch.utils.data.Dataset):
#     """A custom PyTorch dataset for protein sequence-function data"""

#     def __init__(self, data_frame):
#         self.data_df = data_frame

#     def __getitem__(self, idx):
#         sequence = torch.tensor(aa2ind(list(self.data_df.Sequence.iloc[idx]))).to(device) # Extract sequence at index idx
#         return sequence

#     def __len__(self):
#         return len(self.data_df)

# def convert_and_score_sequences(sequences_list, reward_models):
#     # Convert the list of sequences into a DataFrame
#     sequences_df = pd.DataFrame(sequences_list, columns=['Sequence'])

#     # Initialize your custom dataset with the DataFrame
#     sequence_dataset = SeqDataset(sequences_df)

#     # Initialize the DataLoader
#     sequence_loader = data_utils.DataLoader(sequence_dataset, batch_size=1, shuffle=False)

#     # Prepare a tensor to store the scores
#     scores_tensor = torch.zeros(len(sequences_list), dtype=torch.float32)

#     # Ensure all models are in evaluation mode and no need for gradients
#     for model in reward_models:
#         model.eval()

#     with torch.no_grad():
#         for i, sequence_tensor in enumerate(sequence_loader):
#             sequence_tensor = sequence_tensor.squeeze(0)  # Remove batch dimension

#             # Collect scores from each model
#             model_scores = []
#             for model in reward_models:
#                 # Predict the score for the sequence
#                 score = model.predict(sequence_tensor.unsqueeze(0))[0][0]
#                 model_scores.append(score)

#             # Compute the mean score across all models
#             median_score = torch.tensor(model_scores).median()
#             scores_tensor[i] = median_score

#     return scores_tensor.tolist()

# def save_metrics_to_csv(version, metrics):
#     # Define the directory path and file path for the metrics
#     dir_path = f'./designs/vae_designs/version_{version}'
#     os.makedirs(dir_path, exist_ok=True)  # Create the directory if it does not exist
#     metrics_file_path = os.path.join(dir_path, 'mutant_metrics.csv')
    
#     # Save metrics to CSV file
#     with open(metrics_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         # Write the header with the keys of the metrics dictionary
#         writer.writerow(metrics.keys())
#         # Write the values of the metrics dictionary
#         writer.writerow(metrics.values())

#     print(f"Metrics for mutants saved to {metrics_file_path}")

# def identify_mutations(wt_sequence, mutant_sequence):
#     mutations = []
#     for i, (wt_res, mut_res) in enumerate(zip(wt_sequence, mutant_sequence)):
#         if wt_res != mut_res:
#             mutations.append(f"{wt_res}{i+1}{mut_res}")  # Assuming 1-based numbering
#     return ', '.join(mutations)

# def save_sorted_designs_to_csv(version, wt_sequence, adjusted_mutant_designs, scores):
#     # Sort designs by scores
#     sorted_designs_scores = sorted(zip(adjusted_mutant_designs, scores), key=lambda x: x[1], reverse=True)
#     sorted_designs, sorted_scores = zip(*sorted_designs_scores)

#     # Define the directory path based on the version
#     dir_path = f'./designs/vae_designs/version_{version}'
#     os.makedirs(dir_path, exist_ok=True)

#     # Save sorted designs, mutations, and scores to CSV file
#     designs_file_path = os.path.join(dir_path, 'sorted_mutant_designs_scores_mutations.csv')
#     with open(designs_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Adjusted Design', 'Score', 'Mutations'])  # Write the header
#         for design, score in zip(sorted_designs, sorted_scores):
#             mutations = identify_mutations(wt_sequence, design)
#             writer.writerow([design, score, mutations])

#     print(f"Mutant designs sorted by/with scores and mutations data saved to {designs_file_path}")

# # Generate dataset with close to hamming distance of 5 from CreiLOV
# def find_closest_average_hd(vae_model, WT, AAs, num_samples, target_hd=5, initial_scale=None, depth=0, closest_avg_hd=None, closest_dataset=None, closest_dataset_metrics=None):
#     # Initialize variables to track the closest dataset
#     # closest_dataset = None
#     # closest_dataset_metrics = None
#     closest_scale = initial_scale

#     if depth < 5:
#         if closest_avg_hd is not None:
#             if (4.95 <= closest_avg_hd <= 5.05):
#                 scale_adjustment = 0.01
#                 # print(scale_adjustment)
#             if (4.9 <= closest_avg_hd <= 5.1):
#                 scale_adjustment = 0.05
#                 # print(scale_adjustment)
#             else:
#                 scale_adjustment = 0.25
#                 # print(scale_adjustment)
#         else:
#             scale_adjustment = 0.025
#             # print(scale_adjustment)
    
#     else:
#         scale_adjustment = 0.025
        
#     # If initial_scale is provided, adjust scale factors around it
#     if initial_scale is not None:
#         if closest_avg_hd < 5.0:
#             lower_bound = closest_scale
#             upper_bound = closest_scale * (1 + scale_adjustment)
#         else:
#             lower_bound = closest_scale * (1 - scale_adjustment)
#             upper_bound = closest_scale

#         scale_factors = np.linspace(float(lower_bound), float(upper_bound), num=30*(depth+1))
#         # print('New scale factors')
#         # print('New scale_factors:', scale_factors)
    
#     else:
#         # Define default scale factors to test if initial_scale is None
#         closest_avg_hd = float('inf')
#         # scale_factors = np.linspace(0.9, 1, num=20*(depth+1))
#         scale_factors = np.linspace(0.9, 1.2, num=30*(depth+1))
#         # print('OG scale_factors:')
    
#     # Run the generation and evaluation 10 times with different scale factors
#     for scale in scale_factors:
#         dataset, metrics, _, _ = generate_and_evaluate_mutants_vae_training(vae_model, WT, AAs, scale=scale, num_samples=num_samples)
#         avg_hd = metrics['average_hamming_distance']
#         # if depth > 10:
#         #     print(avg_hd)
        
#         # Check if this metrics is closer to the target average Hamming distance
#         if abs(avg_hd - target_hd) < abs(closest_avg_hd - target_hd):
#             closest_avg_hd = avg_hd # Save best current average hamming distance metric
#             closest_dataset = dataset # Save dataset with closest to desired average hamming distance metric
#             closest_dataset_metrics = metrics # Save metrics for dataset
#             closest_scale = scale # Save scale of noise to create dataset

#     # # Log the scale and HD after evaluating all scale factors
#     # print(f"After evaluating scale factors, closest scale: {closest_scale} with average HD: {closest_avg_hd}")

#     # If the depth is less than 10 and the closest average HD is not within the desired range, recurse
#     if depth < 10 and not (4.975 <= closest_avg_hd <= 5.025):
#         print(f"Closest scale: {closest_scale} with aver HD: {closest_avg_hd}")
#         return find_closest_average_hd(vae_model, WT, AAs, num_samples, target_hd, closest_scale, depth+1, closest_avg_hd, closest_dataset, closest_dataset_metrics)
#     else:
#         # Print the closest scale for debugging or logging purposes
#         print(f"Closest scale: {closest_scale} with aver HD: {closest_avg_hd}")
#         return closest_dataset, closest_dataset_metrics, closest_scale # Return the dataset, metrics, and scale factor


# def generate_and_evaluate_mutants_2(seed, vae_model, WT, AAs, scale=0.9111111111111111, num_samples=1000):
#     torch.manual_seed(seed)  # For reproducibility

#     # Load parameters
#     CreiLOV_representation = torch.load('./CreiLOV_representation.pt')
#     std_difference_from_best_single_mutant = np.load('./singlemutant_std_est.npy')
#     latent_dim = CreiLOV_representation.shape[0]

#     # Generate noise
#     noise = torch.normal(mean=0, std=std_difference_from_best_single_mutant * scale, size=(num_samples, latent_dim))

#     # Create new latent representations by adding the noise to the original WT representation
#     new_representations = CreiLOV_representation + noise

#     # Decode sequences
#     decoded_SMs = decoding(vae_model, new_representations)
#     _, max_indices = torch.max(decoded_SMs, dim=1)

#     # Create a reverse mapping manually based on the AAs string
#     ind2aa = {i: aa for i, aa in enumerate(AAs)}

#     # Convert tensor to list of indices and decode sequences
#     decoded_sequences = [''.join([ind2aa[idx] for idx in batch]) for batch in max_indices.tolist()]

#     # Initialize metrics
#     unique_sequences = set()
#     unique_mutations = set()
#     unique_positions = set()
#     filtered_representations = []
#     filtered_sequences = []

#     # Evaluate sequences
#     for idx, decoded_sequence in enumerate(decoded_sequences):
#         hd, mutation_info = hamming_distance_vae_training(WT, decoded_sequence)

#         # Only consider sequences with exactly 5 mutations
#         if hd == 5:
#             if decoded_sequence not in unique_sequences:
#                 unique_sequences.add(decoded_sequence)
#                 filtered_representations.append(new_representations[idx])
#                 filtered_sequences.append(decoded_sequence)

#                 for mut in mutation_info:
#                     unique_mut = f"{mut['pos']}_{mut['orig_aa']}_{mut['mut_aa']}"
#                     unique_mutations.add(unique_mut)
#                     unique_positions.add(mut['pos'])

#             # Stop when 1000 unique sequences are found
#             if len(unique_sequences) >= 1000:
#                 break

#     # Compile metrics
#     mutation_diversity = len(unique_mutations)
#     pos_diversity = len(unique_positions)

#     return filtered_representations, {
#         'mutation_diversity': mutation_diversity,
#         'position_diversity': pos_diversity,
#         'number_of_unique_sequences': len(unique_sequences)}, filtered_sequences

# def generate_and_evaluate_mutants_100000(seed, vae_model, WT, AAs, scale=0.9111111111111111, num_samples=1000):
#     torch.manual_seed(seed)  # For reproducibility

#     # Load parameters
#     CreiLOV_representation = torch.load('./data/for_RLXF/CreiLOV_representation.pt')
#     std_difference_from_best_single_mutant = np.load('./data/for_RLXF/singlemutant_std_est.npy')
#     latent_dim = CreiLOV_representation.shape[0]

#     # Generate noise
#     noise = torch.normal(mean=0, std=std_difference_from_best_single_mutant * scale, size=(num_samples, latent_dim))

#     # Create new latent representations by adding the noise to the original WT representation
#     new_representations = CreiLOV_representation + noise

#     # Decode sequences
#     decoded_SMs = decoding(vae_model, new_representations)
#     _, max_indices = torch.max(decoded_SMs, dim=1)

#     # Create a reverse mapping manually based on the AAs string
#     ind2aa = {i: aa for i, aa in enumerate(AAs)}

#     # Convert tensor to list of indices and decode sequences
#     decoded_sequences = [''.join([ind2aa[idx] for idx in batch]) for batch in max_indices.tolist()]

#     # Initialize metrics
#     unique_sequences = set()
#     unique_mutations = set()
#     unique_positions = set()
#     filtered_representations = []
#     filtered_sequences = []

#     # Evaluate sequences
#     for idx, decoded_sequence in enumerate(decoded_sequences):
#         hd, mutation_info = hamming_distance(WT, decoded_sequence)

#         # Only consider sequences with exactly 5 mutations
#         if hd == 5:
#             if decoded_sequence not in unique_sequences:
#                 unique_sequences.add(decoded_sequence)
#                 filtered_representations.append(new_representations[idx])
#                 filtered_sequences.append(decoded_sequence)

#                 for mut in mutation_info:
#                     unique_mut = f"{mut['pos']}_{mut['orig_aa']}_{mut['mut_aa']}"
#                     unique_mutations.add(unique_mut)
#                     unique_positions.add(mut['pos'])

#             # Stop when 1000 unique sequences are found
#             if len(unique_sequences) >= 100000:
#                 break

#     # Compile metrics
#     mutation_diversity = len(unique_mutations)
#     pos_diversity = len(unique_positions)

#     return filtered_representations, {
#         'mutation_diversity': mutation_diversity,
#         'position_diversity': pos_diversity,
#         'number_of_unique_sequences': len(unique_sequences)}, filtered_sequences

# # Function to identify mutations and count them
# def identify_mutations_and_count(WT, seq):
#     mutations = []
#     for i, (wt_aa, seq_aa) in enumerate(zip(WT, seq), start=1):
#         if wt_aa != seq_aa:
#             mutations.append(f"{wt_aa}{i}{seq_aa}")
#     mutation_str = ', '.join(mutations)
#     num_mutations = len(mutations)
#     return mutation_str, num_mutations

# # Generate DataFrames for RL, SFT, and Fixed sequences
# def generate_df(sequences, scores):
#     mutations_list = []
#     num_mutations_list = []
    
#     for seq in sequences:
#         mutation_str, num_mutations = identify_mutations_and_count(WT, seq)
#         mutations_list.append(mutation_str)
#         num_mutations_list.append(num_mutations)
    
#     return pd.DataFrame({
#         'Sequence': sequences,
#         'Score': scores,
#         'Mutations': mutations_list,
#         'Number of Mutations': num_mutations_list})

# def mutate_sequences_after_training(model, num_seqs, num_muts, WT):
#     """ Generate mutated sequences based on the model """
#     mutated_seqs = []
#     tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D") # EsmTokenizer
#     masked_pos = torch.zeros((num_seqs, num_muts), dtype=torch.long)
    
#     for seq_idx in range(num_seqs):
#         final_seq = WT # Begin with CreiLOV for first timestep of epoch
#         muts_added = 0
        
#         while muts_added < num_muts:
#             # Mask multiple positions at once
#             positions = random.sample(range(1, len(WT)), k=num_muts - muts_added)
#             masked_seq = list(final_seq)
#             masked_pos[seq_idx, :len(positions)] = torch.tensor(positions)
#             for pos in positions:
#                 masked_seq[pos] = '<mask>'
#             inputs = tokenizer(''.join(masked_seq), return_tensors="pt")
#             model.eval()
#             with torch.no_grad():
#                 out = model(**inputs)
            
#             # Process each masked position
#             for mut_idx, pos in enumerate(positions):
#                 mut_idx = mut_idx + muts_added
#                 logits = out.logits[0, pos+1, 4:24]  # Adjust index if needed
#                 log_probs = F.log_softmax(logits, dim=-1)
#                 probs = torch.exp(log_probs)
#                 sampled_idx = torch.multinomial(probs, 1).item()
#                 sampled_aa = tokenizer.convert_ids_to_tokens(sampled_idx+4) # Correct for offset for amino acid tokens
#                 final_seq = list(final_seq)
#                 final_seq[pos] = sampled_aa
#                 final_seq = ''.join(final_seq)
#             muts_added = hamming_distance(WT, final_seq)
#             if muts_added >= num_muts:
#                 mutated_seqs.append(''.join(final_seq))
#                 break
#     return mutated_seqs

# def mutate_sequences_after_training_esm2_max_sampling(model, num_seqs, num_muts, WT, model_identifier=None):
#     """ Generate mutated sequences based on the model """
#     mutated_seqs = []
#     masked_pos = torch.zeros((num_seqs, num_muts), dtype=torch.long).to(device)
    
#     # # ESMC 6B cannot be moved to device
#     # if model_identifier != "esmc-6b-2024-12":
#     #     model.to(device)
#     #     model.eval()
#     # else:
#     #     import time
#     #     from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

#     #     @retry(stop=stop_after_attempt(10), wait=wait_fixed(1.2))  # Retry up to 10 times with a 1.2s wait
#     #     def encode_protein_with_retry(protein):
#     #         return model.encode(protein)

#     # # ESMC models use difference tokenizer than ESM2
#     # if model_identifier == 'esmc_300m' or 'esmc_600m' or 'esmc-6b-2024-12':
#     #     from esm.sdk.api import ESMProtein, LogitsConfig
#     #     print('imported ESMProtein to prepare sequences for ESMC')

#     #     # copy and pasted from https://github.com/evolutionaryscale/esm/blob/main/esm/utils/constants/esm3.py
#     #     SEQUENCE_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P",
#     #     "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "|", "<mask>" ]

#     #     for seq_idx in range(num_seqs):
#     #         final_seq = WT # Begin with CreiLOV for first timestep of epoch
#     #         muts_added = 0

#     #         while muts_added < num_muts:
#     #             # Mask multiple positions at once
#     #             positions = random.sample(range(1, len(WT)), k=num_muts - muts_added)
#     #             masked_seq = list(final_seq)
#     #             masked_pos[seq_idx, :len(positions)] = torch.tensor(positions).to(device)
#     #             for pos in positions:
#     #                 masked_seq[pos] = '<mask>'
#     #             protein = ESMProtein(sequence=''.join(masked_seq))

#     #             # Throttle requests to avoid hitting API limits
#     #             if model_identifier == "esmc-6b-2024-12":
#     #                 time.sleep(1.2)  # Adjust time based on your rate limit (e.g., 50 requests/minute = 1.2 seconds/request)

#     #             # Encode the sequence and process logits
#     #             with torch.no_grad():
    
#     #                 # Throttle requests to avoid hitting API limits
#     #                 if model_identifier == "esmc-6b-2024-12":
#     #                     try:
#     #                         protein_tensor = encode_protein_with_retry(protein)
#     #                     except RetryError as e:
#     #                         print(f"RetryError: Failed to encode protein after multiple attempts: {e}")
#     #                         break  # Skip this sequence and continue to the next
#     #                     except Exception as e:
#     #                         print(f"Unexpected error encoding protein: {e}")
#     #                         time.sleep(60)  # Wait 1 minute to reset the rate limit
#     #                         continue
                    
#     #                 else:
#     #                     protein_tensor = model.encode(protein)

#     #                 logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True))

#     #             sequence_logits = logits_output.logits.sequence.squeeze(0)  # Remove batch dimension
                
#     #             # Process each masked position
#     #             for pos in positions:
#     #                 logits = sequence_logits[pos+1, 4:24]  # Extract logits for amino acids
#     #                 log_probs = F.log_softmax(logits, dim=-1)
#     #                 probs = torch.exp(log_probs)
#     #                 sampled_idx = torch.argmax(probs).item()  # Replace with `torch.multinomial` if sampling is needed
#     #                 sampled_aa = SEQUENCE_VOCAB[sampled_idx + 4]  # Correct for offset for amino acid tokens
#     #                 final_seq = list(final_seq)
#     #                 final_seq[pos] = sampled_aa
#     #                 final_seq = ''.join(final_seq)
#     #             muts_added = hamming_distance(WT, final_seq)  # Calculate mutations added
#     #             if muts_added >= num_muts:
#     #                 mutated_seqs.append(final_seq)
#     #                 break

#     # else:
#     tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D") # esm2 tokenizer, works for all ESM2 model sizes

#     for seq_idx in range(num_seqs):
#         final_seq = WT # Begin with CreiLOV for first timestep of epoch
#         muts_added = 0
        
#         while muts_added < num_muts:
#             # Mask multiple positions at once
#             positions = random.sample(range(1, len(WT)), k=num_muts - muts_added)
#             masked_seq = list(final_seq)
#             masked_pos[seq_idx, :len(positions)] = torch.tensor(positions).to(device)
#             for pos in positions:
#                 masked_seq[pos] = '<mask>'
#             inputs = tokenizer(''.join(masked_seq), return_tensors="pt").to(device)
#             with torch.no_grad():
#                 out = model(**inputs)
            
#             # Process each masked position
#             for mut_idx, pos in enumerate(positions):
#                 mut_idx = mut_idx + muts_added
#                 logits = out.logits[0, pos+1, 4:24]  # Adjust index if needed
#                 log_probs = F.log_softmax(logits, dim=-1)
#                 probs = torch.exp(log_probs)
#                 # sampled_idx = torch.multinomial(probs, 1).item()
#                 sampled_idx = torch.argmax(probs).item()
#                 sampled_aa = tokenizer.convert_ids_to_tokens(sampled_idx+4) # Correct for offset for amino acid tokens
#                 final_seq = list(final_seq)
#                 final_seq[pos] = sampled_aa
#                 final_seq = ''.join(final_seq)
#             muts_added = hamming_distance(WT, final_seq)
#             if muts_added >= num_muts:
#                 mutated_seqs.append(''.join(final_seq))
#                 break
    
#     return mutated_seqs

# def hamming_distance(s1, s2):
#     """Calculates the Hamming distance between two sequences"""
#     return sum(1 for x, y in zip(s1, s2) if x != y and x != '-' and y != '-') # Quantify sequence similarity


# def generate_and_evaluate_mutants_max_sampling(num_designs, num_muts, WT, reward_models, model, seed, model_identifier=None):
#     # Set models to evaluation mode
#     torch.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     if model_identifier != "esmc-6b-2024-12":
#         model.to(device)
#         model.eval()

#     # Mutate sequences using different models
#     mutated_seqs = mutate_sequences_after_training_esm2_max_sampling(model, num_designs, num_muts, WT, model_identifier)
    
#     # Score mutants
#     batch_size = num_designs
#     scores_tensor = torch.zeros((len(reward_models), batch_size), dtype=torch.float32).to(device)
#     for i, reward_model in enumerate(reward_models):
#         reward_model.to(device)
#         reward_model.eval()
#         with torch.no_grad():
#             for j, seq in enumerate(mutated_seqs):
#                 sequence = torch.tensor(aa2ind(list(seq))).to(device)
#                 score = reward_model.predict(sequence)[0][0]
#                 scores_tensor[i, j] = score

#     # Convert PyTorch tensors to NumPy arrays
#     scores_np = scores_tensor.cpu().numpy()

#     return mutated_seqs, scores_np

# def generate_and_evaluate_mutants(num_designs, num_muts, WT, reward_models, model, seed):
#     # Set models to evaluation mode
#     torch.manual_seed(seed)
#     random.seed(seed)
#     model.eval()

#     # Mutate sequences using different models
#     mutated_seqs = mutate_sequences_after_training(model, num_designs, num_muts, WT)
    
#     # Score mutants
#     batch_size = num_designs
#     scores_tensor = torch.zeros((len(reward_models), batch_size), dtype=torch.float32)
#     for i, reward_model in enumerate(reward_models):
#         reward_model.eval()
#         with torch.no_grad():
#             for j, seq in enumerate(mutated_seqs):
#                 sequence = torch.tensor(aa2ind(list(seq)))
#                 score = reward_model.predict(sequence)[0][0]
#                 scores_tensor[i, j] = score

#     # Convert PyTorch tensors to NumPy arrays
#     scores_np = scores_tensor.numpy()

#     return mutated_seqs, scores_np


# def mask_sequence(sequence, mask_pos):
#     """Mask a single position in the sequence and return the masked sequence."""
#     masked_sequence = list(sequence)
#     masked_sequence[mask_pos] = '<mask>'  # Adjust for the <cls> token shift
#     masked_seq_str = ''.join(masked_sequence)
#     return masked_seq_str

# def get_logits_for_all_positions(model, WT, tokenizer, model_identifier=None):
#     """Generate logits for all positions in the WT sequence by masking one position at a time."""
#     sequence_length = len(WT)
#     all_logits = []

#     with torch.no_grad():
#         for pos in range(0, sequence_length):  # Positions excluding <cls> and <eos>
#             masked_seq = mask_sequence(WT, pos)
#             inputs = tokenizer(masked_seq, return_tensors='pt')
            
#             # Get logits from the model
#             outputs = model(**inputs)
#             logits = outputs.logits
        
#             # Extract logits for the masked position
#             masked_logits = logits[0, pos+1]  # Shape: [vocab_size]
#             all_logits.append(masked_logits)
    
#     return torch.stack(all_logits)  # Shape: [sequence_length, vocab_size]

# def generate_heatmap(WT, probabilities, model_identifier, sequence, filepath, ep, version, tokenizer):
#     """Generate and save a heatmap based on the predicted probabilities."""
    
#     # Generate mutations relative to WT
#     muts_rel_WT = get_mutations(sequence, WT)

#     # Set up tokens and color map
#     all_tokens = tokenizer.get_vocab().keys()
#     all_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in all_tokens]

#     # Create heatmap
#     plt.figure(figsize=(30, 6))
#     Magma_r = plt.cm.magma_r(np.linspace(0, 1, 256))
#     Magma_r[0] = [0, 0, 0, 0.03]
#     cmap = LinearSegmentedColormap.from_list("Modified_Magma_r", Magma_r, N=256)
#     heatmap = sns.heatmap(probabilities.detach().numpy().T, cmap=cmap, square=True, linewidths=0.003, linecolor='0.7', vmin=0, vmax=1)
#     cbar = heatmap.collections[0].colorbar
#     cbar.set_label('Predicted Amino Acid Probabilities at Each Position', fontsize=16)
#     cbar.ax.tick_params(labelsize=12)
#     plt.yticks(np.arange(len(all_tokens)) + 0.5, all_tokens, fontsize=8, rotation=0)
#     plt.xlabel("Position in sequence", fontsize=18)
#     plt.ylabel('Tokens', fontsize=18)
#     plt.title(f'Probabilities of single mutants for {muts_rel_WT} from {model_identifier}')

#     # Add dark blue dots for WT residues and orange dots for mutations
#     for pos, token in enumerate(sequence):  
#         token_id = tokenizer.convert_tokens_to_ids(token)
#         if token_id in all_token_ids:  # Check if the token exists in the token list
#             token_index = all_token_ids.index(token_id)
#             dot_color = 'red' if token != WT[pos] else 'black' # Set dot color based on whether it matches WT or is a mutation
#             plt.scatter(pos + 0.5, token_index + 0.5, color=dot_color, s=30)  # Adjust dot size as needed
#     legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='WT'),
#                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Mutation')]
#     plt.legend(handles=legend_elements, loc='upper right')
#     plt.tight_layout()
#     plt.savefig(f'{filepath}/version_{version}/single_mutant_probability_heatmap_for_{muts_rel_WT}_from_{model_identifier}_ep{ep}.png')
#     plt.savefig(f'{filepath}/version_{version}/single_mutant_probability_heatmap_for_{muts_rel_WT}_from_{model_identifier}_ep{ep}.svg')
#     plt.close()

# def get_mutations(seq, wt):
#     # Find mutations and their positions
#     mutations = [f"{wt_res}{i}{seq_res}" for i, (wt_res, seq_res) in enumerate(zip(wt, seq), 1) if seq_res != wt_res]
#     # Return the WT or mutation string
#     if not mutations:
#         return "WT"  # or simply return wt
#     else:
#         return "_".join(mutations)

# def generate_high_confidence_mutant_sequences(WT, probabilities, tokenizer, model_identifier, num_designs=5, threshold=0.9):
#     """
#     Generate mutant sequences by identifying high-confidence mutations and mutating WT based on p-sampling.
    
#     Args:
#     - WT (str): Wild-type sequence.
#     - probabilities (torch.Tensor): Probability tensor of shape [sequence_length, vocab_size].
#     - num_designs (int): Number of mutated sequences to generate.
#     - threshold (float): Probability threshold for mutation detection.

#     Returns:
#     - mutated_sequences (list): List of duplicated mutated sequences based on high-confidence mutations.
#     """
#     all_tokens = list(tokenizer.get_vocab().keys())  # Get the list of all tokens for reference
#     high_conf_mutations = {}

#     # Identify high-confidence mutations
#     for pos, wt_res in enumerate(WT):
#         position_probs = probabilities[pos]  # Get the probability distribution for this position
#         wt_token_id = tokenizer.convert_tokens_to_ids(wt_res)

#         # Find tokens with probability > threshold, excluding WT residue
#         high_conf_tokens = [(all_tokens[token_id], prob.item()) for token_id, prob in enumerate(position_probs)
#                             if token_id != wt_token_id and prob > threshold]

#         if high_conf_tokens:
#             high_conf_mutations[pos + 1] = high_conf_tokens  # Store position as 1-indexed

#     # Generate a single mutated sequence based on high-confidence mutations
#     mutated_seq = list(WT)  # Start with the WT sequence as a list for mutability

#     for pos, mutations in high_conf_mutations.items():
#         # Sample the token with the maximum probability among high-confidence mutations
#         max_token, max_prob = max(mutations, key=lambda x: x[1])
#         mutated_seq[pos - 1] = max_token  # Apply mutation (pos-1 to convert to 0-indexed)

#     # Convert mutated sequence list back to a string
#     sequence_with_high_confidence_mutations = "".join(mutated_seq)

#     return sequence_with_high_confidence_mutations

# def generate_mutated_sequences(WT, sequences, cum_prob_threshold, probabilities, model, tokenizer, num_muts, model_identifier):
#     """
#     Mutates each sequence in `sequences` until they have `num_muts` mutations relative to `WT`.
    
#     Parameters:
#         WT (str): Wildtype sequence.
#         sequences (list of str): Initial sequences to mutate.
#         probabilities (torch.Tensor): Probability tensor for each position in the sequence.
#         model (torch.nn.Module): Model to generate logits for mutation.
#         num_muts (int): Target number of mutations for each sequence.

#     Returns:
#         list of str: Mutated sequences with the specified number of mutations relative to WT.
#     """
#     # Tokenize WT sequence
#     WT_tokens = tokenizer.convert_tokens_to_ids(list(WT))
#     mutated_seqs = []

#     # Identify candidate positions with cumulative probability > 25% for non-wildtype amino acids
#     candidate_positions = []
#     position_weights = []
#     for i, p in enumerate(probabilities):
#         non_wt_prob = p.sum() - p[WT_tokens[i]]  # cumulative probability for non-wildtype amino acids
#         if non_wt_prob > cum_prob_threshold:
#             candidate_positions.append(i)
#             position_weights.append(non_wt_prob.item())  # store the probability as weight

#     # Normalize weights for `random.choices`
#     total_weight = sum(position_weights)
#     normalized_weights = [w / total_weight for w in position_weights]
    
#     with torch.no_grad():
#         for seq in sequences:
#             mutated_seq = list(seq)
#             # print('mutated_seq', mutated_seq)
            
#             while hamming_distance(mutated_seq, WT) < num_muts:
#                 # Randomly choose a candidate position
#                 pos = random.choices(candidate_positions, weights=normalized_weights, k=1)[0]
#                 # print('pos', pos)

#                 # Mask the chosen position
#                 mutated_seq[pos] = tokenizer.mask_token  # Use <mask> token
                
#                 # Prepare input for the model
#                 masked_seq_str = ''.join(mutated_seq)
#                 # print('masked_seq_str', masked_seq_str)
#                 inputs = tokenizer(masked_seq_str, return_tensors="pt")
#                 outputs = model(**inputs)

#                 # Get logits for valid amino acid tokens
#                 logits = outputs.logits[0, pos + 1, 4:24]  # Adjust this range based on valid amino acid tokens
#                 probabilities_pos = F.softmax(logits, dim=-1)
#                 # print('probabilities_pos', probabilities_pos)

#                 # Sample a new amino acid for the position
#                 sampled_idx = np.random.choice(len(probabilities_pos), p=probabilities_pos.detach().numpy())
#                 new_amino_acid_id = 4 + sampled_idx  # Map to actual token ID range for amino acids
#                 new_amino_acid = tokenizer.convert_ids_to_tokens([new_amino_acid_id])[0]
#                 # print('new_amino_acid', new_amino_acid)

#                 # Apply the mutation
#                 mutated_seq[pos] = new_amino_acid
#                 # print('mutated_seq',mutated_seq)

#             # Convert tokenized mutated sequence back to amino acid string
#             mutated_seq = ''.join(mutated_seq)
#             mutated_seqs.append(mutated_seq)

#         # print(mutated_seqs)

#     return mutated_seqs

# def generate_and_evaluate_mutants_p_sampling(WT, reward_models, model, model_identifier, tokenizer, filepath, ep, version, num_designs=5, num_muts=5, cum_prob_threshold=0.25, high_conf_threshold=0.9, seed=None):
#     # Set models to evaluation mode
#     model.eval()

#     # Seed is set during generation after training but not during training
#     if seed is not None:
#         # Set seeds
#         torch.manual_seed(seed)
#         random.seed(seed)
#         np.random.seed(seed)

#     # Generate single mutant probability space by masking one position at a time of WT,
#     single_mutant_logits = get_logits_for_all_positions(model, WT, tokenizer, model_identifier) # get logits for each position
#     probabilities = F.softmax(single_mutant_logits, dim=-1) # convert logits to probabilities
#     generate_heatmap(WT, probabilities, model_identifier, WT, filepath, ep, version, tokenizer) # generate heatmap and save heatmap as svg and png
#     print('Generated heatmap for single mutant space from WT')

#     # Mutate WT with high confidence mutations using p-sampling
#     sequence_with_high_confidence_mutations = generate_high_confidence_mutant_sequences(WT, probabilities, tokenizer, model_identifier, num_designs, high_conf_threshold)
#     print('Mutated WT with high confidence mutations')

#     # Generate single mutant probability space by masking one position at a time of sequence with high confidence mutations,
#     single_mutant_logits = get_logits_for_all_positions(model, sequence_with_high_confidence_mutations, tokenizer, model_identifier) # get logits for each position
#     probabilities = F.softmax(single_mutant_logits, dim=-1) # convert logits to probabilities
#     generate_heatmap(WT, probabilities, model_identifier, sequence_with_high_confidence_mutations, filepath, ep, version, tokenizer) # generate heatmap and save heatmap as svg and png
#     print('Generated heatmap for single mutant space from sequence with high confidence mutations')

#     # Duplicate the mutated sequence to match the number of designs
#     sequences_with_high_confidence_mutations = [sequence_with_high_confidence_mutations] * num_designs

#     # Add mutations until num_muts of mutations relative to WT sequence are obtained for all sequences_with_high_confidence_mutations
#     mutated_seqs = generate_mutated_sequences(WT, sequences_with_high_confidence_mutations, cum_prob_threshold, probabilities, model, tokenizer, num_muts, model_identifier)
#     print('Mutated sequences with 5 mutations')
#     # print(mutated_seqs)
 
#     # Score mutants
#     batch_size = num_designs
#     scores_tensor = torch.zeros((len(reward_models), batch_size), dtype=torch.float32).to(device)
#     for i, reward_model in enumerate(reward_models):
#         reward_model.to(device)
#         reward_model.eval()
#         with torch.no_grad():
#             for j, seq in enumerate(mutated_seqs):
#                 sequence = torch.tensor(aa2ind(list(seq))).to(device)
#                 score = reward_model.predict(sequence)[0][0]
#                 scores_tensor[i, j] = score

#     # Convert PyTorch tensors to NumPy arrays
#     scores_np = scores_tensor.cpu().numpy()

#     return mutated_seqs, scores_np





