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
from functions import (load_vae_model, load_reward_model, find_closest_average_hd, generate_and_evaluate_mutants_2, generate_and_evaluate_mutants_vae_training,
    decoding, Using_VAE, hamming_distance_vae_training, ProtDataModule, ProtRepDataset, adjust_designs, SeqDataset, convert_and_score_sequences,
    save_metrics_to_csv, identify_mutations, save_sorted_designs_to_csv)
from conv_vae_model import ConvVAE
from MLP import MLP
from torch_ema import ExponentialMovingAverage

# Define amino acid dictionary for tokenization, define WT for length of context window
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
WT = "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA" # CreiLOV
Best_Single_Mutant = "MAGLRHSFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
# device = torch.device('cpu') # This will force using CPU even if CUDA is available
slen = len(WT)

# Running RLXF
class RLXF_PPO_vae(pl.LightningModule):
    def __init__(self, reward_models, rl_updated_vae, vae_fixed_model, version, seed,
                     learning_rate, lr_mult, lr_mult_factor, use_scheduler, warm_restart,
                     WD, clip_type, grad_clip_threshold, grad_clip_threshold_factor, epsilon,
                     epochs, iterations,
                     dkl_scale, dkl_scale_init, dkl_backpropagation,
                     num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch,
                     batch_size, inc_batch_size, max_batch_size,
                     target_hd,
                     pairwise_hd_aver_factor, sampling_max, rel_to_WT, average_type, average_type_loss,
                     num_EnsMLPs):
        super().__init__()
        # fix random seeds for reproducibility
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # models for RLXF
        self.rl_updated_vae = rl_updated_vae
        self.ema = ExponentialMovingAverage(self.rl_updated_vae.parameters(), decay=0.992)
        self.vae_fixed_model =  vae_fixed_model
        self.reward_models = reward_models
        self.num_EnsMLPs = num_EnsMLPs


        # hyperparameters
        self.learning_rate = learning_rate
        self.learning_rate_0 = learning_rate
        self.WD = WD
        self.lr_mult = lr_mult
        self.lr_mult_factor = lr_mult_factor
        self.eps = epsilon
        self.epochs = epochs
        self.beta_init = dkl_scale_init
        self.beta = dkl_scale
        self.iterations = iterations
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.inc_batch_size = inc_batch_size
        self.target_hd = target_hd
        self.version = version
        self.dkl_backpropagation = dkl_backpropagation
        self.sampling_max = sampling_max
        self.average_type = average_type
        self.average_type_loss = average_type_loss
        self.rel_to_WT = rel_to_WT
        self.pairwise_hd_aver_factor = pairwise_hd_aver_factor
        self.clip_type = clip_type
        if self.clip_type == 1:
            self.grad_clip_threshold = grad_clip_threshold
            self.grad_clip_threshold_factor = grad_clip_threshold_factor
        self.num_unfrozen_layers = num_unfrozen_layers
        self.num_layers_unfreeze_each_epoch = num_layers_unfreeze_each_epoch
        self.max_num_layers_unfreeze_each_epoch = max_num_layers_unfreeze_each_epoch
        
        # Choosing layers of model to update
        named_decoder_layers = []
        for idx, (name, param) in enumerate(self.rl_updated_vae.named_parameters()):
            named_decoder_layers.append(name) # Append layer name
        named_decoder_layers.reverse()
        selected_layers = named_decoder_layers[0:self.num_unfrozen_layers]

        # store params & learning rates
        self.decoder_params = []
        for idx, name in enumerate(selected_layers):
            # print(f'{idx}: self.learning_rate = {self.learning_rate:.8f}, {name}')
            self.decoder_params += [{'params': [p for n, p in self.rl_updated_vae.named_parameters() if n == name and p.requires_grad],
                            'lr': self.learning_rate}] # append layer parameters
            self.learning_rate *= self.lr_mult # update learning rate
        
        # parameters for custom training
        self.automatic_optimization = False
        self.use_scheduler = use_scheduler
        self.warm_restart = warm_restart
        optimizers_config = self.configure_optimizers()
        if self.use_scheduler == 1:
            self.optimizer = optimizers_config["optimizer"]
            self.scheduler = optimizers_config["lr_scheduler"]
        else:
            self.optimizer = optimizers_config
        self.AAs = 'ACDEFGHIKLMNPQRSTVWY-'
        self.WT = "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
        self.CreiLOV = torch.tensor([10,  0,  5,  9, 14,  6, 16,  4, 17, 17,  0,  2,  0, 16,  9, 12,  2,  1,
                                     12,  9, 17, 19,  0, 15,  3,  5,  4, 19,  0, 10, 16,  5, 19,  5, 12,  2,
                                     3, 17,  9,  5,  6, 11,  0, 14,  4,  9, 13,  5,  3,  5, 16,  2, 12,  8,
                                     3, 17, 13,  8,  7, 14,  2,  0,  7,  8,  8,  5,  3,  0,  1, 15, 17, 14,
                                     9,  9, 11, 19, 14,  8,  2,  5, 16, 12,  4, 18, 11,  9,  9, 16, 17, 16,
                                     12,  7,  8, 16, 12,  2,  5, 14, 17, 15,  8,  4, 17,  5, 17, 13, 17,  2,
                                     17, 16, 15,  8, 16,  3,  5,  8,  0,  9,  0]) # tensor representation of CreiLOV
        
        self.save_hyperparameters('version', 'seed', 'learning_rate', 'lr_mult', 'lr_mult_factor', 'use_scheduler', 'warm_restart', 'WD', 'clip_type', 'grad_clip_threshold', 'grad_clip_threshold_factor', 'epsilon', 'epochs', 'iterations', 'dkl_scale', 'dkl_scale_init', 'dkl_backpropagation', 'num_unfrozen_layers', 'num_layers_unfreeze_each_epoch', 'max_num_layers_unfreeze_each_epoch', 'batch_size', 'inc_batch_size', 'max_batch_size', 'target_hd', 'pairwise_hd_aver_factor', 'sampling_max', 'rel_to_WT', 'average_type', 'average_type_loss') # log hyperparameters to file
    
    def initial_log_probabilities(self, batch):
        """ Computes log probabilities matrices (states) for given batch of sequences using VAEs
        Args:
            batch (torch.FloatTensor): (batch_size, latent_dimensions_VAE)
                batch of proteins representations
        Returns:
            initial_log_states (torch.FloatTensor): (batch_size, length of amico acid dictionary = 21, length of protein = 119)
                log probabilities of initial states from pre-trained model
        """
        with torch.no_grad():  # Ensure training does not occur during scoring
            self.vae_fixed_model.eval() # Ensure vae_fixed_model is in eval mode
            logits = self.vae_fixed_model.decoder(batch) # Decodes noisy CreiLOV representations
            initial_log_states = F.log_softmax(logits, dim=1) # Apply log_softmax to each amino acid of each protein
        return initial_log_states

    def new_log_probabilities(self, batch):
        """ Computes log probabilities matrices (states) for given batch of sequences using VAEs
        Args:
            batch (torch.FloatTensor): (batch_size, latent_dimensions_VAE)
        Returns:
            new_log_states (torch.FloatTensor, grad_fn=<LogSoftmaxBackward0>): (batch_size, length of amico acid dictionary = 21, length of protein = 119)
                log probabilities of new states after policy update
        """
        # Check if NaN or inf values led to unstable training of VAE parameters
        for name, param in self.rl_updated_vae.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"New Probabilities: Invalid values in parameter {name}")
                
        self.rl_updated_vae.eval() # Set model to evaluation mode
        logits = self.rl_updated_vae.decoder(batch) # decode noisy CreiLOV representations
        new_log_states = F.log_softmax(logits, dim=1) # Apply log_softmax to each amino acid of each protein
        self.rl_updated_vae.train() # Set model back to training mode to allow for backpropagation
        return new_log_states

    def clipped_loss(self, prob_ratio, total_reward):
        """ Computes clipped surrogate loss for update (PPO details: https://arxiv.org/abs/1707.06347)
        Args:
            prob_ratio (torch.FloatTensor, grad_fn=<CopySlices>): (batch_size, length of protein)
                 ratio of probabilities between rl-updated vae and pre-trained vae for each action for each protein (choosing amino acids for protein)
            total_reward (torch.FloatTensor): ([scalar])
                This is the total reward calculated from the mean fitness of sequences and Dkl loss
        Returns:
            clipped_loss (torch.FloatTensor, grad_fn=<NegBackward0>): ((batch_size, length of protein = 119))
        """
        # Returns the minimum value of inputs
        clipped_loss = -torch.min(prob_ratio*total_reward, torch.clamp(prob_ratio, 1-self.eps, 1+self.eps)*total_reward)
        return clipped_loss
    
    def action(self, new_log_states, initial_log_states):
        """ Samples from the given state tensor and returns one-hot encoded proteins.
        Args:
            new_log_states (torch.FloatTensor, grad_fn=<LogSoftmaxBackward0>): (batch_size, length of amico acid dictionary = 21, length of protein = 119)
                log probabilities of new states after policy update
        Returns:
            tensor_proteins (torch.LongTensor): (batch_size, length of protein = 119)
                A 2D tensor containing the one-hot encoded representations of the sampled proteins.
            mean_hd_from_CreiLOV (torch.FloatTensor): ([])
                Average hamming distance of sampled protein for batch from WT
        """
        # Sampling is not included in computational graph (non-differientiable)
        rl_log_states_detached = new_log_states.clone()
        pre_log_states_detached = initial_log_states.clone()
        
        # sampling (no backpropagation b/c not differientiable due to randomness)
        with torch.no_grad():
                batch_size, _, sequence_length = rl_log_states_detached.size() # find dimensions
                rl_tensor_proteins = torch.zeros((batch_size, sequence_length), dtype=torch.int64) # Initialize a tensor to store the sampled indices
                if self.rel_to_WT == 0:
                    pre_tensor_proteins = torch.zeros((batch_size, sequence_length), dtype=torch.int64) # Initialize a tensor to store the sampled indices

                # Convert log probabilities to probabilities to sample
                rl_states_detached = torch.exp(rl_log_states_detached)
                rl_states_detached = torch.clamp(rl_states_detached, min=1e-31) # Ensure values do not diverge below 1e-32
                if self.rel_to_WT == 0:
                    pre_states_detached = torch.exp(pre_log_states_detached)
                    pre_states_detached = torch.clamp(pre_states_detached, min=1e-31) # Ensure values do not diverge below 1e-32
                
                # Loop over each sequence in the batch
                for i in range(batch_size):
                    # Set the probability of "-" gap character (21st index) to near 0
                    rl_states_detached[i, 20, :] = 1e-31
                    
                    if self.rel_to_WT == 0:
                        pre_states_detached[i, 20, :] = 1e-31
                    
                    # if i == 0:
                    #     print(states_detached[i])
    
                    # Check if invalid probabilities exist in state
                    if torch.isnan(rl_states_detached[i]).any():
                        print(f"NaN values found in states at index {i}")
                    if torch.isinf(rl_states_detached[i]).any():
                        print(f"Infinity values found in states at index {i}")
                    if (rl_states_detached[i] < 0).any():
                        print(f"Negative values found in states at index {i}")

                    if self.sampling_max == 1:
                        # Select the index with the maximum probability for each amino acid in the protein
                        rl_tensor_proteins[i] = torch.argmax(rl_states_detached[i], dim=0).squeeze()
                        if self.rel_to_WT == 0:
                            pre_tensor_proteins[i] = torch.argmax(pre_states_detached[i], dim=0).squeeze()
                    
                    else:
                        # Sample from the probability distribution for each amino acid index in protein and squeeze into tensor of dimension (sequence len
                        rl_tensor_proteins[i] = torch.multinomial(rl_states_detached[i].transpose(0, 1), 1).squeeze()
                        if self.rel_to_WT == 0:
                            pre_tensor_proteins[i] = torch.multinomial(pre_states_detached[i].transpose(0, 1), 1).squeeze()

                    # Ensure the N terminus begins with methionine and alanine (gaps frequently occur in MSA, based on CreiLOV WT)
                    rl_tensor_proteins[i, 0] = 10
                    rl_tensor_proteins[i, 1] = 0
                    if self.rel_to_WT == 0:
                        pre_tensor_proteins[i, 0] = 10
                        pre_tensor_proteins[i, 1] = 0

                    # Ensure the C terminus ends with KALA (gaps frequently occur in MSA, based on CreiLOV WT)
                    rl_tensor_proteins[i, -4] = 8
                    rl_tensor_proteins[i, -3] = 0
                    rl_tensor_proteins[i, -2] = 9
                    rl_tensor_proteins[i, -1] = 0
                    if self.rel_to_WT == 0:
                        pre_tensor_proteins[i, -4] = 8
                        pre_tensor_proteins[i, -3] = 0
                        pre_tensor_proteins[i, -2] = 9
                        pre_tensor_proteins[i, -1] = 0

                    # Ensure key mutation for CreiLOV fluorescent does not revert back to cysteine (rl_tensor_proteins[i, 42] = 1)
                    # Mukherjee et al., 2015
                    if rl_tensor_proteins[i, 42] == 1:
                        rl_probabilities = rl_states_detached[i]
                        rl_probabilities[1,42] = 1e-31  # Set the probability of cysteine to 0
                        
                        if self.sampling_max == 1:
                            aa_43 = torch.argmax(rl_probabilities[:,42]) # Find the next highest probability index
                        
                        else:
                            aa_43 = torch.multinomial(rl_probabilities[:,42], 1)  # resample
                        rl_tensor_proteins[i, 42] = aa_43

                    if self.rel_to_WT == 0:
                        if pre_tensor_proteins[i, 42] == 1:
                            pre_probabilities = pre_states_detached[i]
                            pre_probabilities[1,42] = 1e-31  # Set the probability of cysteine to 0
                            
                            if self.sampling_max == 1:
                                aa_43 = torch.argmax(pre_probabilities[:,42]) # Find the next highest probability index
                            
                            else:
                                aa_43 = torch.multinomial(pre_probabilities[:,42], 1)  # resample
                            pre_tensor_proteins[i, 42] = aa_43

                # Calculate and print mean hamming distance
                total_hd = 0
                for i in range(batch_size):
                    hd = self.hamming_distance(self.CreiLOV, rl_tensor_proteins[i, :])
                    total_hd += hd
                mean_hd_from_CreiLOV = torch.tensor(total_hd / batch_size)

        if self.rel_to_WT == 0:
            return rl_tensor_proteins, pre_tensor_proteins, mean_hd_from_CreiLOV
        else:
            return rl_tensor_proteins, None, mean_hd_from_CreiLOV

    def reward(self, rl_tensor_proteins, pre_tensor_proteins):
        """ Calculate mean fitness score for proteins created by agent after sampling (action)
        Args:
            tensor_proteins (torch.LongTensor): (batch_size, length of protein = 119)
                A 2D tensor containing the one-hot encoded representations of the sampled proteins.
        Returns:
            reward (torch.FloatTensor): ([])
                Mean fitness for batch of sampled proteins
        Notes:
            if kernel dies, make sure the reward_model.py predict function is formatted correctly
        """
        # Initialize a tensor to store scores for all sequences from all models
        batch_size = rl_tensor_proteins.size(0)
        rl_scores_tensor = torch.zeros((len(self.reward_models), batch_size), dtype=torch.float32)
        if self.rel_to_WT == 0:
            pre_scores_tensor = torch.zeros((len(self.reward_models), batch_size), dtype=torch.float32)

        # Ensure all models are in evaluation mode
        for model in self.reward_models:
            model.eval()

        # Compute score for each sequence in the batch from each model
        with torch.no_grad():
            for i, model in enumerate(self.reward_models):
                for j in range(batch_size):
                    sequence = rl_tensor_proteins[j]  # Extract tensor protein
                    score = model.predict(sequence)[0][0]  # Extract score for protein from current model
                    rl_scores_tensor[i, j] = score
 
        # Calculate the 5th percentile (bottom 5% quantile) fitness for each sequence
        rl_fitness_per_sequence = torch.quantile(rl_scores_tensor, 0.05, dim=0)

        if self.rel_to_WT == 0:
            # Compute score for each sequence in the batch from each model
            with torch.no_grad():
                for i, model in enumerate(self.reward_models):
                    for j in range(batch_size):
                        sequence = pre_tensor_proteins[j]  # Extract tensor protein
                        score = model.predict(sequence)[0][0]  # Extract score for protein from current model
                        pre_scores_tensor[i, j] = score
     
            # Calculate the 5th percentile (bottom 5% quantile) fitness for each sequence
            pre_fitness_per_sequence = torch.quantile(pre_scores_tensor, 0.05, dim=0)

        # Compute the overall fitness score based on average_type
        if self.average_type == 0:
            rl_fitness = rl_fitness_per_sequence.mean()
            if self.rel_to_WT == 0:
                pre_fitness = pre_fitness_per_sequence.mean()
        elif self.average_type == 1:
            rl_fitness = rl_fitness_per_sequence.median()
            if self.rel_to_WT == 0:
                pre_fitness = pre_fitness_per_sequence.median()
        elif self.average_type == 2:
            rl_fitness = rl_fitness_per_sequence.max()
            if self.rel_to_WT == 0:
                pre_fitness = pre_fitness_per_sequence.max()
        elif self.average_type == 3:
            rl_fitness = rl_fitness_per_sequence.min()
            if self.rel_to_WT == 0:
                pre_fitness = pre_fitness_per_sequence.min()

        mean_WT_fitness = (4.023229753 + 4.170359914) / 2
        rel_WT_fitness = rl_fitness / mean_WT_fitness
        
        if self.rel_to_WT == 1:
            fitness_advantage = rel_WT_fitness
        else:
            fitness_advantage = ((rl_fitness - pre_fitness)/pre_fitness)*100

        self.current_fitness_advantage = fitness_advantage # This changes by definition
        self.rel_WT_fitness = rel_WT_fitness

        self.current_rel_WT_fitness = self.rel_WT_fitness.item()
    
        return fitness_advantage, rel_WT_fitness
    
    def compute_ratio(self, initial_log_states, new_log_states, tensor_proteins):
        """ Computes probability ratios for each timestep at indices for amino acids sampled and stored in tensor_proteins
        Args:
            initial_log_states (torch.FloatTensor): (batch_size, length of amino acid dictionary = 21, length of protein = 119)
                log probabilities of initial states from pre-trained model
            new_log_states (torch.FloatTensor, grad_fn=<ClampBackward1>): (batch_size, length of amino acid dictionary = 21, length of protein = 119)
                log probabilities of new states after policy update
            tensor_proteins (torch.LongTensor): Sampled proteins
        Returns:
            ratios (torch.FloatTensor): (batch_size, length of protein = 119)
                tensor of ratios of new probabilities over old probabilities for each amino acid in each protein
        """
        batch_size, _, sequence_length = initial_log_states.size()
    
        # Initialize the tensor for storing ratios for each amino acid in each protein
        ratios = torch.zeros((batch_size, sequence_length), dtype=torch.float32)
        
        for i in range(batch_size):
            for j in range(sequence_length):
                sampled_index = tensor_proteins[i, j].long()
                log_ratio = new_log_states[i, sampled_index, j] - initial_log_states[i, sampled_index, j]
                ratios[i, j] = torch.exp(log_ratio) # Save vector of ratios for 119 sampled amino acids for each protein
        return ratios

    def Dkl_states(self, initial_log_states, new_log_states, dkl_backpropagation):
        """ Measure different in distribution along each amino acid index between initial and new states
        Args:
            initial_log_states (torch.FloatTensor): (batch_size, length of amino acid dictionary = 21, length of protein = 119)
                log probabilities of initial states from pre-trained model
            new_log_states (torch.FloatTensor, grad_fn=<ClampBackward1>): (batch_size, length of amino acid dictionary = 21, length of protein = 119)
                log probabilities of new states after policy update 
        Returns:
            kl_divergence (torch.FloatTensor): ([])
                kl_divergence between initial and new state matrices
        """
        if dkl_backpropagation == 1:
            # backpropagation through Dkl
            kl_divergence = F.kl_div(input=new_log_states, target=initial_log_states, reduction='batchmean', log_target=True)
            # Bug fix: input is expected to be log probabilities: https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html
        else:
            # No backpropagation through Dkl
            new_log_states = new_log_states.clone()
            with torch.no_grad():
                kl_divergence = F.kl_div(input=new_log_states, target=initial_log_states, reduction='batchmean', log_target=True)
                # Bug fix: input is expected to be log probabilities: https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html
        return kl_divergence

    def training_step(self, batch, batch_idx):
        """ PPO for RLXF: The section of the code goes through a timestep with trajectories for each batch of protein representations"""
        
        if self.current_epoch == 0 and batch_idx == 0:
            print("Using protein_representations generated prior to training for first epoch")
        elif self.current_epoch > 0 and batch_idx == 0:
            self.create_and_load_new_dataset() # Create new dataset after 1st epoch with close to average hamming distance of 5 to CreiLOV when decoded using rl_updated_vae

        # if batch_idx == 0:
        #     print("epoch", self.current_epoch)
        #     print(batch)
        
        # Use beta_init if it's the first epoch, else use beta
        current_beta = self.beta_init if self.current_epoch < 1 else self.beta
        
        # Performing PPO
        init_probs = self.initial_log_probabilities(batch) # Find initial state (pre-trained VAE)
        rlxf_probs = self.new_log_probabilities(batch) # Find new state (rl-updated VAE)
        dkl_value = self.Dkl_states(init_probs, rlxf_probs, self.dkl_backpropagation) # Calculate Dkl between outputs (0 for first epoch, first batch)
        rl_tensor_proteins, pre_tensor_proteins, mean_hd_from_CreiLOV = self.action(rlxf_probs, init_probs) # Sample states to get proteins
        pairwise_hd_aver, total_distance, num_pairs = self.average_pairwise_hamming_distance(rl_tensor_proteins)
        ratios = self.compute_ratio(init_probs, rlxf_probs, rl_tensor_proteins) # Calculate prob ratio of new probs over old probs
        fitness_advantage, rel_WT_fitness = self.reward(rl_tensor_proteins, pre_tensor_proteins) # Calculate fitness of proteins using reward model
        total_reward = (fitness_advantage + self.pairwise_hd_aver_factor*pairwise_hd_aver - current_beta * dkl_value) # Calculate total reward using fitness and dkl_value
        if self.average_type_loss == 0:
            ppo_loss = (self.clipped_loss(ratios, total_reward)).mean() # Calculate PPO loss
        else:
           ppo_loss = (self.clipped_loss(ratios, total_reward)).median() # Calculate PPO loss 

        # Check if model parameters became invalid
        for name, param in self.rl_updated_vae.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Pre-Initial Timestep Backpropagation: Invalid values in parameter {name}")

        # Backpropagation
        self.optimizer.zero_grad()
        ppo_loss.backward()
        if self.clip_type == 1:
            torch.nn.utils.clip_grad_norm_(self.rl_updated_vae.parameters(), self.grad_clip_threshold)
        self.optimizer.step()
        if self.use_scheduler == 1:
                self.lr_scheduler_step(self.scheduler['scheduler'], 0, None)
        self.ema.update()

        # Check if model parameters became invalid
        for name, param in self.rl_updated_vae.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Post-Initial Timestep Backpropagation: Invalid values in parameter {name}")
        
        # Log metrics
        self.log("kl_divergence", dkl_value, prog_bar=False, logger=True, on_step = True, on_epoch=False)
        self.log("mean_ratio_initial_iter", ratios.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("median_ratio_initial_iter", ratios.median(), prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("pairwise_hd_aver", pairwise_hd_aver, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("mean_hd_from_CreiLOV", mean_hd_from_CreiLOV, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("fitness_advantage", fitness_advantage, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("rel_WT_fitness", rel_WT_fitness, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("total_reward", total_reward, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("ppo_loss_initial_iter", ppo_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("batch_size", self.batch_size, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        for i in range(self.iterations-1): 
            rlxf_probs = self.new_log_probabilities(batch) # New states generated to calcualte ratio and dkl for loss
            if self.dkl_backpropagation == 1:
                dkl_value = self.Dkl_states(init_probs, rlxf_probs, self.dkl_backpropagation)
            ratios = self.compute_ratio(init_probs, rlxf_probs, rl_tensor_proteins)
            if self.dkl_backpropagation == 1:
                total_reward = (fitness_advantage + self.pairwise_hd_aver_factor*pairwise_hd_aver - current_beta * dkl_value) # Calculate total reward using fitness and dkl_value
            
            if self.average_type_loss == 0:
                ppo_loss = (self.clipped_loss(ratios, total_reward)).mean() # Calculate PPO loss
            else:
               ppo_loss = (self.clipped_loss(ratios, total_reward)).median() # Calculate PPO loss 

            # Check if model parameters became invalid
            for name, param in self.rl_updated_vae.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"Pre-Trajectory Backpropagation: Invalid values in parameter {name}")

            # Backpropagation
            self.optimizer.zero_grad()
            ppo_loss.backward()
            if self.clip_type == 1:
                torch.nn.utils.clip_grad_norm_(self.rl_updated_vae.parameters(), self.grad_clip_threshold/self.grad_clip_threshold_factor)
            self.optimizer.step()
            if self.use_scheduler == 1:
                self.lr_scheduler_step(self.scheduler['scheduler'], 0, None)
            self.ema.update()

            # Check if model parameters became invalid
            for name, param in self.rl_updated_vae.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"Post-Trajectory Backpropagation: Invalid values in parameter {name}")

            # Log trajectory metrics
            mean_ratio_iter = ratios.mean()
            median_ratio_iter = ratios.median()
            ppo_loss_final_iter = ppo_loss
            self.log("mean_ratio_final_iter", mean_ratio_iter, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log("median_ratio_final_iter", median_ratio_iter, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log("ppo_loss_final_iter", ppo_loss_final_iter, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        # return # print("finished batch") 

    def configure_optimizers(self):
        """ Configure optimizers and optionally a scheduler with warm restarts. """
        optimizer = torch.optim.Adam(self.decoder_params) # weight_decay = self.WD
        
        if self.use_scheduler == 1:
            if self.warm_restart == 1: # If using scheduler for learning rate with warm restart
                T_0 = self.iterations # number of updates within cycle of decaying learning rate
                T_mult = 1  # interval between decay cycles is constant
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
                # return {"optimizer": optimizer, "lr_scheduler": scheduler, "interval": "step"}
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
            
            # If using scheduler for learning rate
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
                # return {"optimizer": optimizer, "lr_scheduler": scheduler, "interval": "step"}
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        # No scheduler
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """ Manually steppings learning rate scheduler. """
        scheduler.step()

    def on_train_epoch_end(self):
        """ This function manually steps the scheduler at the end of each epoch. """
        self.batch_size = min(self.batch_size + self.inc_batch_size, self.max_batch_size) # Increase batch size each epoch until max size reached
        self.learning_rate = self.learning_rate_0
        self.num_unfrozen_layers = min(self.max_num_layers_unfreeze_each_epoch,self.num_unfrozen_layers+self.num_layers_unfreeze_each_epoch)
        self.lr_mult *= self.lr_mult_factor
        
        # Collect all currently optimized parameters to avoid duplication
        current_params = set()
        for group in self.optimizer.param_groups:
            current_params.update(set(group['params']))

        # Selecting layers of model to update next
        named_decoder_layers = []
        for idx, (name, param) in enumerate(self.rl_updated_vae.named_parameters()):
            named_decoder_layers.append(name) # Append layer name
        named_decoder_layers.reverse()
        selected_layers = named_decoder_layers[0:self.num_unfrozen_layers]
        # print(selected_layers)

        # Add new layer parameters to the optimizer without reinitializing it
        for name in selected_layers:
            layer_params = [p for n, p in self.rl_updated_vae.named_parameters() if n == name and p.requires_grad and p not in current_params]
            if layer_params:
                self.optimizer.add_param_group({'params': layer_params,'lr': self.learning_rate})
                current_params.update(set(layer_params))
            self.learning_rate *= self.lr_mult

        # Calculate max norm to monitor model collapse
        max_norm = 0
        for name, parameters in self.rl_updated_vae.named_parameters():
            if parameters.requires_grad:
                param_norm = torch.norm(parameters.grad).item() if parameters.grad is not None else 0
                max_norm = max(max_norm, param_norm)
        self.log('max_norm', max_norm, on_epoch=True, prog_bar=True, logger=True)

    def hamming_distance(self, s1, s2):
        """Calculates the Hamming distance between two sequences"""
        return sum(1 for x, y in zip(s1, s2) if x != y and x != '-' and y != '-') # Quantify sequence similarity
        
    def hamming_distance_tensor(self, t1, t2):
        """Calculate the Hamming distance between two tensors."""
        return torch.sum(t1 != t2)
                         
    def average_pairwise_hamming_distance(self, tensor_proteins):
        """Calculate the average pairwise Hamming distance of a batch of protein sequences for all pairs."""
        n = tensor_proteins.size(0)
        total_distance = 0
        num_pairs = 0
    
        # Iterate over all unique pairs
        for i, j in itertools.combinations(range(n), 2):
            total_distance += self.hamming_distance_tensor(tensor_proteins[i], tensor_proteins[j])
            num_pairs += 1
        average_distance = total_distance / num_pairs # Calculate average distance
        return average_distance, total_distance, num_pairs
    
    def save_rl_updated_vae(self, filepath='rl_updated_vae.pt'):
        """ Save the state dictionary of the rl_updated_vae model to a file.
        Args:
            filepath (str): Path to the file where the state dictionary will be saved.
        """
        # Save the model's state_dict
        try:
            torch.save(self.rl_updated_vae.state_dict(), filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")

        try:
            # Save the rl_updated_model state with EMA weights
            filepath_2 = f'rl_updated_vae_EMA.pt'
            self.ema.store(self.rl_updated_vae.parameters())  # Store the original weights of rl_updated_model
            self.ema.copy_to(self.rl_updated_vae.parameters())  # Apply EMA weights to rl_updated_model
            
            torch.save(self.rl_updated_vae.state_dict(), filepath_2)
            print(f"EMA Model saved to {filepath_2}")
            
            self.ema.restore(self.rl_updated_vae.parameters()) # Restore the original weights after saving
            print('1############################################################################################################')
            # print(self.rl_updated_model.parameters())
            print('2############################################################################################################')

        except Exception as e:
            print(f"An error occurred while saving the model: {e}")

    def create_and_load_new_dataset(self):
        """ Create a new dataset that when decoded with the current rl_updated_vae results in an average hamming distance of 5
        """
        # print(self.batch_size)
        dm = ProtDataModule(self.rl_updated_vae,self.WT,self.AAs,self.batch_size,self.target_hd,None,self.current_epoch,self.version)
        dm.setup(stage='fit') # create new dataset
        self.train_dataloader = dm.train_dataloader()


