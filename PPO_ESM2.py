
# Import packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data.distributed import DistributedSampler
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
from sklearn import metrics
import os
import pickle
from transformers import AutoModelForMaskedLM, AutoTokenizer
from MLP import MLP
import itertools
import copy
import logging
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch_ema import ExponentialMovingAverage

# Running RLXF
class PPO_ESM2(pl.LightningModule):
    def __init__(self,
                model_identifier, sft_model, rl_updated_model, reward_models, tokenizer, num_reward_models, sft_model_path, # model selections
                num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch, # model dependent hyperparameters
                seed, epochs, iterations, # training hyperparameters
                learning_rate, lr_mult, lr_mult_factor, # learning rate hyperparameters
                WD, grad_clip_threshold, grad_clip_threshold_factor, # optimizer hyperparameters
                WT, num_sequences, inc_batch_size, max_batch_size, num_mutations, high_conf_threshold, cum_prob_threshold, # generating design hyperparameters
                rel_to_WT, epsilon, # important PPO hyperparameters
                pairwise_hd_aver_factor, dkl_scale, dkl_scale_init, # total reward hyperparameters
                filepath, logger_version, # hyparameters regarding model saving
                epoch_threshold_to_unlock_ESM2):
        super().__init__()

        # Model selections
        self.model_identifier = model_identifier
        self.fixed_model = sft_model
        self.rl_updated_model = rl_updated_model
        self.reward_models = reward_models
        AAs = 'ACDEFGHIKLMNPQRSTVWY' # setup torchtext vocab to map AAs to indices for reward models
        aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
        aa2ind.set_default_index(20) # set unknown charcterers to gap
        self.aa2ind = aa2ind
        self.tokenizer = tokenizer
        self.num_reward_models = num_reward_models
        self.sft_model_path = sft_model_path

        # Hyperparameters regarding model saving
        self.ema = ExponentialMovingAverage(self.rl_updated_model.parameters(), decay=0.8)
        self.filepath = filepath
        self.logger_version = logger_version

        # Model dependent hyperparameters
        self.num_unfrozen_layers = num_unfrozen_layers
        self.num_layers_unfreeze_each_epoch = num_layers_unfreeze_each_epoch
        self.max_num_layers_unfreeze_each_epoch = max_num_layers_unfreeze_each_epoch
        named_esm2_layers = []
        self.rl_updated_model.to(self.device)
        for idx, (name, param) in enumerate(self.rl_updated_model.named_parameters()):
            if "contact_head" in name:
                continue # Skip layers associated with the contact head
            named_esm2_layers.append(name) # Append layer name
        named_esm2_layers.reverse()
        selected_layers = named_esm2_layers[0:self.num_unfrozen_layers]

        # Training hyperparameters
        self.seed = seed
        self.epochs = epochs
        self.epoch_threshold_to_unlock_ESM2 = epoch_threshold_to_unlock_ESM2
        self.iterations = iterations

        # Learning rate hyperparameters
        self.learning_rate = learning_rate
        self.learning_rate_0 = learning_rate
        self.lr_mult = lr_mult
        self.lr_mult_factor = lr_mult_factor

        # Optimizer hyperparameters and configure optimizer
        self.WD = WD
        self.grad_clip_threshold = grad_clip_threshold
        self.grad_clip_threshold_factor = grad_clip_threshold_factor
        self.automatic_optimization = False
        self.esm2_params = []
        for idx, name in enumerate(selected_layers):
            # print(f'{idx}: self.learning_rate = {self.learning_rate:.8f}, {name}')
            self.esm2_params += [{'params': [p for n, p in self.rl_updated_model.named_parameters() if n == name and p.requires_grad],
                            'lr': self.learning_rate}] # append layer parameters
            self.learning_rate *= self.lr_mult # update learning rate

        self.rl_updated_model.to('cpu') # Do not need to clear cache. 0 MB freed
        optimizers_config = self.configure_optimizers()
        self.optimizer = optimizers_config["optimizer"]
        self.scheduler = optimizers_config["lr_scheduler"]

        # Generating design hyperparameters
        self.WT = WT
        self.num_seqs = num_sequences
        self.inc_batch_size = inc_batch_size
        self.max_batch_size = max_batch_size
        self.num_muts = num_mutations
        self.high_conf_threshold = high_conf_threshold
        self.cum_prob_threshold = cum_prob_threshold

        # Important PPO hyperparameters
        self.eps = epsilon
        self.rel_to_WT = rel_to_WT

        # Total reward hyperparameters
        self.beta = dkl_scale
        self.beta_init = dkl_scale_init
        self.pairwise_hd_aver_factor = pairwise_hd_aver_factor

        # parameters for custom training
        self.init_log_probs_with_high_conf_mutations = None
        self.fixed_high_conf_seq = None
        self.fixed_sequences_with_high_confidence_mutations = None
        self.fixed_candidate_positions = None
        self.fixed_normalized_weights = None

        # Save hyperparameters, excluding certain arguments
        self.save_hyperparameters(ignore=["sft_model", "rl_updated_model", "reward_models", "tokenizer"]) # log hyperparameters to file

    def training_step(self, batch):
        current_beta = self.beta_init if self.current_epoch < 10 else self.beta
        print(f"iteration 1")

        # Generate single mutant log probs for fixed model during the first epoch
        if self.current_epoch == 0:
            self.init_log_probs = self.initial_log_probabilities()

        # Generate designs
        new_log_states = self.new_log_probabilities()
        dkl_value = self.Dkl_states(new_log_states)
        ratios, mean_hd_from_CreiLOV, fixed_probs, fixed_mutated_seqs, masked_pos, sampled_idxs, rl_high_conf_seq, rl_mutated_seqs, rl_high_conf_mutations, aver_num_masks_to_add_muts = self.action(new_log_states=new_log_states)
        pairwise_hd_aver, total_distance, num_pairs = self.average_pairwise_hamming_distance(rl_mutated_seqs)
        fitness_advantage, rel_WT_fitness = self.reward(rl_mutated_seqs, fixed_mutated_seqs)  # Calculate rewards for the current batch of sequences
        total_reward = (fitness_advantage + self.pairwise_hd_aver_factor*pairwise_hd_aver - current_beta * dkl_value)
        
        # Calculate PPO loss and backpropagate
        ppo_loss = (self.clipped_loss(ratios, total_reward)).mean()
        self.rl_updated_model.to(self.device)
        self.optimizer.zero_grad()
        ppo_loss.backward()

        # # Normalize gradients
        # for param in self.rl_updated_model.parameters():
        #     if param.grad is not None:
        #         param.grad /= (param.grad.norm() + 1e-6)

        # # Log gradient norms before clipping
        # print("Gradient Norms Before Clipping:")
        # for name, param in self.rl_updated_model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f"{name}: Grad Norm = {param.grad.norm().item()}")

        torch.nn.utils.clip_grad_norm_(self.rl_updated_model.parameters(), self.grad_clip_threshold)
        # clip_value = torch.nn.utils.clip_grad_norm_(self.rl_updated_model.parameters(), self.grad_clip_threshold)
        # print(f"Total Gradient Norm After Clipping: {clip_value}")

        # # Log gradient norms after clipping
        # print("Gradient Norms After Clipping:")
        # for name, param in self.rl_updated_model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f"{name}: Grad Norm = {param.grad.norm().item()}")


        self.optimizer.step()
        self.lr_scheduler_step(self.scheduler['scheduler'], 0, None)
        self.ema.to(self.device)
        self.ema.update()
        self.rl_updated_model.to('cpu')
        self.ema.to('cpu')
        
        # Clear the GPU memory cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Frees 16.66 GBs!

        self.log("kl_divergence", dkl_value, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("mean_ratio_initial_iter", ratios.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("median_ratio_initial_iter", ratios.median(), prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("pairwise_hd_aver", pairwise_hd_aver, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("mean_hd_from_CreiLOV", mean_hd_from_CreiLOV, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("fitness_advantage", fitness_advantage, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("rel_WT_fitness", rel_WT_fitness, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("total_reward", total_reward, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("ppo_loss_initial_iter", ppo_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log('num_muts', float(self.num_muts), on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('aver_num_masks_to_add_muts', aver_num_masks_to_add_muts, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('batch_size', float(self.num_seqs), on_step=True, on_epoch=False, prog_bar=False, logger=True)

        for _ in range(self.iterations - 1):
            print(f"iteration {_+2}")

            # Generate new probabilities for numerator of ratio term
            ratios = self.action(new_log_states=new_log_states, masked_pos=masked_pos, sampled_idxs=sampled_idxs, rl_high_conf_mutations=rl_high_conf_mutations, rl_high_conf_seq=rl_high_conf_seq, fixed_probs=fixed_probs)
            
            # Calculate PPO loss and backpropagate
            ppo_loss = (self.clipped_loss(ratios, total_reward)).mean() # Calculate PPO loss
            self.rl_updated_model.to(self.device)
            self.optimizer.zero_grad()
            ppo_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.rl_updated_model.parameters(), self.grad_clip_threshold/self.grad_clip_threshold_factor)
            self.optimizer.step()
            self.lr_scheduler_step(self.scheduler['scheduler'], 0, None)
            
            # skipping EMA beyond 1st iteration
            
            self.rl_updated_model.to('cpu')
            
            # Clear the GPU memory cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # Frees 22.8 GBs!

            # Log ratios and loss at end of trajectory
            mean_ratio_iter = ratios.mean()
            median_ratio_iter = ratios.median()
            ppo_loss_final_iter = ppo_loss
            
            self.log("mean_ratio_final_iter", mean_ratio_iter, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log("median_ratio_final_iter", median_ratio_iter, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log("ppo_loss_final_iter", ppo_loss_final_iter, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        # Use the logger version number in the filename
        self.save_rl_updated_esm2()
        print(f'Saving models at epoch {self.current_epoch}')

        return rel_WT_fitness
    
    def initial_log_probabilities(self, sequence=None):
        """ Computes log probabilities matrices (states) for CreiLOV
        Returns:
            initial_log_states (torch.FloatTensor): (length of protein = 119, length of amico acid dictionary for ESM2 = 33)
            log probabilities of initial states from pre-trained esm2
        """
        if sequence is None:
            sequence = self.WT
        
        # Pre-allocate a tensor filled with zeros for the initial log probabilities
        initial_log_states = torch.zeros((len(sequence), 20), dtype=torch.bfloat16).to(self.device)
        
        with torch.no_grad():
            # Move the fixed model to the GPU only when needed
            self.fixed_model.to(self.device)
            self.fixed_model.eval()
    
            for mask_pos in range(len(sequence)):
                # Mask the current position
                masked_sequence = self.mask_sequence(sequence, mask_pos)
                inputs = self.tokenizer(masked_sequence, return_tensors="pt").to(self.device)
                logits = self.fixed_model(**inputs).logits[:,:,4:24]
                log_probabilities = F.log_softmax(logits[0, mask_pos + 1], dim=-1)
                initial_log_states[mask_pos] = log_probabilities
    
        # Clear the GPU memory cache
        if torch.cuda.is_available():
            self.fixed_model.to('cpu')
            torch.cuda.empty_cache()  # Frees GPU memory
    
        if self.current_epoch == 0:
            self.generate_heatmap(self.WT, initial_log_states, self.model_identifier, sequence, f'./logs/{self.filepath}', self.logger_version, self.tokenizer)
        
        if sequence == self.WT:
            print(f'Saved heatmap for single mutant space from WT for sft model')
        else:
            print(f'Saved heatmap for single mutant space from sequence with high-confidence mutations for sft model')
    
        return initial_log_states

    def new_log_probabilities(self, sequence=None):
        """ Computes log probabilities matrices (states) for CreiLOV
        Returns:
             new_log_states (torch.FloatTensor, grad_fn=<CopySlices>): (length of protein = 119, length of amico acid dictionary for ESM2 = 33)
             log probabilities of new states after policy update (same as initial states for 1st epoch)
        """
        init_seq = sequence
        if sequence is None:
            sequence = self.WT

        # Pre-allocate a tensor filled with zeros for the new log probabilities
        new_log_states = torch.zeros((len(sequence), 20), dtype=torch.bfloat16).to(self.device)
        
        with torch.no_grad():
            # Move the fixed model to the GPU only when needed
            self.rl_updated_model.to(self.device)
            self.rl_updated_model.eval()
            for mask_pos in range(len(sequence)):
                masked_sequence = self.mask_sequence(sequence, mask_pos) # Mask the current position
                inputs = self.tokenizer(masked_sequence, return_tensors="pt").to(self.device)
                logits = self.rl_updated_model(**inputs).logits[:,:,4:24]
                log_probabilities = F.log_softmax(logits[0, mask_pos+1], dim=-1)
                new_log_states[mask_pos] = log_probabilities
            
            self.rl_updated_model.train()
            
            # Clear the GPU memory cache
            if torch.cuda.is_available():
                self.rl_updated_model.to('cpu')
                # Do not need to delete inputs, outputs, logits, log_probabilities and emtpy cache. 0.02 MB freed
                torch.cuda.empty_cache() # Frees 2.722 GB
            
            # Save heatmap at beginning of every 10 epochs for WT single mutant probability space (1st iteration)
            if self.current_epoch % 1 == 0 and init_seq is None:
                self.generate_heatmap(self.WT, new_log_states, self.model_identifier, self.WT, f'./logs/{self.filepath}', self.logger_version, self.tokenizer)
                print(f'Saved heatmap for single mutant space from WT for aligned model')

            
        return new_log_states

    def Dkl_states(self, new_log_states):
        """ Measure different in distribution along each amino acid index between initial and new states
        Args:
            new_log_states (torch.FloatTensor, grad_fn=<ClampBackward1>): (batch_size, length of amino acid dictionary = 21, length of protein = 119)
                log probabilities of new states after policy update
        Returns:
            kl_divergence (torch.FloatTensor): ([])
                kl_divergence between initial and new state matrices, starts at 0
        """
        # No backpropagation through Dkl
        new_log_states = new_log_states.clone()
        with torch.no_grad():
            kl_divergence = F.kl_div(input=new_log_states, target=self.init_log_probs, reduction='batchmean', log_target=True)

        # Deleting new_log_states onnly frees 0.01 MB

        return kl_divergence

    def action(self, new_log_states=None, masked_pos=None, sampled_idxs=None, rl_high_conf_mutations=None, rl_high_conf_seq=None, fixed_probs=None):
        """ Recursive sampling of state space using the rl_updated_model to design sequences
            Args:
                new_log_states (torch.FloatTensor, grad_fn=<CopySlices>): (length of protein = 119, length of amico acid dictionary for ESM2 = 33)
                    log probabilities of new states after policy update (same as initial states for 1st epoch)
                masked_pos = None or (torch.LongTensor): torch.Size([num_seqs, num_muts])
                sampled_idxs = None or (torch.LongTensor): torch.Size([num_seqs, num_muts])
                rl_high_conf_mutations : (dict, optional) Dictionary mapping positions to high-confidence mutations and their probabilities for the RL-updated model. 
                    Calculated during the first iteration of sampling.
                rl_high_conf_seq : (str, optional) String representation of the sequence containing high-confidence mutations for the RL-updated model.
                fixed_probs = None or (torch.FloatTensor): torch.Size([num_seqs, num_muts, num_aas=20])
            Returns:
                masked_pos (torch.LongTensor): torch.Size([num_seqs, num_muts])
                    positions of CreiLOV iteratively and randomly masked
                rl_probs (torch.FloatTensor): torch.Size([num_seqs, num_muts, num_aas=20])
                    log probabilities of rl updated model for position and amino acid mutated each mutation
                fixed_probs (torch.FloatTensor): torch.Size([num_seqs, num_muts, num_aas=20])
                    log probabilities of fixed model for position and amino acid mutated each mutation
                mutated_seqs (list): num_seqs
                    sequences designed by sampled probs from rl updated model
                sampled_idxs (torch.LongTensor): torch.Size([num_seqs, num_muts])
                    amino acid indices sampled by rl updated model
                sampled_aas (list of lists): num_seqs by num_muts
                    amino acids sampled by rl updated model
                ratios (torch.LongTensor, grad_fn=<CopySlices>): torch.Size([num_seqs, num_muts]) Starts at 1, ratio btw rl_probs / fixed_probs
        """
        # Initialize tensors for probabilities and positions
        all_tokens = list(self.tokenizer.get_vocab().keys())[4:24] # Get the list of all tokens for reference
        self.rl_updated_model.to(self.device)
        self.fixed_model.to(self.device)
            
        # Initialize variables if not provided
        if masked_pos is None:
            new_sampling = True
            while True:
                initial_num_muts = self.num_muts  # Store the initial value of num_muts

                # print('new_sampling')
                rl_high_conf_mutations, self.num_muts, self.high_conf_threshold = self.identify_high_conf_mutations(new_log_states, self.tokenizer, self.WT, self.high_conf_threshold, self.num_muts)
                fixed_high_conf_mutations, self.num_muts, self.high_conf_threshold = self.identify_high_conf_mutations(self.init_log_probs, self.tokenizer, self.WT, self.high_conf_threshold, self.num_muts)

                # Restart the action method if num_muts was updated
                if self.num_muts > initial_num_muts:
                    print(f"num_muts increased from {initial_num_muts} to {self.num_muts}. Restarting action step...")
                    continue  # Restart the loop

                # If num_muts is stable, exit the loop
                break

        else:
            new_sampling = False
            rl_probs = torch.zeros((self.num_seqs, self.num_muts, 20), dtype=torch.bfloat16).to(self.device)
            ratios = torch.zeros((self.num_seqs, self.num_muts), dtype=torch.bfloat16).to(self.device)

        # Find high-confidence mutations for both models in the 1st iteration of an epoch
        if new_sampling:
            fixed_mutated_seqs = [] # Mutated sequences from fixed model
            rl_mutated_seqs = [] # Mutated sequences from aligned model
            sampled_aas = [[] for _ in range(self.num_seqs)] # Initialize a list of lists for sampled amino acids
            masked_pos = torch.zeros((self.num_seqs, self.num_muts), dtype=torch.long).to(self.device)
            sampled_idxs = torch.zeros((self.num_seqs, self.num_muts), dtype=torch.long).to(self.device)
            fixed_probs = torch.zeros((self.num_seqs, self.num_muts, 20), dtype=torch.bfloat16).to(self.device)
            rl_probs = torch.zeros((self.num_seqs, self.num_muts, 20), dtype=torch.bfloat16).to(self.device)
            ratios = torch.zeros((self.num_seqs, self.num_muts), dtype=torch.bfloat16).to(self.device)
                
            # Calculate single mutant probability space for sequence with high confidence mutations from fixed model (constant throughout training)
            if self.current_epoch == 0:
                # Generate sequences for fixed model for 1st iteration of epoch
                fixed_mutated_seq = list(self.WT)
                for pos, mutations in fixed_high_conf_mutations.items():
                    max_token, max_prob = max(mutations, key=lambda x: x[1])
                    fixed_mutated_seq[pos - 1] = max_token
                self.fixed_high_conf_seq = "".join(fixed_mutated_seq)
                self.fixed_sequences_with_high_confidence_mutations = [self.fixed_high_conf_seq] * self.num_seqs
                print(f"Generated sequence with high confidence mutations from fixed model: {fixed_high_conf_mutations}")
                # print('f_seq, self.fixed_high_conf_seq)
                # print('WT', self.WT)

                self.init_log_probs_with_high_conf_mutations = self.initial_log_probabilities(sequence=self.fixed_high_conf_seq)
                self.fixed_candidate_positions, self.fixed_normalized_weights, self.cum_prob_threshold = self.identify_candidate_positions(self.init_log_probs_with_high_conf_mutations, self.WT, self.cum_prob_threshold, self.tokenizer)
                # print('Generated candidate positions from fixed model and normalized weights')
            else:
                self.fixed_sequences_with_high_confidence_mutations = [self.fixed_high_conf_seq] * self.num_seqs
                
            # Apply high-confidence mutations
            rl_mutated_seq = list(self.WT)
            positions_to_mask = list(rl_high_conf_mutations.keys())
            for pos, mutations in rl_high_conf_mutations.items():
                max_token, max_prob = max(mutations, key=lambda x: x[1])
                rl_mutated_seq[pos - 1] = max_token
            rl_high_conf_seq = "".join(rl_mutated_seq)
            rl_sequences_with_high_confidence_mutations = [rl_high_conf_seq] * self.num_seqs
            print(f"Generated sequence with high confidence mutations from aligned model: {rl_high_conf_mutations}")
            # print('r_seq', rl_high_conf_seq)
            # print('WT', self.WT)

            # Create masked sequences by masking the high-confidence mutation positions
            rl_mutated_seq = list(self.WT)
            for pos in positions_to_mask:
                rl_mutated_seq[pos - 1] = self.tokenizer.mask_token  # Adjust for 0-indexed list
            masked_rl_mutated_seq = "".join(rl_mutated_seq)

            # Perform single forward pass
            inputs = self.tokenizer([masked_rl_mutated_seq], return_tensors='pt').to(self.device)
            self.rl_updated_model.to(self.device)
            self.rl_updated_model.eval()
            rl_outputs = self.rl_updated_model(**inputs)
            self.rl_updated_model.train()
            rl_logits = rl_outputs.logits[:,:,4:24]
            # print('rl_logits', rl_logits)
            rl_log_probabilities = F.log_softmax(rl_logits, dim=-1)
            # print(f"Generated log probabilities for high confidence mutations from aligned model")
            
            # Process masked positions
            mut_idx = 0
            for pos in positions_to_mask:
                masked_pos[:, mut_idx] = pos - 1 # Adjust to 0-indexed tensor indexing
                # print('masked_pos[:, mut_idx]',masked_pos[:, mut_idx])
                
                rl_probs[:, mut_idx, :] = rl_log_probabilities.squeeze(0)[pos]  # Log probs at the masked position
                # print('shape of rl_log_probabilities', rl_log_probabilities.shape)
                # print(f"Position in rl_log_probabilities: {pos}, Values: {rl_log_probabilities.squeeze(0)[pos]}")

                # Get high-confidence mutation details
                mutations = rl_high_conf_mutations[pos]
                max_token, max_prob = max(mutations, key=lambda x: x[1])
                sampled_idx = self.tokenizer.convert_tokens_to_ids(max_token) - 4 # convert to valid amino acid indexes
                sampled_idxs[:, mut_idx] = sampled_idx
                # print(f"Max token: {max_token}, Token ID: {self.tokenizer.convert_tokens_to_ids(max_token)}, Sampled idx: {sampled_idx}")

                # Extract log probabilities for the sampled index
                rl_prob_at_sampled_idx = rl_probs[:, mut_idx, sampled_idx]
                # print('rl_prob_at_sampled_idx', rl_prob_at_sampled_idx)
                
                fixed_probs[:, mut_idx, :] = self.init_log_probs[pos - 1, :]
                # print('shape of self.init_log_probs', self.init_log_probs.shape)

                fixed_prob_at_sampled_idx = fixed_probs[:, mut_idx, sampled_idx]
                # print('fixed_prob_at_sampled_idx', fixed_prob_at_sampled_idx)

                # Calculate the ratio
                ratios[:, mut_idx] = torch.exp(rl_prob_at_sampled_idx - fixed_prob_at_sampled_idx).to(self.device)
                # print('ratios', ratios)
                mut_idx += 1  # Increment mutation index
            print(f"Generated ratios for high confidence mutations from aligned model")
                
        # Calculate rl_probs to calculate ratio for previously masked positions (high confidence mutations only)
        else:
            # Apply high-confidence mutations
            positions_to_mask = list(rl_high_conf_mutations.keys())
            rl_mutated_seq = list(self.WT)
            for pos in positions_to_mask:
                rl_mutated_seq[pos - 1] = self.tokenizer.mask_token  # Adjust for 0-indexed list
            masked_rl_mutated_seq = "".join(rl_mutated_seq)

            # Perform single forward pass
            inputs = self.tokenizer([masked_rl_mutated_seq], return_tensors='pt').to(self.device)
            self.rl_updated_model.eval()
            rl_outputs = self.rl_updated_model(**inputs)
            self.rl_updated_model.train()
            rl_logits = rl_outputs.logits[:,:,4:24]
            # print('rl_logits 2', rl_logits)
            rl_log_probabilities = F.log_softmax(rl_logits, dim=-1)
            # print(f"Generated log probabilities for high confidence mutations from aligned model")
            
            # Process masked positions
            mut_idx = 0
            for pos in positions_to_mask:
                rl_probs[:, mut_idx, :] = rl_log_probabilities.squeeze(0)[pos, :]  # Log probs at the masked position
                # print('shape of rl_log_probabilities', rl_log_probabilities.shape)

                # Get high-confidence mutation details
                sampled_idx = sampled_idxs[0, mut_idx]
                # print(f"Sampled idx: {sampled_idx}")

                # Extract log probabilities for the sampled index
                rl_prob_at_sampled_idx = rl_probs[:, mut_idx, sampled_idx]
                # print('rl_prob_at_sampled_idx 2', rl_prob_at_sampled_idx)

                fixed_probs[:, mut_idx, :] = self.init_log_probs[pos - 1, :]
                # print('shape of self.init_log_probs', self.init_log_probs.shape)

                fixed_prob_at_sampled_idx = fixed_probs[:, mut_idx, sampled_idx]
                # print('fixed_prob_at_sampled_idx', fixed_prob_at_sampled_idx)

                # Calculate the ratio
                ratios[:, mut_idx] = torch.exp(rl_prob_at_sampled_idx - fixed_prob_at_sampled_idx).to(self.device)
                # print('ratios 2', ratios)
                
                mut_idx += 1  # Increment mutation index
            print(f"Generated ratios for high confidence mutations from aligned model")
                
        # Calculate single mutant probability space for sequence with high confidence mutations from aligned model every iteration
        new_log_states_with_high_conf_mutations = self.new_log_probabilities(sequence=rl_high_conf_seq)
        
        # Identify positions using self.cum_prob_threshold to explore mutating for the 1st iteration of epoch and generate designs from aligned model
        if new_sampling:
            # Save heatmap every 5 epochs for rl_high_conf_seq single mutant probability space (1st iteration)
            if self.current_epoch % 1 == 0:
                self.generate_heatmap(self.WT, new_log_states_with_high_conf_mutations, self.model_identifier, rl_high_conf_seq, f'./logs/{self.filepath}', self.logger_version, self.tokenizer)
                print(f'Saved heatmap for single mutant space from sequence with high confidence mutations for aligned model')

            rl_candidate_positions, rl_normalized_weights, self.cum_prob_threshold = self.identify_candidate_positions(new_log_states_with_high_conf_mutations, self.WT, self.cum_prob_threshold, self.tokenizer, for_aligned_model=True)
            # print('Number of high confidence mutations from aligned model:', mut_idx)
            
            # Add mutations until num_muts of mutations relative to WT sequence are obtained for all sequences_with_high_confidence_mutations for aligned model
            seq_idx = 0
            num_masks_to_add_muts_list = []
            for seq in rl_sequences_with_high_confidence_mutations:
                mutated_seq = list(seq)
                mut_idx = self.hamming_distance(mutated_seq, self.WT)
                # print(f"Initial Hamming distance for sequence {seq_idx}: {mut_idx}")

                num_masks_to_add_muts = 0  # Initialize counter for this sequence
                while self.hamming_distance(mutated_seq, self.WT) < self.num_muts:
                    num_masks_to_add_muts += 1
                    # print('seq_idx', seq_idx)
                    # print('mut_idx', mut_idx)
                    
                    # Randomly choose a candidate position
                    selected_pos = random.choices(rl_candidate_positions, weights=rl_normalized_weights, k=1)[0]
                    # print(f"Selected position {selected_pos} for mutation in sequence {seq_idx}")
                    
                    # Calculate log prob for amino acid mutation for aligned model
                    mutated_seq[selected_pos] = self.tokenizer.mask_token  # Use <mask> token
                    masked_seq_str = ''.join(mutated_seq)
                    # print(f"Masked sequence: {masked_seq_str}")
                    
                    inputs = self.tokenizer(masked_seq_str, return_tensors="pt").to(self.device)
                    self.rl_updated_model.to(self.device)
                    self.rl_updated_model.eval()
                    rl_outputs = self.rl_updated_model(**inputs)
                    self.rl_updated_model.train()
                    rl_logits = rl_outputs.logits[0, selected_pos + 1, 4:24]
                    rl_log_probabilities_pos = F.log_softmax(rl_logits, dim=-1)
                    # print(f"Model log probabilities at position {selected_pos}: {rl_log_probabilities_pos}")

                    rl_probabilities_pos = torch.exp(rl_log_probabilities_pos).to(self.device)
                    # print(f"Model probabilities at position {selected_pos}: {rl_probabilities_pos}")
                    
                    sampled_idx = torch.multinomial(rl_probabilities_pos, 1).item()
                    # print('sampled_idx', sampled_idx)

                    new_amino_acid_id = sampled_idx + 4 # Map to actual token ID range for amino acids
                    new_amino_acid = self.tokenizer.convert_ids_to_tokens([new_amino_acid_id])[0]
                    mutated_seq[selected_pos] = new_amino_acid
                    # print('mutated_seq', mutated_seq)
                    # print(f"Mutated sequence after replacing position {selected_pos} with '{new_amino_acid}': {''.join(mutated_seq)}")
                    
                    # Calculate log prob for amino acid mutation for fixed model
                    with torch.no_grad():
                        self.fixed_model.eval()
                        self.fixed_model.to(self.device)
                        fixed_outputs = self.fixed_model(**inputs)
                        fixed_logits = fixed_outputs.logits[0, selected_pos + 1, 4:24]  # Adjust this range based on valid amino acid tokens
                        fixed_log_probabilities_pos = F.log_softmax(fixed_logits, dim=-1)
                        # print(f"Fixed model log probabilities at position {selected_pos}: {fixed_log_probabilities_pos}")
                        
                        fixed_probabilities_pos = torch.exp(fixed_log_probabilities_pos).to(self.device)
                        # print(f"Fixed model probabilities at position {selected_pos}: {fixed_probabilities_pos}")
                    
                    # If sequence mutated, calculate 
                    new_mut_idx = self.hamming_distance(mutated_seq, self.WT)
                    if new_mut_idx > mut_idx:
                        # print(f"Mutation at position {selected_pos} increased Hamming distance to {new_mut_idx}")
                        
                        # Update tracking arrays with mutation information
                        masked_pos[seq_idx, mut_idx] = selected_pos
                        # print('selected_pos', selected_pos)
                        sampled_idxs[seq_idx, mut_idx] = sampled_idx
                        # print('sampled_idx', sampled_idx)

                        rl_probs[seq_idx, mut_idx] = rl_log_probabilities_pos
                        rl_prob_at_sampled_idx = rl_probs[seq_idx, mut_idx][sampled_idx]
                        # print('rl_prob_at_sampled_idx', rl_prob_at_sampled_idx)

                        fixed_probs[seq_idx, mut_idx] = fixed_log_probabilities_pos
                        fixed_prob_at_sampled_idx = fixed_probs[seq_idx, mut_idx][sampled_idx]
                        # print('fixed_prob_at_sampled_idx', fixed_prob_at_sampled_idx)

                        ratios[seq_idx, mut_idx] = torch.exp(rl_prob_at_sampled_idx - fixed_prob_at_sampled_idx).to(self.device)
                        mut_idx = new_mut_idx
                    else:
                        # print(f"Mutation at position {selected_pos} did not change Hamming distance.")
                        del inputs, rl_outputs, rl_logits, rl_log_probabilities_pos, rl_probabilities_pos, 
                        torch.cuda.empty_cache() # Saves 279.34 MB each iteration with batch size of 20
                        
                # Convert tokenized mutated sequence back to amino acid string
                mutated_seq = ''.join(mutated_seq)
                rl_mutated_seqs.append(mutated_seq)
                num_masks_to_add_muts_list.append(num_masks_to_add_muts)
                seq_idx += 1

            # print('rl_mutated_seqs', rl_mutated_seqs)
            aver_num_masks_to_add_muts = sum(num_masks_to_add_muts_list) / len(num_masks_to_add_muts_list)
            # print(f"Average number of masks added per sequence: {aver_num_masks_to_add_muts:.2f}")
            # print(f"Generated sequences with {self.num_muts} mutations using aligned model:")
            # for idx, seq in enumerate(rl_mutated_seqs):
            #     print(f"Sequence {idx}: {seq}")
            
            # Generate designs with 5 mutations from fixed model
            mut_idx = self.hamming_distance(self.fixed_high_conf_seq, self.WT)
            seq_idx = 0
            for seq in self.fixed_sequences_with_high_confidence_mutations:
                mutated_seq = list(seq)
                # print('mutated_seq', mutated_seq)

                with torch.no_grad(): 
                    while self.hamming_distance(mutated_seq, self.WT) < self.num_muts:
                        # Randomly choose a candidate position
                        selected_pos = random.choices(self.fixed_candidate_positions, weights=self.fixed_normalized_weights, k=1)[0]
                        # print('selected_pos', selected_pos)
                        
                        # Calculate log prob for amino acid mutation for aligned model (if site is actually mutated)
                        mutated_seq[selected_pos] = self.tokenizer.mask_token  # Use <mask> token
                        masked_seq_str = ''.join(mutated_seq)
                        # print('masked_seq_str', masked_seq_str)
                        inputs = self.tokenizer(masked_seq_str, return_tensors="pt").to(self.device)
                        fixed_outputs = self.fixed_model(**inputs)
                        fixed_logits = fixed_outputs.logits[0, selected_pos + 1, 4:24]  # Adjust this range based on valid amino acid tokens
                        fixed_log_probabilities_pos = F.log_softmax(fixed_logits, dim=-1)
                        fixed_probabilities_pos = torch.exp(fixed_log_probabilities_pos).to(self.device)
                        # print('fixed_probabilities_pos', fixed_probabilities_pos)
                        sampled_idx = torch.multinomial(fixed_probabilities_pos, 1).item()
                        new_amino_acid_id = sampled_idx + 4  # Map to actual token ID range for amino acids
                        new_amino_acid = self.tokenizer.convert_ids_to_tokens([new_amino_acid_id])[0]
                        mutated_seq[selected_pos] = new_amino_acid
                        mut_idx = self.hamming_distance(mutated_seq, self.WT)
                    
                    # Convert tokenized mutated sequence back to amino acid string
                    mutated_seq = ''.join(mutated_seq)
                    fixed_mutated_seqs.append(mutated_seq)
                    seq_idx += 1
                
            # print('fixed_mutated_seqs', fixed_mutated_seqs)
            # print(f'Generated sequences with {self.num_muts} mutations using fixed model')
            # print('fixed_mutated_seqs', fixed_mutated_seqs)
            
            # Clear the GPU memory cache
            if torch.cuda.is_available():
                self.rl_updated_model.to('cpu')
                self.fixed_model.to('cpu')
                torch.cuda.empty_cache() # Frees 1.805 GB + 1.818 GB
                
            mean_hd_from_CreiLOV = torch.tensor(self.average_hamming_distance(rl_mutated_seqs), dtype=torch.bfloat16).to(self.device)
            # print('ratios: ', ratios)
            return ratios, mean_hd_from_CreiLOV, fixed_probs, fixed_mutated_seqs, masked_pos, sampled_idxs, rl_high_conf_seq, rl_mutated_seqs, rl_high_conf_mutations, aver_num_masks_to_add_muts
        
        # Calculate rl_probs to calculate ratio for previously masked positions (mutations after high confidence mutations only)
        else:
            # Add mutations until num_muts of mutations relative to WT sequence are obtained for all sequences_with_high_confidence_mutations for aligned model
            rl_sequences_with_high_confidence_mutations = [rl_high_conf_seq] * self.num_seqs
            seq_idx = 0
    
            for seq in rl_sequences_with_high_confidence_mutations:
                rl_mutated_seq = list(seq)
                # print('mutated_seq', mutated_seq)
                mut_idx = self.hamming_distance(rl_mutated_seq, self.WT)
                while mut_idx < self.num_muts:
                    # Calculate ratio every iteration to generate new log probs
                    # print('mut_idx', mut_idx)
                    # print('self.num_muts', self.num_muts)

                    pos = masked_pos[seq_idx, mut_idx]
                    # print('pos', pos)
                    # print("sampled_idx:", sampled_idxs[seq_idx, mut_idx])

                    rl_mutated_seq[pos] = self.tokenizer.mask_token
                    masked_seq_str = ''.join(rl_mutated_seq)  # Convert list to string
                    inputs = self.tokenizer(masked_seq_str, return_tensors="pt").to(self.device) # Tokenize and move to device
                    self.rl_updated_model.to(self.device)
                    rl_outputs = self.rl_updated_model(**inputs)
                    # print(f"Sequence index: {seq_idx}, Mutation index: {mut_idx}, Position: {pos}, Logits shape: {rl_outputs.logits.shape}")
                    
                    rl_logits = rl_outputs.logits[0, pos, 4:24] # Extract logits for the masked position
                    rl_log_probabilities_pos = F.log_softmax(rl_logits, dim=-1) # Convert to log probabilities
                    rl_probabilities_pos = torch.exp(rl_log_probabilities_pos).to(self.device)
                    # print(f"Position in rl_probabilities_pos: {pos}, Values: {rl_probabilities_pos}")
                    
                    # Calculate new ratio
                    rl_probs[seq_idx, mut_idx] = rl_probabilities_pos # insert logits for single position
                    rl_prob_at_sampled_idx = rl_probs[seq_idx, mut_idx, sampled_idxs[seq_idx, mut_idx]]
                    fixed_prob_at_sampled_idx = fixed_probs[seq_idx, mut_idx, sampled_idxs[seq_idx, mut_idx]]
                    ratios[seq_idx, mut_idx] = torch.exp(rl_prob_at_sampled_idx - fixed_prob_at_sampled_idx).to(self.device)
                    new_amino_acid_id = sampled_idxs[seq_idx, mut_idx]  # Map to actual token ID range for amino acids
                    new_amino_acid = self.tokenizer.convert_ids_to_tokens([new_amino_acid_id])[0]
                    rl_mutated_seq[pos] = new_amino_acid
                    mut_idx += 1 #2
                
                seq_idx += 1
            print(f'Generated ratios for sequences from aligned model')
                
            # Clear the GPU memory cache
            if torch.cuda.is_available():
                self.rl_updated_model.to('cpu')
                self.fixed_model.to('cpu')
                torch.cuda.empty_cache() # Frees 1.805 GB
            # print('ratios: ', ratios)
            return ratios

    def reward(self, mutated_seqs, pretrained_mutated_seqs):
        """ Calculate fitness for proteins created by the rl updated model
        Args:
            mutated_seqs (list): num_seqs
                    sequences designed by sampled probs from rl updated model
        Returns:
            reward (torch.FloatTensor): torch.Size([])
                fitness for the batch of sampled proteins
        """
        batch_size = len(mutated_seqs)  # Use mutated_sequences as the batch
        scores_tensor = torch.zeros((len(self.reward_models), batch_size), dtype=torch.float32).to(self.device)
        pre_scores_tensor = torch.zeros((len(self.reward_models), batch_size), dtype=torch.float32).to(self.device)

        # Load all reward models onto the GPU
        for model in self.reward_models:
            model.to(self.device)

        # Compute scores for mutated sequences
        with torch.no_grad():
            for i, model in enumerate(self.reward_models):
                model.eval()  # Set the model to evaluation mode

                for j, seq in enumerate(mutated_seqs):
                    score = model.predict(seq)[0][0]  # Extract score for the sequence from the current model
                    scores_tensor[i, j] = score

                for j, seq in enumerate(pretrained_mutated_seqs):
                    score = model.predict(seq)[0][0]  # Extract score for the sequence from the current model
                    pre_scores_tensor[i, j] = score

        # Unload all reward models from the GPU
        for model in self.reward_models:
            model.to('cpu')
        
        # Emptying cache frees 0 MB here

        # Calculate fitness
        predicted_WT_fitness = 4.1498 # Predicted WT score
        rl_fitness_per_sequence = torch.quantile(scores_tensor, 0.05, dim=0)
        pre_fitness_per_sequence = torch.quantile(pre_scores_tensor, 0.05, dim=0)
        print(f"RL-updated mean fitness: {rl_fitness_per_sequence.mean()}")
        print(f"Pre-trained mean fitness: {pre_fitness_per_sequence.mean()}")

        # Visualize how many sequences with predicted fitness greater than WT
        valid_fitness_mask = rl_fitness_per_sequence >= predicted_WT_fitness
        valid_fitness = torch.masked_select(rl_fitness_per_sequence, valid_fitness_mask)
        print(f"Fitness Values > Predicted WT: {valid_fitness}")

        # Compute the overall fitness score based on average_type
        rl_fitness = rl_fitness_per_sequence.max()
        rel_WT_fitness = rl_fitness / predicted_WT_fitness

        if self.rel_to_WT == 1:
            fitness_advantage = rel_WT_fitness
        else:
            fitness_advantage = ((rl_fitness - pre_fitness)/pre_fitness)*100

        self.current_rel_WT_fitness = rel_WT_fitness.item()

        # Deleting scores_tensor, pre_scores_tensor does not save any space

        return fitness_advantage, rel_WT_fitness

    def clipped_loss(self, ratios, total_reward):
        """ Computes clipped surrogate loss for update
        Args:
            ratios (torch.FloatTensor, grad_fn=<CopySlices>): torch.Size([num_seqs, num_muts])
                 Ratio is sum of probabilities from new_states divided by sum of probabilities from initial_states
                 (Prob of states from new policy / Prob of states from old policy)
            total_reward (torch.FloatTensor): torch.Size([])
                total reward calculated from the mean fitness of sequences and Dkl loss
        Returns:
            clipped_loss (torch.FloatTensor, grad_fn=<NegBackward0>): torch.Size([2, 2])
        """
        clipped_loss = -torch.min(ratios * total_reward, torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * total_reward)

        self.current_clipped_loss = clipped_loss.mean().item()

        return clipped_loss

    def configure_optimizers(self):
        """ Configure optimizers and optionally a scheduler with warm restarts. """
        optimizer = torch.optim.Adam(self.esm2_params, weight_decay = self.WD)
        # print(self.esm2_params)

        T_0 = max(1, int((self.epochs / 100) * self.iterations)) # Number of updates within the first cycle
        T_mult = 2  # interval between decay cycles is constant
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}


    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """ Manually steppings learning rate scheduler. """
        # optimizer = self.optimizers()
        # current_lr = optimizer.param_groups[0]['lr']
        # print(f"Learning rate before step: {current_lr:.6f}")
        scheduler.step()
        # new_lr = optimizer.param_groups[0]['lr']
        # print(f"Learning rate after step: {new_lr:.6f}")

    def on_train_epoch_end(self):

        """ This function manually steps the scheduler at the end of each epoch. """
        self.num_seqs = min(self.num_seqs + self.inc_batch_size, self.max_batch_size) # Increase batch size each epoch until max size reached
        
        if self.current_epoch % self.epoch_threshold_to_unlock_ESM2 == 0:
            self.learning_rate = self.learning_rate_0
            initial_num_layers = self.num_unfrozen_layers
            self.num_unfrozen_layers = min(self.max_num_layers_unfreeze_each_epoch,self.num_unfrozen_layers+self.num_layers_unfreeze_each_epoch)
            self.lr_mult *= self.lr_mult_factor
        
            # Setting up layers for training
            current_params = set()
            for group in self.optimizer.param_groups:
                current_params.update(set(group['params']))
            named_esm2_layers = []
            self.rl_updated_model.to(self.device)
            for idx, (name, param) in enumerate(self.rl_updated_model.named_parameters()):
                if "contact_head" in name:
                    continue # Skip layers associated with the contact head
                named_esm2_layers.append(name) # Append layer name
            named_esm2_layers.reverse()
            selected_layers = named_esm2_layers[0:self.num_unfrozen_layers]

            # Add new layer parameters to the optimizer without reinitializing it
            for name in selected_layers:
                layer_params = [p for n, p in self.rl_updated_model.named_parameters() if n == name and p.requires_grad and p not in current_params]
                if layer_params:
                    # # Print information about the selected layer
                    # print(f"Name = {name}, Learning Rate = {self.learning_rate:.8f}")

                    # # Check if all parameters in the layer require gradients
                    # params_require_grad = all(param.requires_grad for param in layer_params)
                    # if not params_require_grad:
                    #     print(f"Warning: Some parameters in {name} do not require gradients.")

                    # # Print each parameter in the layer
                    # for param_idx, param in enumerate(layer_params):
                    #     print(f"Param {param_idx}: {param}")

                    # Add parameters to the optimizer and update current_params
                    self.optimizer.add_param_group({'params': layer_params,'lr': self.learning_rate})
                    current_params.update(set(layer_params))
                
                # else:
                #     print(f"Layer {name} skipped: Either no parameters or already in optimizer.")

                self.learning_rate *= self.lr_mult
            


            if self.num_unfrozen_layers > initial_num_layers:
                print(f'Set up parameters for next epoch of training. Unlocked {self.num_unfrozen_layers-initial_num_layers} layers')
            else:
                print('Max number of layers unlocked')

        # Calculate max norm to monitor model collapse
        max_norm = 0
        for name, parameters in self.rl_updated_model.named_parameters():
            if parameters.requires_grad:
                param_norm = torch.norm(parameters.grad).item() if parameters.grad is not None else 0
                max_norm = max(max_norm, param_norm)
        self.log('max_norm', max_norm, on_epoch=True, prog_bar=True, logger=True)

        self.rl_updated_model.to('cpu')

        # Clear the GPU memory cache
        torch.cuda.empty_cache() # Saves 3.2 GB of space

    def save_rl_updated_esm2(self):
        """
        Save the state dictionary of the rl_updated_vae model to a file, for both the non-EMA and EMA-applied versions.
        """
        self.rl_updated_model.to(self.device)
        self.ema.to(self.device)
        
        version = self.logger_version if hasattr(self.logger, 'version') else 'unknown_version'
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        base_path = f'./logs/{self.filepath}_{self.model_identifier}/version_{version}'
        path_to_non_ema_model = f'{base_path}/non_ema_aligned_{self.model_identifier}_v{version}_ep{self.current_epoch}.pt'
        path_to_ema_model = f'{base_path}/ema_aligned_{self.model_identifier}_v{version}_ep{self.current_epoch}.pt'

        try:
            # # Save the non-EMA version of the model
            # torch.save(self.rl_updated_model.state_dict(), path_to_non_ema_model)
            # print(f"Saved non-EMA {self.model_identifier} model to {path_to_non_ema_model}")

            # Save the EMA version of the model
            self.ema.store(self.rl_updated_model.parameters())  # Store the original weights of rl_updated_model
            self.ema.copy_to(self.rl_updated_model.parameters())  # Apply EMA weights to rl_updated_model
            torch.save(self.rl_updated_model.state_dict(), path_to_ema_model)
            self.ema.restore(self.rl_updated_model.parameters())  # Restore the original weights after saving
            print(f"Saved EMA {self.model_identifier} model to {path_to_ema_model}")

        except Exception as e:
            print(f"An error occurred while saving the models: {e}")

        self.ema.to('cpu')
        self.rl_updated_model.to('cpu')

    def hamming_distance(self, s1, s2):
        """Calculates the Hamming distance between two sequences"""
        return sum(1 for x, y in zip(s1, s2) if x != y and x != '-' and y != '-') # Quantify sequence similarity

    def hamming_distance_tensor(self, t1, t2):
        """Calculate the Hamming distance between two tensors."""
        return torch.sum(t1 != t2)

    def average_hamming_distance(self, sequences):
        """Calculate the average pairwise Hamming distance among a list of protein sequences."""
        total_distance = 0
        num_pairs = 0
    
        # Iterate over all unique pairs of sequences
        for seqs in sequences:
            total_distance += self.hamming_distance(seqs, self.WT)
            num_pairs += 1
    
        # Calculate average distance
        average_distance = total_distance / num_pairs if num_pairs > 0 else 0

        ###### Does not save any memory to delete sequences ######

        return average_distance

    def average_pairwise_hamming_distance(self, mutated_seqs):
        """Calculate the average pairwise Hamming distance of a batch of protein sequences for all pairs."""
        batch_size = len(mutated_seqs)
        protein_tensors = torch.zeros((batch_size, len(self.WT)), dtype=torch.bfloat16).to(self.device)
        for i, seq in enumerate(mutated_seqs):
            protein_tensors[i] = torch.tensor(self.aa2ind(list(seq))).to(self.device)
            # print('protein_tensors', protein_tensors[i])

        n = protein_tensors.size(0)
        total_distance = 0
        num_pairs = 0
    
        # Iterate over all unique pairs
        for i, j in itertools.combinations(range(n), 2):
            total_distance += self.hamming_distance_tensor(protein_tensors[i], protein_tensors[j])
            num_pairs += 1
        average_distance = total_distance / num_pairs # Calculate average distance
        # print('average_distance', average_distance)

        ###### Does not save any memory to delete protein_tensors ######
        
        return average_distance, total_distance, num_pairs
    
    def hamming_distance(self, s1, s2):
        """Calculates the Hamming distance between two sequences"""
        return sum(1 for x, y in zip(s1, s2) if x != y and x != '-' and y != '-') # Quantify sequence similarity

    def identify_high_conf_mutations(self, log_probs, tokenizer, WT, high_conf_threshold, num_muts):
        """
        Identify high-confidence mutations based on probabilities exceeding the threshold.
        """
        max_high_conf_threshold = 0.99
        all_tokens = list(tokenizer.get_vocab().keys())[4:24]
        WT_token_ids = [tokenizer.convert_tokens_to_ids(wt) - 4 for wt in WT]

        while True:
            high_conf_mutations = {}
            # Identify high-confidence mutations
            for pos, wt_token_id in enumerate(WT_token_ids):
                pos_probs = torch.exp(log_probs[pos]).to(self.device)  # Convert log probabilities to probabilities
                high_conf_tokens = [
                    (all_tokens[token_id], prob.item())
                    for token_id, prob in enumerate(pos_probs)
                    if token_id != wt_token_id and prob > high_conf_threshold
                ]
                if high_conf_tokens:
                    high_conf_mutations[pos + 1] = high_conf_tokens  # Store as 1-indexed positions

            # If the number of high-confidence mutations is below num_muts, return the result
            if len(high_conf_mutations) < num_muts:
                break

            # If the high_conf_threshold has reached max_high_conf_threshold, increase num_muts and exit
            if high_conf_threshold >= max_high_conf_threshold:
                num_muts = len(high_conf_mutations) + 1
                print(f"Max threshold for high confidence mutations reached {high_conf_threshold:.2f}. Increasing num_muts to {num_muts}.")
                break

            # Otherwise, increase the high_conf_threshold
            high_conf_threshold = min(high_conf_threshold*1.01, max_high_conf_threshold)
            print(f"Increasing threshold to {high_conf_threshold:.2f} to reduce high-confidence mutations.")

        return high_conf_mutations, num_muts, high_conf_threshold
    
    def identify_candidate_positions(self, log_states, WT, cum_prob_threshold, tokenizer, for_aligned_model=False):
        """
        Identify candidate positions with cumulative probability > threshold for non-wildtype amino acids.
        Args:
            new_log_states (torch.Tensor): Log probabilities for each position (shape: num_positions x vocab_size).
            WT : Wild-type string.
            cum_prob_threshold (float): Threshold for cumulative probability to consider a position.
        Returns:
            rl_candidate_positions (list): Indices of candidate positions.
            rl_normalized_weights (list): Normalized weights for candidate positions
        """
        WT_tokens = [tokenizer.convert_tokens_to_ids(wt) - 4 for wt in WT]
        probabilities = torch.exp(log_states).to(self.device)

        # Re-identify candidate positions until there are at least 5 positions
        while True:
            rl_candidate_positions = []
            rl_position_weights = []
        
            # Calculate cumulative probability for non-wildtype amino acids
            for i, position_probs in enumerate(probabilities):
                non_wt_prob = position_probs.sum() - position_probs[WT_tokens[i]]
                if non_wt_prob > cum_prob_threshold:
                    rl_candidate_positions.append(i)
                    rl_position_weights.append(non_wt_prob.item())

            if len(rl_candidate_positions) >= 25:  # Stop if the number of candidate positions drops below threshold
                break

            # Decrease threshold by 5% if len(rl_candidate_positions) < 5
            cum_prob_threshold *= 0.99
            print(f"Threshold decreased to {cum_prob_threshold:.4f} due to insufficient candidate positions.")

        rl_total_weight = sum(rl_position_weights)
        rl_normalized_weights = [w / rl_total_weight for w in rl_position_weights] if rl_total_weight > 0 else []

        # Print detailed information about candidate positions
        if for_aligned_model:
            print(f"Number of candidate positions: {len(rl_candidate_positions)}")

        return rl_candidate_positions, rl_normalized_weights, cum_prob_threshold

    def mask_sequence(self, sequence, mask_pos):
        """Mask a single position in the sequence and return the masked sequence."""
        masked_sequence = list(sequence)
        masked_sequence[mask_pos] = '<mask>'  # Adjust for the <cls> token shift
        masked_seq_str = ''.join(masked_sequence)
        return masked_seq_str
    
    def get_mutations(self, seq, wt):
        """Find mutations and their positions"""
        mutations = [f"{wt_res}{i}{seq_res}" for i, (wt_res, seq_res) in enumerate(zip(wt, seq), 1) if seq_res != wt_res]
        if not mutations:
            return "WT"
        else:
            return "_".join(mutations)

    def generate_heatmap(self, WT, log_probabilities, model_identifier, sequence, filepath, version, tokenizer):
        """Generate and save a heatmap based on the predicted probabilities."""
        # Generate mutations relative to WT
        muts_rel_WT = self.get_mutations(sequence, WT)
        probabilities = torch.exp(log_probabilities).to(torch.float32).cpu()
    
        # Set up tokens and color map
        all_tokens = list(tokenizer.get_vocab().keys())[4:24]
        all_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in all_tokens]
    
        # Create heatmap
        plt.figure(figsize=(30, 6))
        Magma_r = plt.cm.magma_r(np.linspace(0, 1, 256))
        Magma_r[0] = [0, 0, 0, 0.03]
        cmap = LinearSegmentedColormap.from_list("Modified_Magma_r", Magma_r, N=256)
        heatmap = sns.heatmap(probabilities.detach().numpy().T, cmap=cmap, square=True, linewidths=0.003, linecolor='0.7', vmin=0, vmax=1)
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Predicted Amino Acid Probabilities at Each Position', fontsize=16)
        cbar.ax.tick_params(labelsize=12)
        plt.yticks(np.arange(20) + 0.5, all_tokens, fontsize=8, rotation=0)
        plt.xlabel("Position in sequence", fontsize=18)
        plt.ylabel('Tokens', fontsize=18)
        plt.title(f'Probabilities of single mutants for {muts_rel_WT} from {model_identifier}')
    
        # Add dark blue dots for WT residues and orange dots for mutations
        for pos, token in enumerate(sequence):  
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id in all_token_ids:  # Check if the token exists in the token list
                token_index = all_token_ids.index(token_id)
                dot_color = 'red' if token != WT[pos] else 'black' # Set dot color based on whether it matches WT or is a mutation
                plt.scatter(pos + 0.5, token_index + 0.5, color=dot_color, s=30)  # Adjust dot size as needed
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='WT'),
                           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Mutation')]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        if sequence is self.WT:
            plt.savefig(f'{filepath}_{model_identifier}/version_{version}/single_mut_probs_for_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}.png')
            plt.savefig(f'{filepath}_{model_identifier}/version_{version}/single_mut_probs_for_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}.svg')
        else:
            plt.savefig(f'{filepath}_{model_identifier}/version_{version}/single_mut_probs_for_high_conf_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}.png')
            plt.savefig(f'{filepath}_{model_identifier}/version_{version}/single_mut_probs_for_high_conf_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}.svg')
        plt.close()
        
    def print_tensor_devices(self):
        print("Tensors in RL Updated Model:")
        for name, param in self.rl_updated_model.named_parameters():
            print(f"{name}: {param.device}")
        print("Tensors in Fixed Model:")
        for name, param in self.fixed_model.named_parameters():
            print(f"{name}: {param.device}")
        for model in self.reward_models:
            print("Tensors in Reward Model:")
            for name, param in model.named_parameters():
                print(f"{name}: {param.device}")

    def log_tensor_device(self, tensor, tensor_name):
        print(f"{tensor_name} is on device: {tensor.device}")

    def capture_model_weights(self):
        """Captures and logs the initial weights of the model at the start of each epoch."""
        print("Capturing Model Weights at the Start of the Epoch:")
        for name, param in self.rl_updated_model.named_parameters():
            print(f"{name}: mean={param.mean().item()}, std={param.std().item()}")

class ProtDataModuleESM2(pl.LightningDataModule):
    def __init__(self, WT, batch_size, seed):
        super().__init__()
        self.wt_sequence = WT
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = ProtRepDatasetESM2(self.wt_sequence)

    def train_dataloader(self):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() %2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        return data_utils.DataLoader(
            self.train_ds,  # The dataset to load, in this case, the training dataset
            batch_size=self.batch_size,  # The number of samples in each batch to load
            shuffle=True,  # Enable shuffling to randomize the order of data before each epoch
            worker_init_fn=seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
            generator=generator,  # Specify the generator used for random number generation in shuffling
        )

class ProtRepDatasetESM2(torch.utils.data.Dataset):
    def __init__(self, wt_sequence):
        self.wt_sequence = wt_sequence

    def __len__(self):
        return 1 # 1 sequence

    def __getitem__(self, idx):
        # Return the protein sequence as a string and its length
        return self.wt_sequence, len(self.wt_sequence)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class ProtDataModuleESM2_DDP(pl.LightningDataModule):
    def __init__(self, WT, batch_size, seed):
        super().__init__()
        self.wt_sequence = WT
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = ProtRepDatasetESM2(self.wt_sequence)

    def train_dataloader(self):
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        else:
            generator.manual_seed(2549)

        # Detect if CUDA is available, and adjust the sampler accordingly
        if torch.cuda.is_available():
            print('Loading data to GPU')
            sampler = DistributedSampler(self.train_ds, shuffle=True)
            pin_memory = True
        else:
            print('Loading data to CPU')
            sampler = None  # No distributed sampling for CPU
            pin_memory = False

        return data_utils.DataLoader(
            self.train_ds,  # Dataset to load
            batch_size=self.batch_size,  # Number of samples in each batch
            sampler=sampler,  # DistributedSampler to split data across GPUs, if applicable
            shuffle=not torch.cuda.is_available(),  # Shuffle data on CPU
            num_workers=8,  # Adjust as needed depending on your system's resources
            worker_init_fn=seed_worker,  # Function to seed each worker
            generator=generator,  # Random number generator for shuffling
            pin_memory=pin_memory,  # Improve data loading performance if you're using GPUs
        )

class ProtRepDatasetESM2_DDP(torch.utils.data.Dataset):
    def __init__(self, wt_sequence):
        self.wt_sequence = WT

    def __len__(self):
        return 1 # 1 sequence

    def __getitem__(self, idx):
        # Return the protein sequence as a string and its length
        return self.wt_sequence, len(self.wt_sequence)



