# import packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from collections import OrderedDict
from torchtext import vocab # This package can give problems sometimes, it may be necessary to downgrade to a specific version
from pytorch_lightning.loggers import CSVLogger
from random import choice
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn import metrics
import torchmetrics
import enum
import argparse
from argparse import ArgumentParser
import os
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import csv
from matplotlib.colors import LinearSegmentedColormap
from torch_ema import ExponentialMovingAverage

# define data handling class
class SeqFcnDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for DPO protein sequence-function data"""

    def __init__(self, data_frame):
        self.data_df = data_frame
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dataset loaded on {self.device}")

    def __getitem__(self, idx):
        sequence = self.data_df.Sequence.iloc[idx]  # Directly get the sequence string for ESM2 tokenizer
        pretrained_ESM2_score = self.data_df.ESM2_score.iloc[idx]
        experimental_score = self.data_df.log_mean.iloc[idx]
        return sequence, pretrained_ESM2_score, experimental_score

    def __len__(self):
        return len(self.data_df)

# define datasplitting class
class ProtDataModule(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data splitting"""

    def __init__(self, data_frame, batch_size, seed=0):
        # Call the __init__ method of the parent class
        super().__init__()

        # Store the batch size
        self.data_df = data_frame
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dataset initialized with {len(self.data_df)} sequences.")

    def setup(self, stage=None):
        """Splits dataset into paired preferred and rejected samples using SeqFcnDataset."""
        if stage == "fit" or stage is None:
            preferred_df = self.data_df[self.data_df["Sample_Type"] == "preferred"]
            rejected_df = self.data_df[self.data_df["Sample_Type"] == "rejected"]
            # print('preferred_df', preferred_df)
            # print('rejected_df', rejected_df)

            # Ensure balanced pairs
            min_size = min(len(preferred_df), len(rejected_df)) # ! check
            preferred_df = preferred_df.sample(n=min_size, random_state=self.seed).reset_index(drop=True)
            rejected_df = rejected_df.sample(n=min_size, random_state=self.seed).reset_index(drop=True)
            print(f"Paired dataset size: {min_size} samples")

            # Convert to SeqFcnDataset instances
            preferred_dataset = SeqFcnDataset(preferred_df) # ! check
            rejected_dataset = SeqFcnDataset(rejected_df) # ! check

            # Store paired data as a list of tuples (without using a separate class)
            self.paired_data = list(zip(preferred_dataset, rejected_dataset))
            
    def seed_worker(worker_id, worker_info):
	    worker_seed = torch.initial_seed() % 2**32  # Compute a seed for the worker based on the initial seed of the torch Generator
	    np.random.seed(worker_seed)  # Set NumPy's random seed based on the worker seed
	    random.seed(worker_seed)  # Set Python's built-in random module's seed
            
    #The DataLoader object is created using the train_ds/val_ds/test_ds objects with the batch size set during initialization of the class and shuffle=True.
    def train_dataloader(self):
        # Determine if we're running on a GPU
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            generator = torch.Generator()  # Create a new torch Generator
            generator.manual_seed(self.seed)  # Manually seed the generator with the predefined seed from the class
            # Create and return a DataLoader configured for training
            print('Sending data to GPU')
            return data_utils.DataLoader(
                self.paired_data,  # The dataset to load, in this case, the training dataset
                batch_size=self.batch_size,  # The number of samples in each batch to load
                shuffle=True,  # Enable shuffling to randomize the order of data before each epoch
                worker_init_fn=self.seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
                generator=generator,  # Specify the generator used for random number generation in shuffling
                # num_workers=32,  # The number of subprocesses to use for data loading. More workers can increase the speed of data loading
                # pin_memory=True  # Pins memory, allowing faster and more efficient transfer of data from host to GPU when training on GPUs
            )
        else:
            print('Sending data to CPU')
            return data_utils.DataLoader(self.paired_data, batch_size=self.batch_size, shuffle=True)

# define DPO class
class finetuning_ESM2_with_DPO(pl.LightningModule):
    """PyTorch Lightning Module that defines model and training"""
      
    # define network
    def __init__(self,
                 ESM2, huggingface_identifier, tokenizer, num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch,
                 epochs, batch_size, seed, patience,
                 beta, adam_beta_1, adam_beta_2, adam_eps, WD,
                 learning_rate,
                 WT, slen,
                 using_EMA, decay,
                 data, logger_version,
                 using_sft_model
                ):
        super().__init__()
        print("Initializing module...")

        # models hyperparameters
        self.ESM2 = ESM2
        self.huggingface_identifier = huggingface_identifier
        self.tokenizer = tokenizer
        self.num_unfrozen_layers = num_unfrozen_layers
        self.num_layers_unfreeze_each_epoch = num_layers_unfreeze_each_epoch
        self.max_num_layers_unfreeze_each_epoch = max_num_layers_unfreeze_each_epoch
        self.decay = decay
        self.ema = ExponentialMovingAverage(self.ESM2.parameters(), decay=self.decay)
        self.ESM2 = ESM2.to(self.device)
        
        # hyperparameters for training
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.patience = patience

        # DPO hyperparameters
        self.beta = beta
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.adam_eps = adam_eps
        self.WD = WD
        
        # learning rate hyperparameters
        self.learning_rate = learning_rate

        # data hyperparameters
        self.WT = WT
        self.slen = slen # synthetic query sequence length (128 a.a.)

        # Setting up layers for training
        named_esm2_layers = []
        for idx, (name, param) in enumerate(self.ESM2.named_parameters()):
            if "contact_head" in name:
                continue # Skip layers associated with the contact head
            named_esm2_layers.append(name) # Append layer name
        named_esm2_layers.reverse()
        selected_layers = named_esm2_layers[0:self.num_unfrozen_layers]

        # store params & learning rates
        self.esm2_params = []
        for idx, name in enumerate(selected_layers):
            # print(f'{idx}: self.learning_rate = {self.learning_rate:.8f}, {name}')
            self.esm2_params += [{'params': [p for n, p in self.ESM2.named_parameters() if n == name and p.requires_grad],
                            'lr':     self.learning_rate}] # append layer parameters

        # parameters for custom training
        self.stop_training_status = False
        self.automatic_optimization = False
        optimizers_config = self.configure_optimizers()
        self.optimizer = optimizers_config["optimizer"]
        self.scheduler = optimizers_config["lr_scheduler"]
        self.using_EMA = using_EMA
        if self.using_EMA == 1:
        	self.decay = decay
        	self.ema = ExponentialMovingAverage(self.ESM2.parameters(), decay=self.decay)

        # saving hyperparamters
        self.data = data
        self.logger_version = logger_version

        self.save_hyperparameters(ignore=["ESM2","tokenizer"])

    def training_step(self, batch, batch_idx):

        if self.current_epoch == 0:
            initial_log_states = self.calculate_log_probabilities(self.WT)
            self.generate_heatmap(self.WT, initial_log_states, self.huggingface_identifier, self.WT, f'DPO_{self.huggingface_identifier}_with_{self.data}', self.logger_version, self.tokenizer)
            
        # Unpack the batch data
        preferred_batch, rejected_batch = batch
        # print("preferred_batch device:", preferred_batch.device)
        
        # Unpacking batch tuples
        preferred_sequences, preferred_pretrained_ESM2_scores, preferred_experimental_scores = preferred_batch
        rejected_sequences, rejected_pretrained_ESM2_scores, rejected_experimental_scores = rejected_batch
        # print("preferred_pretrained_ESM2_scores device:", preferred_pretrained_ESM2_scores.device)
        
        # Compute aligned ESM2 scores using the log likelihood function
        self.ESM2.to(self.device)
        preferred_sequences = list(preferred_sequences)
        rejected_sequences = list(rejected_sequences)
        preferred_aligned_ESM2_log_probs = self.log_likelihood(preferred_sequences, no_grad=False)
        rejected_aligned_ESM2_log_probs = self.log_likelihood(rejected_sequences, no_grad=False)
        print("preferred_aligned_ESM2_log_probs device:", preferred_aligned_ESM2_log_probs.device)
        
        # Compute ratios
        preferred_ratios = self.beta * (preferred_aligned_ESM2_log_probs - preferred_pretrained_ESM2_scores)
        rejected_ratios = self.beta * (rejected_aligned_ESM2_log_probs - rejected_pretrained_ESM2_scores)
        print("preferred_ratios device:", preferred_ratios.device)
    
        # Compute loss using log-sigmoid difference
        DPO_loss = -F.logsigmoid(preferred_ratios - rejected_ratios)
        print("DPO_loss device:", DPO_loss.device)

        average_DPO_loss = torch.mean(DPO_loss)  # Average over batch
        print("average_DPO_loss device:", average_DPO_loss.device)

        # Backpropagation
        self.optimizer.zero_grad()
        average_DPO_loss.backward()
        self.optimizer.step()
        self.lr_scheduler_step(self.scheduler, 0, None)
        self.ema.to(self.device)
        self.ema.update()
        self.ESM2.to('cpu')
        self.ema.to('cpu')

        self.log('DPO_loss', average_DPO_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log('mean_preferred_ratios', preferred_ratios.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log('mean_rejected_ratios', rejected_ratios.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)

    def log_likelihood(self, sequences, no_grad=True):
        '''Adatped from the following preprint/github:
                @misc{stocco2024guidinggenerativeproteinlanguage,
                  title={Guiding Generative Protein Language Models with Reinforcement Learning}, 
                  author={Filippo Stocco and Maria Artigues-Lleixa and Andrea Hunklinger and Talal Widatalla and Marc Guell and Noelia Ferruz},
                  year={2024},
                  eprint={2412.12979},
                  archivePrefix={arXiv},
                  primaryClass={q-bio.BM},
                  url={https://arxiv.org/abs/2412.12979}, }
          '''
        
        all_loss = []  # List to store loss for each sequence

        for sequence in sequences:
            if no_grad:
                with torch.no_grad():  # Disable gradients for inference
                    inputs = self.tokenizer.encode(sequence, return_tensors='pt').to(self.device)  # Ensure inputs are on the correct device
                    outputs = self.ESM2(inputs, labels=inputs)
                    loss, logits = outputs[:2]
                    all_loss.append(loss.unsqueeze(0).to(self.device))  # Ensure loss is on device

            else:
                inputs = self.tokenizer.encode(sequence, return_tensors='pt').to(self.device)  # Ensure inputs are on the correct device
                outputs = self.ESM2(inputs, labels=inputs)
                loss, logits = outputs[:2]
                all_loss.append(loss.unsqueeze(0).to(self.device))  # Ensure loss is on device
        
        # Ensure all losses are on the correct device before concatenating
        all_loss = torch.cat(all_loss).to(self.device)

        return all_loss

    def calculate_log_probabilities(self, sequence=None):
        """ Computes log probabilities matrices (states) for CreiLOV
        Returns:
            initial_log_states (torch.FloatTensor): (length of protein = 119, length of amico acid dictionary for ESM2 = 33)
            log probabilities of initial states from pre-trained esm2
        """
        if sequence is None:
            sequence = self.WT
        
        # Pre-allocate a tensor filled with zeros for the initial log probabilities
        # all_tokens = self.tokenizer.get_vocab().keys()
        log_states = torch.zeros((len(sequence), 20), dtype=torch.bfloat16).to(self.device)
        
        with torch.no_grad():
            # Move the fixed model to the GPU only when needed
            self.ESM2.to(self.device)
            self.ESM2.eval()
    
            for mask_pos in range(len(sequence)):
                # Mask the current position
                masked_sequence = self.mask_sequence(sequence, mask_pos)
                inputs = self.tokenizer(masked_sequence, return_tensors="pt").to(self.device)
                logits = self.ESM2(**inputs).logits[:,:,4:24]
                log_probabilities = F.log_softmax(logits[0, mask_pos + 1], dim=-1)
                log_states[mask_pos] = log_probabilities
    
        # Clear the GPU memory cache
        if torch.cuda.is_available():
            self.ESM2.to('cpu')
            torch.cuda.empty_cache()  # Frees GPU memory
    
        return log_states

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
            plt.savefig(f'./logs/{filepath}/version_{version}/single_mut_probs_for_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}.png')
            plt.savefig(f'./logs/{filepath}/version_{version}/single_mut_probs_for_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}.svg')
        else:
            plt.savefig(f'./logs/{filepath}/version_{version}/single_mut_probs_for_high_conf_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}.png')
            plt.savefig(f'./logs/{filepath}/version_{version}/single_mut_probs_for_high_conf_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}.svg')
        plt.close()

        print(f'Created heatmap at epoch {self.current_epoch}')

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
        
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """ This function manually steps the scheduler. """
        scheduler['scheduler'].step()

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.esm2_params, weight_decay=self.WD)
        optimizer = torch.optim.AdamW(self.esm2_params, betas=(self.adam_beta_1, self.adam_beta_2), eps=self.adam_eps, weight_decay=self.WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler}}

    def on_train_epoch_end(self):
        """ Occurs at the end of each epoch """
        self.num_unfrozen_layers = min(self.max_num_layers_unfreeze_each_epoch,self.num_unfrozen_layers+self.num_layers_unfreeze_each_epoch)

        # Collect all currently optimized parameters to avoid duplication
        current_params = set()
        for group in self.optimizer.param_groups:
            current_params.update(set(group['params']))

        # Increase layers for training
        named_esm2_layers = []
        for idx, (name, param) in enumerate(self.ESM2.named_parameters()):
            if "contact_head" in name:
                continue # Skip layers associated with the contact head
            named_esm2_layers.append(name) # Append layer name
        named_esm2_layers.reverse()
        selected_layers = named_esm2_layers[0:self.num_unfrozen_layers]

        # Add new layer parameters to the optimizer without reinitializing it
        for name in selected_layers:
            layer_params = [p for n, p in self.ESM2.named_parameters() if n == name and p.requires_grad and p not in current_params]
            if layer_params:
                self.optimizer.add_param_group({'params': layer_params,'lr': self.learning_rate})
                current_params.update(set(layer_params))

        optimizers_config = self.configure_optimizers()
        self.optimizer = optimizers_config["optimizer"]
        self.scheduler = optimizers_config["lr_scheduler"]

        # Report gradient max norm
        max_norm = 0
        for name, parameters in self.ESM2.named_parameters():
            if parameters.requires_grad:
                param_norm = torch.norm(parameters.grad).item() if parameters.grad is not None else 0
                max_norm = max(max_norm, param_norm)
        self.log('max_norm', max_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        log_states = self.calculate_log_probabilities(self.WT)
        self.generate_heatmap(self.WT, log_states, self.huggingface_identifier, self.WT, f'DPO_{self.huggingface_identifier}_with_{self.data}', self.logger_version, self.tokenizer)
        
    def save_model(self, save_path, ema_save_path):
	    """
	    Save two versions of the model's state dictionary:
	    1. Non-EMA applied parameters.
	    2. EMA-applied parameters for ESM2_wo_lmhead.
	    """
	    # Save non-EMA version of the state_dict
	    try:
	        torch.save(self.state_dict(), save_path)
	        print(f"Non-EMA model saved to {save_path}")
	    except Exception as e:
	        print(f"An error occurred while saving the non-EMA model: {e}")

	    # Save EMA-applied version of the state_dict
	    if self.using_EMA == 1:
	        self.ema.to(self.device)  # Ensure EMA is on the same device
	        try:
	            # Store the original ESM2_wo_lmhead parameters
	            self.ema.store(self.ESM2.parameters())

	            # Apply EMA weights to ESM2_wo_lmhead
	            self.ema.copy_to(self.ESM2.parameters())

	            # Save the state_dict with EMA weights applied
	            torch.save(self.state_dict(), ema_save_path)
	            print(f"EMA model saved to {ema_save_path}")

	            # Restore the original ESM2_wo_lmhead parameters
	            self.ema.restore(self.ESM2.parameters())
	        except Exception as e:
	            print(f"An error occurred while saving the EMA model: {e}")
	        finally:
	        	self.ema.to('cpu')
