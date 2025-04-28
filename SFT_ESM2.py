#!/usr/bin/env python
# coding: utf-8

# Import packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import pytorch_lightning as pl
import random
from random import choice
import os
import pickle
from transformers import AutoModelForMaskedLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                                   

# SeqFcnDataset is a data handling class.
class SeqSeqDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein sequence-sequence data"""

    def __init__(self, data_frame):
        self.data_df = data_frame

    def __getitem__(self, idx):
        sequence = self.data_df.iloc[idx]['Masked_Sequence']  # Extract masked sequence at index idx
        labels = self.data_df.iloc[idx]['Sequence']  # Extract original sequence at index idx
        return sequence, labels

    def __len__(self):
        return len(self.data_df)

# SFTDataModule handles data loading (updated for GPU)
class SFTDataModule(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data loading"""

    def __init__(self, data_frame, batch_size, seed):
        super().__init__()
        self.batch_size = batch_size
        self.data_df = data_frame
        self.seed = seed

    def setup(self, stage=None):
        # Assign datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_ds = SeqSeqDataset(self.data_df)

    def train_dataloader(self):
        # Function to initialize random seeds for each worker process
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32  # Compute a seed for the worker based on the initial seed of the torch Generator
            np.random.seed(worker_seed)  # Set NumPy's random seed based on the worker seed
            random.seed(worker_seed)  # Set Python's built-in random module's seed

        generator = torch.Generator()  # Create a new torch Generator
        generator.manual_seed(self.seed)  # Manually seed the generator with the predefined seed from the class
            
        # Create and return a DataLoader configured for training
        return data_utils.DataLoader(
            self.train_ds,  # The dataset to load, in this case, the training dataset
            batch_size=self.batch_size,  # The number of samples in each batch to load
            shuffle=True,  # Enable shuffling to randomize the order of data before each epoch
            worker_init_fn=seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
            generator=generator,  # Specify the generator used for random number generation in shuffling
            num_workers=32,  # The number of subprocesses to use for data loading. More workers can increase the speed of data loading
            pin_memory=True  # Pins memory, allowing faster and more efficient transfer of data from host to GPU when training on GPUs
        )

# Running SFT
class SFT_ESM2(pl.LightningModule):
    def __init__(self,
        WT, ESM2, reward_models,
        seed,
        learning_rate, lr_mult, lr_mult_factor,
        WD, grad_clip_threshold, 
        epochs,
        num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch,
        batch_size,
        random_masking, model_identifier):
        super().__init__()

        # fix random seeds for reproducibility
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # models for RLXF
        self.model_being_updated = ESM2.to(device)
        self.model_identifier = model_identifier
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{self.model_identifier}") #EsmTokenizer
        self.reward_models = [model.to(device) for model in reward_models]

        # hyperparameters
        self.stop_training_status = False
        self.learning_rate = learning_rate
        self.learning_rate_0 = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_unfrozen_layers = num_unfrozen_layers
        self.num_layers_unfreeze_each_epoch = num_layers_unfreeze_each_epoch
        self.max_num_layers_unfreeze_each_epoch = max_num_layers_unfreeze_each_epoch
        self.WD = WD
        self.grad_clip_threshold = grad_clip_threshold
        self.lr_mult = lr_mult
        self.lr_mult_factor = lr_mult_factor
        self.random_masking = random_masking
        
        # Setting up layers for training
        named_esm2_layers = []
        for idx, (name, param) in enumerate(self.model_being_updated.named_parameters()):
            if "contact_head" in name:
                continue # Skip layers associated with the contact head
            named_esm2_layers.append(name) # Append layer name
        named_esm2_layers.reverse()
        selected_layers = named_esm2_layers[0:self.num_unfrozen_layers]

        # store params & learning rates
        self.esm2_params = []
        for idx, name in enumerate(selected_layers):
            # print(f'{idx}: self.learning_rate = {self.learning_rate:.8f}, {name}')
            self.esm2_params += [{'params': [p for n, p in self.model_being_updated.named_parameters() if n == name and p.requires_grad],
                            'lr':     self.learning_rate}] # append layer parameters
            self.learning_rate *= self.lr_mult # update learning rate

        # parameters for custom training
        self.WT = WT
        self.automatic_optimization = False
        optimizers_config = self.configure_optimizers()
        self.optimizer = optimizers_config["optimizer"]
        self.scheduler = optimizers_config["lr_scheduler"]
        
        self.generated_sequences = set()

        self.save_hyperparameters(ignore=["ESM2", "reward_models"]) # log hyperparameters to file

    def training_step(self, batch, batch_idx):
        # Unpack the batch data
        masked_seqs, original_seqs = batch  # Get batch data

        if self.random_masking == 1:
            # Randomly add 15 masks to each sequence in masked_seqs
            masked_seqs = self.add_random_masks(masked_seqs, self.tokenizer, num_masks=20)

        inputs = self.tokenizer(masked_seqs, return_tensors='pt', padding=True)
        labels = self.tokenizer(original_seqs, return_tensors='pt', padding=True).input_ids

        # Move inputs and labels to the device (e.g., GPU) that is being used for training. 
        # `non_blocking=True` allows asynchronous memory copying to the device, improving performance.
        inputs = {key: val.to(device, non_blocking=True) for key, val in inputs.items()}
        labels = labels.to(device, non_blocking=True)

        # Pass the inputs to the model.
        outputs = self.model_being_updated(**inputs)
        logits = outputs.logits
        
        # Identify masked positions: this assumes <mask> is tokenized as a specific id
        mask_token_id = self.tokenizer.convert_tokens_to_ids('<mask>')
        mask = inputs['input_ids'] == mask_token_id

        # Filter logits and labels to consider only masked positions for loss calculation
        masked_logits = logits[mask]
        masked_labels = labels[mask]

        # Calculate loss only on masked positions
        loss = F.cross_entropy(masked_logits, masked_labels)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_being_updated.parameters(), self.grad_clip_threshold)
        self.optimizer.step()
        
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=self.batch_size)
        
        self.lr_scheduler_step(self.scheduler, 0, None)

        last_batch = batch_idx == self.trainer.num_training_batches - 1

    def add_random_masks(self, seqs, tokenizer, num_masks=15):
        mask_token = tokenizer.mask_token  # Get the actual mask token string from the tokenizer
        masked_seqs = []
        
        for seq in seqs:
            seq_tokens = self.tokenizer.tokenize(seq)  # Tokenize the sequence
            available_indices = [i for i, token in enumerate(seq_tokens) if token != mask_token]  # Avoid existing masks
            
            if len(available_indices) > 0:
                # Randomly select positions to mask
                mask_indices = np.random.choice(available_indices, min(len(available_indices), num_masks), replace=False)
                for idx in mask_indices:
                    seq_tokens[idx] = mask_token  # Replace selected token with mask
            
            # Join the tokens back to a sequence
            masked_seq = tokenizer.convert_tokens_to_string(seq_tokens)
            masked_seqs.append(masked_seq)
        
        return masked_seqs

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """ This function manually steps the scheduler. """
        scheduler['scheduler'].step()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.esm2_params, weight_decay=self.WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

    def on_train_epoch_end(self):
        """ Occurs at the end of each epoch """
        self.learning_rate = self.learning_rate_0
        self.num_unfrozen_layers = min(self.max_num_layers_unfreeze_each_epoch,self.num_unfrozen_layers+self.num_layers_unfreeze_each_epoch)

        # Collect all currently optimized parameters to avoid duplication
        current_params = set()
        for group in self.optimizer.param_groups:
            current_params.update(set(group['params']))

        # Increase layers for training
        named_esm2_layers = []
        for idx, (name, param) in enumerate(self.model_being_updated.named_parameters()):
            if "contact_head" in name:
                continue # Skip layers associated with the contact head
            named_esm2_layers.append(name) # Append layer name
        named_esm2_layers.reverse()
        selected_layers = named_esm2_layers[0:self.num_unfrozen_layers]

        # Add new layer parameters to the optimizer without reinitializing it
        for name in selected_layers:
            layer_params = [p for n, p in self.model_being_updated.named_parameters() if n == name and p.requires_grad and p not in current_params]
            if layer_params:
                self.optimizer.add_param_group({'params': layer_params,'lr': self.learning_rate})
                current_params.update(set(layer_params))
            self.learning_rate *= self.lr_mult

        # Report gradient max norm
        max_norm = 0
        for name, parameters in self.model_being_updated.named_parameters():
            if parameters.requires_grad:
                param_norm = torch.norm(parameters.grad).item() if parameters.grad is not None else 0
                max_norm = max(max_norm, param_norm)
        self.log('max_norm', max_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        
    def save_sft_updated_esm2(self, filepath='sft_updated_esm2.pt'):
        """ Save the state dictionary of the rl_updated_vae model to a file.
        Args:
            filepath (str): Path to the file where the state dictionary will be saved.
        """
        # Save the model's state_dict
        try:
            torch.save(self.model_being_updated.state_dict(), filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")




