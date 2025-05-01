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
import torchmetrics
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
from transformers import AutoModelForMaskedLM, AutoTokenizer
from functions import generate_and_evaluate_mutants_max_sampling
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from MLP import MLP
import itertools
import copy
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define amino acid dictionary for tokenization, define WT for length of context window
AAs = 'ACDEFGHIKLMNPQRSTVWY' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
sequence_length = len(WT)

# SeqFcnDataset is a data handling class.
class SeqSeqDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein sequence-sequence data"""

    def __init__(self, data_frame):
        self.data_df = data_frame

    def __getitem__(self, idx):
        sequence = self.data_df.iloc[idx]['Masked_Sequence']  # Extract masked sequence at index idx
        labels = self.data_df.iloc[idx]['Sequence']  # Extract original sequence at index idx
        weights = (self.data_df.iloc[idx]['Fitness']-4.1498)/(4.201938-4.1498) # Extract log_mean score for original sequence, normalize by min and max predited scores
        # weights = (self.data_df.iloc[idx]['log_mean']-4.1498)/(4.201938-4.1498) # Extract log_mean score for original sequence, normalize by min and max predited scores
        return sequence, labels, weights

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
    def __init__(self, ESM2, reward_models,
                    seed,
                    learning_rate, lr_mult, lr_mult_factor, use_scheduler, warm_restart, reinit_optimizer,
                    WD, grad_clip_threshold, 
                    epochs,
                    num_unfrozen_layers, num_layers_unfreeze_each_epoch, max_num_layers_unfreeze_each_epoch, training_pos_emb,
                    batch_size,
                    dataset, dataset_version, use_weights, random_masking, model_identifier,
                    filepath, logger_version):
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
        self.training_pos_emb = training_pos_emb
        self.WD = WD
        self.grad_clip_threshold = grad_clip_threshold
        self.lr_mult = lr_mult
        self.lr_mult_factor = lr_mult_factor
        self.reinit_optimizer = reinit_optimizer
        self.use_weights = use_weights
        self.random_masking = random_masking
        
        # Setting up layers for training
        named_esm2_layers = []
        for idx, (name, param) in enumerate(self.model_being_updated.named_parameters()):
            if "contact_head" in name:
                continue # Skip layers associated with the contact head
            named_esm2_layers.append(name) # Append layer name
        named_esm2_layers.reverse()
        selected_layers = named_esm2_layers[0:self.num_unfrozen_layers]

        if (self.training_pos_emb == 1 and self.max_num_layers_unfreeze_each_epoch < 103):
            # print("here 1")
            selected_layers.append('esm.embeddings.position_embeddings.weight')

        # store params & learning rates
        self.esm2_params = []
        for idx, name in enumerate(selected_layers):
            # print(f'{idx}: self.learning_rate = {self.learning_rate:.8f}, {name}')
            self.esm2_params += [{'params': [p for n, p in self.model_being_updated.named_parameters() if n == name and p.requires_grad],
                            'lr':     self.learning_rate}] # append layer parameters
            self.learning_rate *= self.lr_mult # update learning rate

        # parameters for custom training
        self.WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
        self.automatic_optimization = False
        self.use_scheduler = use_scheduler
        self.warm_restart = warm_restart
        optimizers_config = self.configure_optimizers()
        if self.use_scheduler == 1:
            self.optimizer = optimizers_config["optimizer"]
            self.scheduler = optimizers_config["lr_scheduler"]
        else:
            self.optimizer = optimizers_config
        self.generated_sequences = set()
        self.filepath = filepath
        self.logger_version = logger_version

        self.save_hyperparameters(ignore=["ESM2", "reward_models"]) # log hyperparameters to file

    def training_step(self, batch, batch_idx):
        # Unpack the batch data
        masked_seqs, original_seqs, weights = batch  # Get batch data

        # Calculate single mutant probability space for sequence with high confidence mutations from aligned model every iteration
        new_log_SM_states = self.new_log_probabilities(sequence=self.WT)
        self.generate_heatmap(self.WT, new_log_SM_states, self.model_identifier, WT, f'./logs/{self.filepath}', self.logger_version, self.tokenizer, batch_idx)
        print(f'Saved heatmap for single mutant space for WT from aligned model')

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
        
        # # Print the shapes of the tensors to check their sizes
        # print(f"mask shape: {mask.shape}")  # Shape of the mask
        # print(f"mask (masked sequence positions): {mask}")  # Actual mask tensor

        # # Logits and labels length comparison
        # print(f"logits shape: {logits.shape}")  # Shape of the logits
        # print(f"labels shape: {labels.shape}")  # Shape of the labels

        # # Optionally print the actual logits and labels, but keep in mind it might be large depending on batch size
        # print(f"logits: {logits}")  # Logits tensor
        # print(f"labels: {labels}")  # Labels tensor

        # # Additional logging to see the batch size and lengths of input sequences
        # print(f"Sequence size: {len(masked_seqs[0])}")
        # print(f"Masked sequence: {masked_seqs}")

        # Filter logits and labels to consider only masked positions for loss calculation
        masked_logits = logits[mask]
        masked_labels = labels[mask]

        # Calculate loss only on masked positions
        loss = F.cross_entropy(masked_logits, masked_labels)

        # Backpropagate through weighted loss
        if self.use_weights == 1:
            sequence_indices = mask.nonzero(as_tuple=True)[0]  # This extracts the batch indices part of masked_indices
            losses = F.cross_entropy(masked_logits, masked_labels, reduction='none')  # Calculate loss without reduction
            weighted_losses = losses * weights[sequence_indices].float()
            final_loss = weighted_losses.mean()

            # Backpropagation
            self.optimizer.zero_grad()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_being_updated.parameters(), self.grad_clip_threshold)
            self.optimizer.step()

            self.log('final_loss', final_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=self.batch_size)
        else:
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_being_updated.parameters(), self.grad_clip_threshold)
            self.optimizer.step()

            self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=self.batch_size)
        
        if self.use_scheduler == 1:
            self.lr_scheduler_step(self.scheduler, 0, None)

        last_batch = batch_idx == self.trainer.num_training_batches - 1
        if last_batch:
            # Calculate single mutant probability space for sequence with high confidence mutations from aligned model every iteration
            new_log_SM_states = self.new_log_probabilities(sequence=self.WT)
            self.generate_heatmap(self.WT, new_log_SM_states, self.model_identifier, WT, f'./logs/{self.filepath}', self.logger_version, self.tokenizer, batch_idx + 1)
            print(f'Saved heatmap for single mutant space for WT from aligned model after last update')

        #     print(f"Generating scores for sequences from model for logger...")
        #     try:
        #         if self.current_epoch == self.epochs - 1:
        #             _, sft_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=1000,
        #                                                                         num_muts=5,
        #                                                                         WT=self.WT,
        #                                                                         reward_models=self.reward_models,
        #                                                                         model=self.model_being_updated,
        #                                                                         seed=42)
        #         else:
        #             _, sft_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=1000,
        #                                                                         num_muts=5,
        #                                                                         WT=self.WT,
        #                                                                         reward_models=self.reward_models,
        #                                                                         model=self.model_being_updated,
        #                                                                         seed=42)

        #         # Calculate statistics
        #         scores = np.median(sft_scores_np, axis=0)
        #         max_score = scores.max()
        #         median_score = np.median(scores)
        #         mean_score = scores.mean()

        #         self.current_mean_score = mean_score.item()
        #         self.current_max_score = max_score.item()

        #         # Log statistics
        #         self.log('max_score', max_score, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=self.batch_size)
        #         self.log('median_score', median_score, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=self.batch_size)
        #         self.log('mean_score', mean_score, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=self.batch_size)

        #     except Exception as e:
        #         print(f"Failed to generate or evaluate mutants: {str(e)}")

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
        if self.use_scheduler == 1:
            if self.warm_restart == 1:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
                return {"optimizer": optimizer,
                        "lr_scheduler": {"scheduler": scheduler}}
            
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        
        return optimizer # No scheduler

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

        if (self.training_pos_emb == 1 and self.max_num_layers_unfreeze_each_epoch < 103):
            # print("here 2")
            selected_layers.append('esm.embeddings.position_embeddings.weight')

        # Add new layer parameters to the optimizer without reinitializing it
        for name in selected_layers:
            layer_params = [p for n, p in self.model_being_updated.named_parameters() if n == name and p.requires_grad and p not in current_params]
            if layer_params:
                self.optimizer.add_param_group({'params': layer_params,'lr': self.learning_rate})
                current_params.update(set(layer_params))
            self.learning_rate *= self.lr_mult

        if self.reinit_optimizer == 1:
            optimizers_config = self.configure_optimizers()
            if self.use_scheduler == 1:
                self.optimizer = optimizers_config["optimizer"]
                self.scheduler = optimizers_config["lr_scheduler"]
            else:
                self.optimizer = optimizers_config

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

    def generate_heatmap(self, WT, log_probabilities, model_identifier, sequence, filepath, version, tokenizer, batch_idx=None):
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
        
        if batch_idx is None:
            if sequence is self.WT:
                plt.savefig(f'{filepath}/version_{version}/single_mut_probs_for_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}.png')
                plt.savefig(f'{filepath}/version_{version}/single_mut_probs_for_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}.svg')
            else:
                plt.savefig(f'{filepath}/version_{version}/single_mut_probs_for_high_conf_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}.png')
                plt.savefig(f'{filepath}/version_{version}/single_mut_probs_for_high_conf_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}.svg')
            plt.close()
        else:
            if sequence is self.WT:
                # Save numpy file
                np.save(f'{filepath}/version_{version}/single_mut_probs_for_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}_batch_idx{batch_idx}.npy', 
                       probabilities.detach().numpy())

                # save figure
                plt.savefig(f'{filepath}/version_{version}/single_mut_probs_for_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}_batch_idx{batch_idx}.png')
                plt.savefig(f'{filepath}/version_{version}/single_mut_probs_for_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}_batch_idx{batch_idx}.svg')
            else:
                plt.savefig(f'{filepath}/version_{version}/single_mut_probs_for_high_conf_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}_batch_idx{batch_idx}.png')
                plt.savefig(f'{filepath}/version_{version}/single_mut_probs_for_high_conf_{muts_rel_WT}_from_{model_identifier}_ep{self.current_epoch}_batch_idx{batch_idx}.svg')
            plt.close()

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
        # all_tokens = self.tokenizer.get_vocab().keys()
        new_log_states = torch.zeros((len(sequence), 20), dtype=torch.bfloat16).to(self.device)
        
        with torch.no_grad():
            # Move the fixed model to the GPU only when needed
            self.model_being_updated.to(self.device)
            self.model_being_updated.eval()
            for mask_pos in range(len(sequence)):
                masked_sequence = self.mask_sequence(sequence, mask_pos) # Mask the current position
                inputs = self.tokenizer(masked_sequence, return_tensors="pt").to(self.device)
                logits = self.model_being_updated(**inputs).logits[:,:,4:24]
                log_probabilities = F.log_softmax(logits[0, mask_pos+1], dim=-1)
                new_log_states[mask_pos] = log_probabilities
            
            self.model_being_updated.train()

        return new_log_states

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






