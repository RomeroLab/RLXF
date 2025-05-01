# Functions and models for simulated annealing

### Importing Modules
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pytorch_lightning as pl
from collections import OrderedDict
from torchtext import vocab
import matplotlib.pyplot as plt
import os
import random
import pickle
import csv

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap

# SeqFcnDataset is a data handling class.
# I convert amino acid sequences to torch tensors for model inputs
# I convert mean to torch tensors
class SeqFcnDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein sequence-function data"""

    def __init__(self, data_frame):
        self.data_df = data_frame

    def __getitem__(self, idx):
        sequence = torch.tensor(aa2ind(list(self.data_df.Sequence.iloc[idx]))) # Extract sequence at index idx
        labels = torch.tensor(self.data_df.iloc[idx, 8].tolist()).float() # Extract mean fitness score for sequence at index idx and convert to a list
        return sequence, labels

    def __len__(self):
        return len(self.data_df)

# ProtDataModule splits the data into three different datasets.
# Training data is used during model training
# Validation data is used to evaluate the model after epochs
# We want validation loss to be less than the training loss while both values decrease
# If validation loss > training loss, the model is likely overfit and cannot generalize to samples outside training set
# Testing data is used to evaluate the model after model training is complete
class ProtDataModule(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data splitting"""

    def __init__(self, data_frame, batch_size, splits_path=None):
        # Call the __init__ method of the parent class
        super().__init__()

        # Store the batch size
        self.batch_size = batch_size
        self.data_df = data_frame
        
        if splits_path is not None:
            train_indices, val_indices, test_indices = self.load_splits(splits_path)
            # print(test_indices)
            
            # Shuffle the indices to ensure that the data from each cluster is mixed. Do I want this?
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)
            
            # Store the indices for the training, validation, and test sets
            self.train_idx = train_indices
            self.val_idx = val_indices
            self.test_idx = test_indices
                
        else:
            
            # Initialize empty lists to hold the indices for the training, validation, and test sets
            train_indices = []
            val_indices = []
            test_indices = []
            
            gen = torch.Generator()
            gen.manual_seed(0)
            
            # Loop over each unique cluster in the DataFrame
            for cluster in self.data_df['Cluster'].unique():
                # Get the indices of the rows in the DataFrame that belong to the current cluster
                cluster_indices = self.data_df[self.data_df['Cluster'] == cluster].index.tolist()
                # Define the fractions of the data that should go to the training, validation, and test sets
                train_val_test_split = [0.8, 0.1, 0.1]
                # Calculate the number of samples that should go to each set based on the fractions defined above
                n_train_val_test = np.round(np.array(train_val_test_split)*len(cluster_indices)).astype(int)
                # If the sum of the calculated numbers of samples is less than the total number of samples in the cluster,
                # increment the number of training samples by 1
                if sum(n_train_val_test)<len(cluster_indices): n_train_val_test[0] += 1 # necessary when round is off by 1
                # If the sum of the calculated numbers of samples is more than the total number of samples in the cluster,
                # decrement the number of training samples by 1
                if sum(n_train_val_test)>len(cluster_indices): n_train_val_test[0] -= 1 
                # Split the indices of the current cluster into training, validation, and test sets
                train, val, test = data_utils.random_split(cluster_indices,n_train_val_test,generator=gen)
                # Add the indices for the current cluster to the overall lists of indices
                train_indices.extend(train.indices)
                val_indices.extend(val.indices)
                test_indices.extend(test.indices)
            
            # Shuffle the indices to ensure that the data from each cluster is mixed
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)
            
            # Store the indices for the training, validation, and test sets
            self.train_idx = train_indices
            self.val_idx = val_indices
            self.test_idx = test_indices
            # print(test_indices)

    # Prepare_data is called from a single GPU. Do not use it to assign state (self.x = y). Use this method to do
    # things that might write to disk or that need to be done only from a single process in distributed settings.
    def prepare_data(self):
        pass

    # Assigns train, validation and test datasets for use in dataloaders.
    def setup(self, stage=None):
              
        # Assign train/validation datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_data_frame = self.data_df.iloc[list(self.train_idx)]
            self.train_ds = SeqFcnDataset(train_data_frame)
            val_data_frame = self.data_df.iloc[list(self.val_idx)]
            self.val_ds = SeqFcnDataset(val_data_frame)
                    
        # Assigns test dataset for use in dataloader
        if stage == 'test' or stage is None:
            test_data_frame = self.data_df.iloc[list(self.test_idx)]
            self.test_ds = SeqFcnDataset(test_data_frame)
            
    #The DataLoader object is created using the train_ds/val_ds/test_ds objects with the batch size set during
    # initialization of the class and shuffle=True.
    def train_dataloader(self):
        return data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return data_utils.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)
    def test_dataloader(self):
        return data_utils.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True)
    
    def save_splits(self, path):
        """Save the data splits to a file at the given path"""
        with open(path, 'wb') as f:
            pickle.dump((self.train_idx, self.val_idx, self.test_idx), f)

    def load_splits(self, path):
        """Load the data splits from a file at the given path"""
        with open(path, 'rb') as f:
            self.train_idx, self.val_idx, self.test_idx = pickle.load(f)
            
            train_indices = self.train_idx
            val_indices = self.val_idx
            test_indices = self.test_idx
            
        return train_indices, val_indices, test_indices

# PTLModule is the actual neural network. Model architecture can be altered here.
class CNN(pl.LightningModule):
    """PyTorch Lightning Module that defines model and training"""
      
    # define network
    def __init__(self, slen, ks, learning_rate, epochs, batch_size, factor_2=2, factor_3=2, dim_4=400, dim_5=50):
        super().__init__()
        
        # Creates an embedding layer in PyTorch and initializes it with the pretrained weights stored in aaindex
        self.embed = nn.Embedding(len(AAs), 16) # maps integer indices (a.a.'s)' to 16-dimensional vectors
        # self.embed = nn.Embedding.from_pretrained(torch.eye(len(AAs)), freeze=True) for one hot encoding
        self.slen = slen # CreiLOV sequence length
        self.ndim = self.embed.embedding_dim # dimensions of AA embedding
        self.ks = ks # kernel size describes how many positions the neural network sees in each convolution
        
        conv_out_dim = factor_3*self.ndim # determines output size of last conv layer
        self.nparam = slen*conv_out_dim # desired (flattened) output size for last convolutional layer
        # self.dropout1 = nn.Dropout(p=0)
        self.dropout2 = nn.Dropout(p=0.2)
        pad = int((self.ks - 1)/2)

        # Convolutional layers block
        self.enc_conv_1 = torch.nn.Conv1d(in_channels= self.ndim, out_channels=factor_2*self.ndim, kernel_size=ks, padding=pad)
        self.enc_conv_2 = torch.nn.Conv1d(in_channels= factor_2*self.ndim, out_channels=conv_out_dim, kernel_size=ks, padding=pad) 
        
        # Fully connected layers block
        self.linear1 = nn.Linear(self.nparam, dim_4)
        self.linear2 = nn.Linear(dim_4,dim_5)
        self.linear3 = nn.Linear(dim_5,1)
        # print(self.nparam)
        
        # learning rate
        self.learning_rate = learning_rate
        self.save_hyperparameters('learning_rate', 'batch_size', 'ks', 'epochs', 'slen') # log hyperparameters to file
             
    def forward(self, x):
        
        x = self.embed(x) # PyTorch will learn embedding with the specified dimensions
        
        # Convolutional layers block
        x = x.permute(0,2,1) # swap length and channel dims        
        x = self.enc_conv_1(x)   
        x = F.leaky_relu(x) # this is an activation fucniton and is non-linear component of a neural network
        # x = self.dropout1(x)
        
        x = self.enc_conv_2(x)
        x = F.leaky_relu(x) # this is an activation fucniton and is non-linear component of a neural network
        # x = self.dropout1(x)
        
        # Fully connected layers block
        x = x.view(-1,self.nparam) # flatten (input for linear/FC layers must be 1D)
        x = self.linear1(x)
        x = self.dropout2(x)  # Add dropout after the first fully connected layer
        x = F.relu(x) # this is an activation fucniton and is non-linear component of a neural network
        
        x = x.view(-1,dim_4)
        x = self.linear2(x)
        x = self.dropout2(x)  # Add dropout after the first fully connected layer
        x = F.relu(x) # this is an activation fucniton and is non-linear component of a neural network

        x = x.view(-1,dim_5)
        x = self.linear3(x)
        
        return x
      
    def training_step(self, batch, batch_idx):
        sequence,scores = batch
        scores = scores.unsqueeze(1)  # Add an extra dimension to the target tensor
        output = self(sequence)
        loss = nn.MSELoss()(output, scores) # Calculate MSE
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step = False, on_epoch=True) # reports MSE loss to model
        return loss

    def validation_step(self, batch, batch_idx):
        sequence,scores = batch
        scores = scores.unsqueeze(1)  # Add an extra dimension to the target tensor
        output = self(sequence)
        loss = nn.MSELoss()(output, scores) # Calculate MSE
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step = False, on_epoch=True) # reports MSE loss to model
        return loss

    def test_step(self, batch):
        sequence,scores = batch
        output = self(sequence)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.001) # Weight Decay to penalize too large of weights
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) # No weight decay
        return optimizer
    
    def predict(self, sequence):
        # ind = torch.tensor(aa2ind(list(sequence))) # Convert the amino acid sequence to a tensor of indices
        x = sequence.view(1,-1) # Add a batch dimension to the tensor (put here instead of forward function)
        pred = self(x) # Apply the model to the tensor to get the prediction
        return pred # .detach().numpy() # Detach the prediction from the computation graph and convert it to a NumPy array

    def predict_2(self, sequence):
        # ind = torch.tensor(aa2ind(list(sequence))) # Convert the amino acid sequence to a tensor of indices
        # x = sequence.view(1,-1) # Add a batch dimension to the tensor (put here instead of forward function)
        pred = self(sequence) # Apply the model to the tensor to get the prediction
        return pred # .detach().numpy() # Detach the prediction from the computation graph and convert it to a NumPy array


def get_non_gap_indices(seq):
    """Get the indices of non-gap positions in the input MSA sequence"""
    return [i for i, aa in enumerate(seq) if aa != "-"]

def generate_all_point_mutants(seq, non_gap_indices, AA_options):
    """Generate all possible single point mutants of a sequence at non-gap positions
    Arguments:
    seq: starting seq - the original sequence to mutate
    non_gap_indices: list of indices corresponding to non-gap positions in the sequence
    AA_options: list of amino acid options at each position, if none defaults to all 20 AAs (default None)
    """
    all_mutants = []  # Initialize an empty list to store all the possible mutants
    for pos in non_gap_indices:  # Loop through each non-gap position in the input sequence
        for aa in AA_options[pos]:  # Loop through each amino acid at that position
            if seq[pos] != aa:  # If the current amino acid is not the same as the original one at that position
                mut = seq[pos] + str(pos) + aa  # Create a string to represent the mutation (e.g. G12A)
                all_mutants.append(mut)  # Add the mutation to the list of all mutants
                
    return all_mutants  # Return the list of all mutants

def mut2seq(seq, mutations):
    """Create mutations in form of A94T to seq
    Arguments:
    seq: starting seq - the original sequence to mutate
    mutations: list of mutations in form of ["A94T", "H99R"] or "A94T,H99R"
    """
    mutant_seq = seq  # Initialize the mutant sequence as the original sequence

    if type(mutations) is str:  # If mutations is a string, split it into a list of mutations
        mutations = mutations.split(',')
    for mut in mutations:  # Loop through each mutation in the list
        pos = int(mut[1:-1])  # Get the position of the mutation
        newAA = mut[-1]  # Get the new amino acid for the mutation
        if mut[0] != seq[pos]:  # If the wild-type amino acid at the mutation position does not match the original sequence, print a warning
            print('Warning: WT residue in mutation %s does not match WT sequence' % mut)
        mutant_seq = mutant_seq[:pos] + newAA + mutant_seq[pos + 1:]  # Apply the mutation to the mutant sequence

    return mutant_seq  # Return the mutant sequence

def find_top_n_mutations(VAE_fitness, all_mutants, WT, n=10):
    """
    Find the top n mutations with the highest fitness score from a list of all possible single point mutations.
    Arguments:
        VAE_fitness: function to calculate fitness score for a given sequence
        all_mutants: list of all possible single point mutants for the starting sequence
        WT: wild-type starting sequence
        n: number of top mutations to return (default 10)
    Returns:
        topn: list of n top mutations sorted by fitness score in descending order with the format 'A8C'
    """
    # evaluate fitness of all single mutants from WT
    single_mut_fitness = []
    for mut in all_mutants:
        pos = int(mut[1:-1])
        seq = WT[:pos] + mut[-1] + WT[pos+1:]
        fit = VAE_fitness(seq)
        single_mut_fitness.append((mut, fit))
    
    # find the best mutation per position
    best_mut_per_position = []
    for pos in range(len(WT)):
        # select the mutation with the highest fitness score for the current position
        position_mutants = [m for m in single_mut_fitness if int(m[0][1:-1]) == pos]
        if not position_mutants:
            continue
        best_mut_per_position.append(max(position_mutants, key=lambda x: x[1]))
    
    # take the top n mutations
    sorted_by_fitness = sorted(best_mut_per_position, key=lambda x: x[1], reverse=True)
    topn = [m[0] for m in sorted_by_fitness[:n]]

    # sort the top n mutations by position and format them as 'A8C'
    # topn_formatted = [WT[int(m[1:-1])] + str(int(m[1:-1])+1) + m[-1] for m in topn]
    
    # take the top n
    topn = tuple([n[1] for n in sorted([(int(m[1:-1]), m) for m in topn])])  # sort by position

    return topn_formatted

### This can mutate gaps that we do not want to mutate
def generate_random_mut(WT, AA_options, num_mut):
    # Create a list of all possible mutations for each position in the wild-type sequence
    AA_mut_options = []
    for WT_AA, AA_options_pos in zip(WT, AA_options):
        if WT_AA in AA_options_pos: # If the wild-type amino acid is an option at this position
            options = list(AA_options_pos).copy() # Create a copy of the list of possible AAs
            options.remove(WT_AA) # Remove the wild-type AA from the list of possible AAs
            AA_mut_options.append(options) # Add the list of possible mutations to the list of AA_mut_options
    
    # Create a list of random mutations
    mutations = []
    for n in range(num_mut):
        # Calculate the probability of each position mutating
        num_mut_pos = sum([len(row) for row in AA_mut_options]) # Count the number of positions that can mutate
        prob_each_pos = [len(row) / num_mut_pos for row in AA_mut_options] # Calculate the probability of each position mutating
        
        # Choose a position to mutate based on its probability
        rand_num = random.random() # Choose a random number between 0 and 1
        for i, prob_pos in enumerate(prob_each_pos):
            rand_num -= prob_pos
            if rand_num <= 0: # If the random number is less than or equal to the probability of this position mutating, choose this position
                # Choose a random mutation for this position
                mutations.append(WT[i] + str(i) + random.choice(AA_mut_options[i]))
                AA_mut_options.pop(i) # Remove this position from the list of AA_mut_options
                AA_mut_options.insert(i, []) # Add an empty list to the list of AA_mut_options to indicate that this position has already mutated
                break
    # Return the list of random mutations as a string
    return ','.join(mutations)


def generate_random_mut_non_gap_indices(WT, AA_options, num_mut, non_gap_indices, mutating_window_size):
    # Create a list of all possible mutations for each position in the wild-type sequence
    AA_mut_options = [[] for _ in range(len(WT))]  # Initialize a list with the length of WT

    if num_mut > mutating_window_size:
        raise ValueError('Number of mutations must be less than the length of WT being mutated (mutating_window_size)')
    
    # Fill only non-gap indices with mutation options
    for idx in non_gap_indices:
        WT_AA = WT[idx]
        AA_options_pos = AA_options[idx]
        if WT_AA in AA_options_pos:  # If the wild-type amino acid is an option at this position
            options = list(AA_options_pos).copy()  # Create a copy of the list of possible AAs
            options.remove(WT_AA)  # Remove the wild-type AA from the list of possible AAs
            AA_mut_options[idx] = options  # Set the list of possible mutations at the correct index
    
    # Create a list of random mutations
    mutations = []
    for _ in range(num_mut):
        # Calculate the probability of each position mutating
        num_mut_pos = sum([len(x) for x in AA_mut_options if x])  # Count the number of positions that can mutate
        if num_mut_pos == 0:
            break  # No more mutations possible
        
        prob_each_pos = [(len(x) / num_mut_pos if x else 0) for x in AA_mut_options]  # Probability for each position
        
        # Choose a position to mutate based on its probability
        rand_num = random.random()  # Choose a random number between 0 and 1
        cumulative_prob = 0
        
        for i, prob_pos in enumerate(prob_each_pos):
            cumulative_prob += prob_pos
            if rand_num <= cumulative_prob and AA_mut_options[i]:  # If the random number is less than or equal to the cumulative probability of this position mutating
                # Choose a random mutation for this position
                mutations.append(WT[i] + str(i) + random.choice(AA_mut_options[i]))
                AA_mut_options[i] = []  # Set this position to an empty list to indicate it cannot mutate again
                break

    # Return the list of random mutations as a string
    return ','.join(mutations)







