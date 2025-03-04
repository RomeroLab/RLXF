#!/usr/bin/env python
# coding: utf-8

### Importing Modules
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchtext import vocab
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import random
import pickle
import csv

# Simulated Annealing Class
class SA_optimizer:
    def __init__(self, seq_fitness, WT, AA_options, num_mut, mut_rate, nsteps, cool_sched, non_gap_indices, start_temp, final_temp):
        
        """Initializes the SA_optimizer class with the following inputs:
        seq_fitness: a function that takes a sequence and returns its fitness
        WT: the wild-type sequence to be mutated
        AA_options: a list of possible amino acid substitutions for each position in the sequence
        num_mut: the number of mutations to make in each mutant sequence
        mut_rate: the rate at which mutations occur during simulated annealing
        nsteps: the number of steps in the cooling schedule for simulated annealing
        cool_sched: the cooling schedule to use for simulated annealing (either 'log' or 'lin')"""
        
        self.seq_fitness = seq_fitness
        self.WT = WT
        self.AA_options = AA_options
        self.num_mut = num_mut
        self.mut_rate = mut_rate
        self.nsteps = nsteps
        self.cool_sched = cool_sched
        self.start_temp = start_temp
        self.final_temp = final_temp
        self.close_sequences = []


    def optimize(self, non_gap_indices, dir_path, num_mut, version, wt_functional_threshold=None, start_mut=None):

        # If no starting mutation is provided, generate one randomly
        if start_mut is None:
            start_mut = generate_random_mut(self.WT, self.AA_options, self.num_mut).split(',')

        # Generate a list of all possible point mutants for the wild-type sequence
        all_mutants = generate_all_point_mutants(self.WT, non_gap_indices, self.AA_options)

        # Ensure that the cooling schedule is either logarithmic or linear
        assert ((self.cool_sched == 'log') or (self.cool_sched == 'lin')), 'cool_sched must be \'log\' or \'lin\''

        # Set the temperature schedule based on the cooling schedule
        if self.cool_sched == 'log':
            temp = np.logspace(self.start_temp, self.final_temp, self.nsteps)
        if self.cool_sched == 'lin':
            temp = np.linspace(1000, 1e-9, self.nsteps)

        # Initialize variables to track progress and store results
        print('Simulated Annealing Progress: New Simulation')
        seq = mut2seq(self.WT, start_mut)
        fit = self.seq_fitness(seq)
        current_seq = [start_mut, fit]  # Store the current sequence and its fitness
        self.best_seq = [start_mut, fit]  # Store the best sequence and its fitness found so far
        self.fitness_trajectory = [[fit, fit]]  # Store the trajectory of fitness values over time

        # for loop over decreasing temperatures
        for T in temp:
            # Create a mutant sequence based on the current sequence
            mutant = list(current_seq[0])

            # Choose the number of mutations to make to the current sequence
            n = np.random.poisson(self.mut_rate)
            n = min([self.num_mut - 1, max([1, n])])  # Bound the number of mutations within the range [1,num_mut-1]

            # Remove random mutations from the current sequence until it contains (num_mut-n) mutations
            while len(mutant) > (self.num_mut - n):
                mutant.pop(random.choice(range(len(mutant))))

            # Add back n random mutations to generate a new mutant sequence
            occupied = [m[1:-1] for m in mutant]  # Positions that already have a mutation
            mut_options = [m for m in all_mutants if m[1:-1] not in occupied]  # Mutations at unoccupied positions
            while len(mutant) < self.num_mut:
                mutant.append(random.choice(mut_options))
                occupied = [m[1:-1] for m in mutant]
                mut_options = [m for m in all_mutants if m[1:-1] not in occupied]

            # Sort mutations by position to clean up the format
            mutant = tuple([n[1] for n in sorted([(int(m[1:-1]), m) for m in mutant])])

            # Evaluate the fitness of the new mutant sequence
            fitness = self.seq_fitness(mut2seq(self.WT, mutant))

            # Determine if the current sequence is close to the maximum fitness value
            if wt_functional_threshold:
                if fitness > wt_functional_threshold: # This is WT predicted fitness
                    # Add the current sequence and its fitness to the list
                    self.close_sequences.append((mutant, fitness))

            # If the mutant sequence is better than the best sequence found so far, update the best sequence
            if fitness > self.best_seq[1]:
                self.best_seq = [mutant, fitness]

            # Simulated annealing acceptance criteria:
            # If the mutant sequence is better than the current sequence, move to the mutant sequence
            # If mutant is worse than current seq, move to mutant with some exponentially decreasing probability with delta_F
            delta_F = fitness - current_seq[1]  # calculate the difference in fitness between the mutant sequence and the current sequence

            ###############################################################################################
            # Printing first few accept probabilities. We want this to be 30-50%, preferrably 30-40%, but the overall SA curves appearance is more important
            accept_prob = np.exp(min([0, delta_F / (T)]))
            # print(f"Acceptance probability: {accept_prob}")
            ###############################################################################################
            
            if np.exp(min([0, delta_F / (T)])) > random.random():  # calculate the acceptance probability based on the temperature and delta_F
                current_seq = [mutant, fitness]  # if the mutant is accepted, set the current sequence to the mutant sequence

            # store the current fitness in the fitness trajectory
            self.fitness_trajectory.append([self.best_seq[1], current_seq[1]])

        # Define your directory path and file name
        file_path = os.path.join(dir_path, f"close_sequences_{num_mut}mut_v{version}.pickle")
        # Serialize the list of close sequences to a pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(self.close_sequences, f)
            
        print('Simulated Annealing Progress: Done')

        return self.best_seq  # returns [best_mut, best_fit]

    def plot_trajectory(self, savefig_name=None):
        """
        Plots the fitness trajectory of the simulated annealing optimization algorithm.
        Args:
            savefig_name (str): optional file name to save the plot as an image file.
        """
        # Plot the fitness trajectory of the best and current mutants
        plt.plot(np.array(self.fitness_trajectory)[:, 0],'x', markersize=8, markeredgecolor='black', color='black')
        plt.plot(np.array(self.fitness_trajectory)[:, 1],'orange')
        
        # Add labels and legend
        plt.xlabel('Step')
        plt.ylabel('Fitness')
        plt.legend(['Best mut found', 'Current mut'])
        
        # Show or save the plot
        if savefig_name is None:
            plt.show()
        else:
            plt.savefig(savefig_name)
        
        # Close the plot window
        plt.close()

# sequence-function handler, calculates conservative design score using ensemble
class seq_function_handler:
    def __init__(self, seq, models):
        self.seq = seq
        self.models = models

        AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
        aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
        aa2ind.set_default_index(20) # set unknown charcterers to gap
        self.aa2ind = aa2ind
        
    def seq2fitness(self, seq):
        labels = []

        # Score Sequence for all models
        with torch.no_grad():
            for model in self.models:
                model.eval()  # Set model to evaluation mode
                pred_Y = model.predict(seq).cpu().numpy().astype(float)  # Predict Label Scores
                labels.append(pred_Y)  # Append label scores for each enzyme from all models

        # Calculate lower confidence bound for all labels across all models
        score = np.quantile(labels, q=0.05, axis=0)[0]

        return score

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

# Functions to process data
def get_last_fitness_value(fitness_csv_path):
    # Read the last value from the "Fitness" column in the CSV file
    with open(fitness_csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        fitness_values = [float(row['Fitness']) for row in csv_reader]
    return fitness_values[-1] if fitness_values else None

def get_mutations(pickle_path):
    # Load mutations from the pickle file
    with open(pickle_path, 'rb') as pkl_file:
        mutations_data = pickle.load(pkl_file)
    return mutations_data

# Define function to apply mutations to a sequence
def apply_mutations(sequence, mutations):
    seq_list = list(sequence)
    for mutation in mutations:
        position = int(mutation[1:-1])
        new_aa = mutation[-1]
        seq_list[position] = new_aa
    return ''.join(seq_list)

def plot_heatmap_for_configuration(df, AAs, title, save_path, WT):
    
    # Unzip sequences to align positions
    alignment = tuple(zip(*df.Sequence))
    
    # Count AAs
    # AA_count = np.array([[p.count(a) for a in AAs] for p in alignment]) # raw AA counts
    AA_count = np.array([[sum(1 for seq_at_pos in alignment[pos] if seq_at_pos == a and WT[pos] != a) for a in AAs] for pos in range(len(WT))])

    Magma_r = plt.cm.magma_r(np.linspace(0, 1, 256))
    Magma_r[0] = [0, 0, 0, 0.03]  # Set the first entry (corresponding to 0 value) to white
    # Magma_r[0] = [0.9, 0.9, 0.9, 1]  # Set the first entry (corresponding to 0 value) to grey
    cmap = LinearSegmentedColormap.from_list("Modified_Magma_r", Magma_r, N=256)

    # Plot the heatmap
    plt.figure(figsize=(30,6))
    heatmap = sns.heatmap(AA_count.T, cmap=cmap, square=True, linewidths=0.003, linecolor='0.7')
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Count of Amino Acid Mutations', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    pos = cbar.ax.get_position()  # Get the original position
    cbar.ax.set_position([pos.x0 - 0.03, pos.y0, pos.width, pos.height])  # Shift the colorbar closer
    plt.yticks(np.arange(len(AAs)) + 0.5, AAs)
    plt.xlabel('Position', fontsize=18)
    plt.ylabel('Amino Acid', fontsize=18)
    plt.title(title)

    # Add black dots for WT sequence
    for pos, aa in enumerate(WT):
        if aa in AAs:  # Check if the AA is one of the considered AAs
            aa_index = AAs.index(aa)
            # Plot black dot; adjust dot size with 's' as needed
            plt.scatter(pos + 0.5, aa_index + 0.5, color='black', s=30)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='WT')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Save the plot
    plt.savefig(save_path)
    plt.show()
    plt.close()





