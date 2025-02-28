#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from functions import (load_reward_model, identify_mutations_and_count, generate_df, generate_and_evaluate_mutants, mutate_sequences_after_training, mutate_sequences_after_training_esm2_max_sampling, get_sft_version_file)
from MLP import MLP

from SA_utils import get_non_gap_indices, generate_all_point_mutants, mut2seq, find_top_n_mutations, generate_random_mut

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap

######################################### hyperparameter that can be altered #########################################
# Altering hyperparameters can sometimes change model performance or training time
learning_rate = 1e-6 # important to optimize this
batch_size = 128 # typically powers of 2: 32, 64, 128, 256, ...
epochs = 2000 # rounds of training
slen = len(WT) # length of protein
num_models = 100 # number of models in ensemble
patience = 400
######################################### hyperparameter that can be altered #########################################

models = []
for i in range(100):
    model_name = f"best_model_v{i}.ckpt"
    # checkpoint_path = f"/Users/nathanielblalock/Desktop/RLXF_PPO_from_pretrained_ESM2_GPU/MLP_Reward_Models/{model_name}"
    checkpoint_path = f"./MLP_Reward_Models/{model_name}"
    reward_model = load_reward_model(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    models.append(reward_model)

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


    def optimize(self, start_mut=None):

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
#         if self.cool_sched == 'lin':
#             temp = np.linspace(1000, 1e-9, self.nsteps)

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
            if fitness > 4.1498: # This is WT predicted fitness
                # Add the current sequence and its fitness to the list
                self.close_sequences.append((mutant, fitness))


            # If the mutant sequence is better than the best sequence found so far, update the best sequence
            if fitness > self.best_seq[1]:
                self.best_seq = [mutant, fitness]

            # Simulated annealing acceptance criteria:
            # If the mutant sequence is better than the current sequence, move to the mutant sequence
            # If mutant is worse than current seq, move to mutant with some exponentially decreasing probability with delta_F
            delta_F = fitness - current_seq[1]  # calculate the difference in fitness between the mutant sequence and the current sequence

            # ###############################################################################################
            # # Printing first few accept probabilities. We want this to be 30-50%, preferrably 30-40%, but the overall SA curves appearance is more important
            # if T > -1.5:
            #     accept_prob = np.exp(min([0, delta_F / (T)]))
            #     print(f"Acceptance probability: {accept_prob}")
            # ###############################################################################################
            
            if np.exp(min([0, delta_F / (T)])) > random.random():  # calculate the acceptance probability based on the temperature and delta_F
                current_seq = [mutant, fitness]  # if the mutant is accepted, set the current sequence to the mutant sequence

            # store the current fitness in the fitness trajectory
            self.fitness_trajectory.append([self.best_seq[1], current_seq[1]])

        # Define your directory path and file name
        file_path = os.path.join(dir_path, f"close_sequences_{num_mut}mut_start_pos{start_position}_v{i}.pickle")
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
    
# MTFCNN_fitness for C6 experiments
class fitness_handler:
    def __init__(self, seq, models):
        self.seq = seq
        self.models = models
        
    def seq2fitness(self, seq):
        labels = []

        # Convert the sequence to tensor representation for model prediction
        sequence_tensor = torch.tensor(aa2ind(list(seq)))

        # Score Sequence for all models
        with torch.no_grad():
            for model in self.models:
                model.eval()  # Set model to evaluation mode
                pred_Y = model.predict(sequence_tensor.unsqueeze(0)).cpu().numpy().astype(float)  # Predict Label Scores
                labels.append(pred_Y)  # Append label scores for each enzyme from all models

        # Calculate lower confidence bound for all labels across all models
        score = np.quantile(labels, q=0.05, axis=0)[0]

        return score

def mutating_window_rational_approach(non_gap_indices, WT_no_gaps, residues_within_4A, start_pos, mutating_window_size, aa_weights=None): ###### Edit Here ######
    """
    This function applies a sliding window approach to update the amino acid (AA)
    options for a sequence, starting from a given position. It sets the AA options
    to the corresponding AAs from `WT_no_gaps` within the specified window.
    
    Parameters:
    - WT_no_gaps (str): WT sequence without gaps.
    - start_pos (int): The starting position for the sliding window.
    - mutating_window (int): The length of the window where mutations are allowed.
    
    Returns:
    - list: A list of tuples, where each tuple contains the AA options for each position.
      Positions within the cloning window are set to the specific AA from `WT_no_gaps`,
      while other positions remain unchanged (all possible AAs).
    """
    AAs_options = 'ACDEFGHIKLMNPQRSTVWY'
    AA_options = [tuple([AA for AA in AAs_options]) for i in range(len(WT))]
    AA_options[non_gap_indices[0]] = WT_no_gaps[0] # Keep start codon

    window_end = start_pos + mutating_window_size
    for i in range(len(WT_no_gaps)):
        
        # Choose amino acids to keep frozen for cloning
        if i <= start_pos or i > window_end:
            AA_options[non_gap_indices[i]] = (WT_no_gaps[i])

        # Get amino acids that are either the same or have a smaller molecular weight
        elif i+1 in residues_within_4A:
            wt_aa_weight = aa_weights[WT_no_gaps[i]]
            smaller_or_same_AAs = [aa for aa in AAs_options if aa_weights[aa] <= wt_aa_weight]
            AA_options[non_gap_indices[i]] = smaller_or_same_AAs
    
    return AA_options

# # Load the lookup table from the file
# with open('amino_acid_weight_lookup.pkl', 'rb') as f:
#     aa_weights = pickle.load(f)

# WT = '-------MAVKHLIVLKFKDEITEAQKEEFFKTYVNLVN--IIPAMKDVYW----GK-DVTQKNKEEGYTHIVEVTFESVETIQDYII-HPAHVGFGDVYRSFWEKLLIFDY-----TPRK-------'
# WT_no_gaps = 'MAVKHLIVLKFKDEITEAQKEEFFKTYVNLVNIIPAMKDVYWGKDVTQKNKEEGYTHIVEVTFESVETIQDYIIHPAHVGFGDVYRSFWEKLLIFDYTPRK' # PKC1.0 with no gaps
# non_gap_indices = get_non_gap_indices(WT)
# residues_within_4A = [5, 7, 9, 23, 24, 27, 28, 30, 40, 49, 59, 72, 73, 78, 81, 82, 89, 92, 94, 96] # Define residues within 4 angstroms of docked OA and in active site

# # Get the updated AA options
# AA_options = mutating_window_rational_approach(non_gap_indices, WT_no_gaps, residues_within_4A, aa_weights, start_pos=6, mutating_window_size=82)
# for index in non_gap_indices:
#     print(AA_options[index])


# In[13]:


################################################ Simulated Annealing Parameters ################################################
non_gap_indices = get_non_gap_indices(WT)
WT_no_gaps = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
fixed_window_size = 0
mutating_window_size = len(WT_no_gaps)-fixed_window_size # size of window where amino acids can be altered
# with open('amino_acid_weight_lookup.pkl', 'rb') as f:
#     aa_weights = pickle.load(f) # molecular weights for aas
residues_within_4A = [] # Define residues within 4 angstroms of docked OA and in active site
start_position = 0 # position window where amino acids can be altered begins
nsteps = 50000
num_trials = 100
num_mut = 5
mut_rate = 2
start_temp = -1.6
final_temp = -3.1
type = '5mut_CreiLOV' # substrate
seed = 1
random.seed(seed) # Set random seeds for reproducibility
np.random.seed(seed) # Set random seeds for reproducibility
################################################ Simulated Annealing Parameters ################################################

# create SA_trials folder if it doesn't exist
if not os.path.exists('SA_trials'):
    os.makedirs('SA_trials')

dir_path = f'SA_trials/{type}_max_{num_mut}mut_{nsteps}steps'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


# Saving parameters
params_str = f"""################################################
Simulated Annealing Parameters
################################################
non_gap_indices = {non_gap_indices}
WT_no_gaps = '{WT_no_gaps}'
fixed_window_size = {fixed_window_size}
mutating_window_size = {mutating_window_size}
start_position = {start_position}
nsteps = {nsteps}
num_trials = {num_trials}
num_mut = {num_mut}
mut_rate = {mut_rate}
start_temp = {start_temp}
final_temp = {final_temp}
type = '{type}'
seed = {seed}
################################################
Simulated Annealing Parameters
################################################
"""

# Path for the parameters text file
file_path = os.path.join(dir_path, "parameters.txt")

# Write parameters to the file
with open(file_path, "w") as file:
    file.write(params_str)

print(f"Parameters saved to {file_path}")


# Determine AA_options given sliding window
AA_options = mutating_window_rational_approach(non_gap_indices, WT_no_gaps, residues_within_4A, start_position, mutating_window_size, aa_weights=None)
for i in range(num_trials):
    # Set the file names with version numbers
    best_mutant_file = f"{dir_path}/best_{type}_{num_mut}mut_start_pos{start_position}_v{i}.pickle"
    trajectory_file = f"{dir_path}/traj_{type}_{num_mut}mut_start_pos{start_position}_v{i}.png"
    csv_filename = f"{dir_path}/fitness_trajectory_{type}_{num_mut}mut_start_pos{start_position}_v{i}.csv"

    # Create an instance of seq_fitness class for the current mutant
    seq_fitness = fitness_handler(WT, models)

    # Create an instance of SA_optimizer class for the current mutant
    sa_optimizer = SA_optimizer(seq_fitness.seq2fitness,
                                 WT,
                                 AA_options,
                                 num_mut=num_mut,
                                 mut_rate=mut_rate,
                                 nsteps=nsteps,
                                 cool_sched='log',
                                 non_gap_indices=non_gap_indices,
                                 start_temp=start_temp,
                                 final_temp=final_temp)

    # Optimize the mutant and store the best mutant and its fitness in a pickle file
    best_mut, fitness = sa_optimizer.optimize()
    with open(best_mutant_file, 'wb') as f:
        pickle.dump((best_mut, fitness), f)

    # Save fitness trajectory in a CSV file
    with open(csv_filename, mode='w') as csv_file:
        fieldnames = ['Step', 'Fitness']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for step, (_, fitness) in enumerate(sa_optimizer.fitness_trajectory):
            writer.writerow({'Step': step, 'Fitness': float(fitness)})

    # Save Plotted Trajectory
    sa_optimizer.plot_trajectory(savefig_name=trajectory_file)


