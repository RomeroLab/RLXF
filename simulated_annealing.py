#!/usr/bin/env python
# coding: utf-8

### Importing Modules
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchtext import vocab
import os
import random
import pickle
import csv

# import helper scripts
from MLP import (SeqFcnDataset, ProtDataModule, MLP)
from simulated_annealing_utils import (seq_function_handler, SA_optimizer, get_non_gap_indices, get_last_fitness_value, get_mutations, apply_mutations, plot_heatmap_for_configuration)

# Parameters to update
AAs = 'ACDEFGHIKLMNPQRSTVWY-'
WT = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK' # parent sequence
wt_functional_threshold = None # find the predicted function of parent sequence with ensemble of reward models if you want to find more sequences
num_models = 10 # number of models in ensemble
num_trials = 100 # number of simulated annealing trials
num_mut = 5 # number of mutations ideally for the number of mutations in the designs you want to characterize after functional alignment
nsteps = 50000 # number of steps during simulated annealing
mut_rate = 2 # number of mutations per step
start_temp = -1.6 # initial temperature, this may need to be optimized for your functional score - ideally have 30-60% of accepting random mutations at first
final_temp = -3.1 # final temperature, this may need to be optimized for your functional score
seed = 1

# use seed for reproducibility
random.seed(seed)
np.random.seed(seed)

# define parameters
non_gap_indices = get_non_gap_indices(WT)

# load ensemble of reward models
models = []
for i in range(num_models):
    model_name = f"reward_model_v{i}.ckpt"
    checkpoint_path = f"./reward_models/{model_name}"
    reward_model = MLP.load_from_checkpoint(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    models.append(reward_model)

# create simulated_annealing_results folder it doesn't exist
if not os.path.exists('simulated_annealing_results'):
    os.makedirs('simulated_annealing_results')

# create folder named with important simulated annealing parameters
dir_path = f'simulated_annealing_results/{num_mut}mut_{nsteps}steps'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Save parameters to text file
params_str = f"""
################################################
Simulated Annealing Parameters
################################################
WT = '{WT}'
non_gap_indices = {non_gap_indices}
nsteps = {nsteps}
num_trials = {num_trials}
num_mut = {num_mut}
mut_rate = {mut_rate}
start_temp = {start_temp}
final_temp = {final_temp}
seed = {seed}
################################################
Simulated Annealing Parameters
################################################
"""

# Write parameters to the file
file_path = os.path.join(dir_path, "parameters.txt")
with open(file_path, "w") as file:
    file.write(params_str)
print(f"Parameters saved to {file_path}")

# Determine AA_options
AA_options = [tuple([AA for AA in AAs]) for i in range(len(WT))]
for i in range(num_trials):
    # Set the file names with version numbers
    best_mutant_file = f"{dir_path}/best_{num_mut}mut_v{i}.pickle"
    trajectory_file = f"{dir_path}/traj_{num_mut}mut_v{i}.png"
    csv_filename = f"{dir_path}/fitness_trajectory_{num_mut}mut_v{i}.csv"

    # Create an instance of seq_fitness class for the current mutant
    seq_fitness = seq_function_handler(WT, models)

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
    best_mut, fitness = sa_optimizer.optimize(non_gap_indices, dir_path, num_mut, i, wt_functional_threshold)
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

# Define filenames to load
csv_filename_template = f'fitness_trajectory_{num_mut}mut_v{{}}.csv'
pickle_filename_template = f'best_{num_mut}mut_v{{}}.pickle'
files_info = {
    f"Trial {i}": {
        'fitness_csv': os.path.join(dir_path, csv_filename_template.format(i)),
        'mutations_pickle': os.path.join(dir_path, pickle_filename_template.format(i))
    }
    for i in range(num_trials)
}

# Iterate over both files to find the max scores, mutations, and mutated sequences
results = []
for key, file_info in files_info.items():
    last_fitness_value = get_last_fitness_value(file_info['fitness_csv'])
    mutations_data = get_mutations(file_info['mutations_pickle'])
    
    if last_fitness_value is not None and mutations_data is not None:
        mutations, _ = mutations_data  # Assuming mutations_data contains (mutations, array([fitness_value]))
        mutated_sequence = apply_mutations(WT, mutations)
        
        results.append({
            'Trial': key,
            'Fitness': last_fitness_value,
            'Mutations': mutations,
            'Sequence': mutated_sequence
        })

# Convert results into a DataFrame and display/save
results_df = pd.DataFrame(results)
results_df.to_pickle(f'all_optimized_designs_from_simulated_annealing.pkl')
results_df.head()

# Filter to unique sequences
unique_sequences_df = results_df.drop_duplicates(subset=['Sequence'])
unique_sequences_df.to_pickle('unique_optimized_designs_from_simulated_annealing.pkl')
unique_sequences_df.head()

# Generate heatmap for amino acids vs. sequence position
plot_heatmap_for_configuration(unique_sequences_df, AAs,
                               'Distribution of Amino Acid Mutations for Unique Designs from Simulated Annealing',
                               './simulated_annealing_results/SA_mutation_distribution.png', WT)
plot_heatmap_for_configuration(unique_sequences_df, AAs,
                               'Distribution of Amino Acid Mutations for Unique Designs from Simulated Annealing',
                               './simulated_annealing_results/SA_mutation_distribution.svg', WT)





