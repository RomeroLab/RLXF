
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
from MLP import MLP
from transformers import AutoModelForMaskedLM, AutoTokenizer
from matplotlib.colors import LinearSegmentedColormap

# Training on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap


# add the number of mutations
def count_mutations(mutation_str):
        """Count the number of mutations in the mutation string."""
        return len(mutation_str.split(','))

# ESM2 score
def log_likelihood(sequences, device, model, tokenizer, no_grad=True):
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

    progress = 0
    for sequence in sequences:
        if no_grad is True:
            with torch.no_grad():  # Disable gradients for inference
                inputs = tokenizer.encode(sequence, return_tensors='pt').to(device)
                outputs = model(inputs, labels=inputs)
                loss, logits = outputs[:2]
                all_loss.append(loss.unsqueeze(0))
                progress += 1
                if progress % 100 == 0:
                    print('Sequences processed without gradients:', progress)

        else:
            inputs = tokenizer.encode(sequence, return_tensors='pt').to(device)
            outputs = model(inputs, labels=inputs)
            loss, logits = outputs[:2]
            all_loss.append(loss.unsqueeze(0))
            progress += 1
            if progress % 100 == 0:
                print('Sequences processed with gradients:', progress)
        
    all_loss = torch.cat(all_loss)

    return all_loss

# Loading reward model
def load_reward_model(checkpoint_path):
    """
    Load a reward model from a checkpoint.
    Args:
        checkpoint_path (str): Path to the saved checkpoint file.
    Returns:
        Reward Model (nn.Module)
    """
    reward_model = MLP.load_from_checkpoint(checkpoint_path)
    return reward_model

def hamming_distance(s1, s2):
    """Calculates the Hamming distance between two sequences"""
    return sum(1 for x, y in zip(s1, s2) if x != y and x != '-' and y != '-') # Quantify sequence similarity

# Generate DataFrames for RL, SFT, and Fixed sequences
def generate_df(sequences, scores):
    mutations_list = []
    num_mutations_list = []
    
    for seq in sequences:
        mutation_str, num_mutations = identify_mutations_and_count(WT, seq)
        mutations_list.append(mutation_str)
        num_mutations_list.append(num_mutations)
    
    return pd.DataFrame({
        'Sequence': sequences,
        'Score': scores,
        'Mutations': mutations_list,
        'Number of Mutations': num_mutations_list})

def mask_sequence(sequence, mask_pos):
    """Mask a single position in the sequence and return the masked sequence."""
    masked_sequence = list(sequence)
    masked_sequence[mask_pos] = '<mask>'  # Adjust for the <cls> token shift
    masked_seq_str = ''.join(masked_sequence)
    return masked_seq_str

def get_logits_for_all_positions(model, WT, tokenizer, model_identifier=None):
    """Generate logits for all positions in the WT sequence by masking one position at a time."""
    sequence_length = len(WT)
    all_logits = []

    with torch.no_grad():
        for pos in range(0, sequence_length):  # Positions excluding <cls> and <eos>
            masked_seq = mask_sequence(WT, pos)
            inputs = tokenizer(masked_seq, return_tensors='pt')
            
            # Get logits from the model
            outputs = model(**inputs)
            logits = outputs.logits
        
            # Extract logits for the masked position
            masked_logits = logits[0, pos+1]  # Shape: [vocab_size]
            all_logits.append(masked_logits)
    
    return torch.stack(all_logits)  # Shape: [sequence_length, vocab_size]

def generate_heatmap(WT, probabilities, model_identifier, sequence, filepath, ep, version, tokenizer):
    """Generate and save a heatmap based on the predicted probabilities."""
    
    # Generate mutations relative to WT
    muts_rel_WT = get_mutations(sequence, WT)

    # Set up tokens and color map
    all_tokens = tokenizer.get_vocab().keys()
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
    plt.yticks(np.arange(len(all_tokens)) + 0.5, all_tokens, fontsize=8, rotation=0)
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
    plt.savefig(f'{filepath}/version_{version}/single_mutant_probability_heatmap_for_{muts_rel_WT}_from_{model_identifier}_ep{ep}.png')
    plt.savefig(f'{filepath}/version_{version}/single_mutant_probability_heatmap_for_{muts_rel_WT}_from_{model_identifier}_ep{ep}.svg')
    plt.close()

def get_mutations(seq, wt):
    # Find mutations and their positions
    mutations = [f"{wt_res}{i}{seq_res}" for i, (wt_res, seq_res) in enumerate(zip(wt, seq), 1) if seq_res != wt_res]
    # Return the WT or mutation string
    if not mutations:
        return "WT"  # or simply return wt
    else:
        return "_".join(mutations)

def generate_high_confidence_mutant_sequences(WT, probabilities, tokenizer, model_identifier, num_designs=5, threshold=0.9):
    """
    Generate mutant sequences by identifying high-confidence mutations and mutating WT based on p-sampling.
    
    Args:
    - WT (str): Wild-type sequence.
    - probabilities (torch.Tensor): Probability tensor of shape [sequence_length, vocab_size].
    - num_designs (int): Number of mutated sequences to generate.
    - threshold (float): Probability threshold for mutation detection.

    Returns:
    - mutated_sequences (list): List of duplicated mutated sequences based on high-confidence mutations.
    """
    all_tokens = list(tokenizer.get_vocab().keys())  # Get the list of all tokens for reference
    high_conf_mutations = {}

    # Identify high-confidence mutations
    for pos, wt_res in enumerate(WT):
        position_probs = probabilities[pos]  # Get the probability distribution for this position
        wt_token_id = tokenizer.convert_tokens_to_ids(wt_res)

        # Find tokens with probability > threshold, excluding WT residue
        high_conf_tokens = [(all_tokens[token_id], prob.item()) for token_id, prob in enumerate(position_probs)
                            if token_id != wt_token_id and prob > threshold]

        if high_conf_tokens:
            high_conf_mutations[pos + 1] = high_conf_tokens  # Store position as 1-indexed

    # Generate a single mutated sequence based on high-confidence mutations
    mutated_seq = list(WT)  # Start with the WT sequence as a list for mutability

    for pos, mutations in high_conf_mutations.items():
        # Sample the token with the maximum probability among high-confidence mutations
        max_token, max_prob = max(mutations, key=lambda x: x[1])
        mutated_seq[pos - 1] = max_token  # Apply mutation (pos-1 to convert to 0-indexed)

    # Convert mutated sequence list back to a string
    sequence_with_high_confidence_mutations = "".join(mutated_seq)

    return sequence_with_high_confidence_mutations

def generate_mutated_sequences(WT, sequences, cum_prob_threshold, probabilities, model, tokenizer, num_muts, model_identifier):
    """
    Mutates each sequence in `sequences` until they have `num_muts` mutations relative to `WT`.
    
    Parameters:
        WT (str): Wildtype sequence.
        sequences (list of str): Initial sequences to mutate.
        probabilities (torch.Tensor): Probability tensor for each position in the sequence.
        model (torch.nn.Module): Model to generate logits for mutation.
        num_muts (int): Target number of mutations for each sequence.

    Returns:
        list of str: Mutated sequences with the specified number of mutations relative to WT.
    """
    # Tokenize WT sequence
    WT_tokens = tokenizer.convert_tokens_to_ids(list(WT))
    mutated_seqs = []

    # Identify candidate positions with cumulative probability > 25% for non-wildtype amino acids
    candidate_positions = []
    position_weights = []
    for i, p in enumerate(probabilities):
        non_wt_prob = p.sum() - p[WT_tokens[i]]  # cumulative probability for non-wildtype amino acids
        if non_wt_prob > cum_prob_threshold:
            candidate_positions.append(i)
            position_weights.append(non_wt_prob.item())  # store the probability as weight

    # Normalize weights for `random.choices`
    total_weight = sum(position_weights)
    normalized_weights = [w / total_weight for w in position_weights]
    
    with torch.no_grad():
        for seq in sequences:
            mutated_seq = list(seq)
            # print('mutated_seq', mutated_seq)
            
            while hamming_distance(mutated_seq, WT) < num_muts:
                # Randomly choose a candidate position
                pos = random.choices(candidate_positions, weights=normalized_weights, k=1)[0]
                # print('pos', pos)

                # Mask the chosen position
                mutated_seq[pos] = tokenizer.mask_token  # Use <mask> token
                
                # Prepare input for the model
                masked_seq_str = ''.join(mutated_seq)
                # print('masked_seq_str', masked_seq_str)
                inputs = tokenizer(masked_seq_str, return_tensors="pt")
                outputs = model(**inputs)

                # Get logits for valid amino acid tokens
                logits = outputs.logits[0, pos + 1, 4:24]  # Adjust this range based on valid amino acid tokens
                probabilities_pos = F.softmax(logits, dim=-1)
                # print('probabilities_pos', probabilities_pos)

                # Sample a new amino acid for the position
                sampled_idx = np.random.choice(len(probabilities_pos), p=probabilities_pos.detach().numpy())
                new_amino_acid_id = 4 + sampled_idx  # Map to actual token ID range for amino acids
                new_amino_acid = tokenizer.convert_ids_to_tokens([new_amino_acid_id])[0]
                # print('new_amino_acid', new_amino_acid)

                # Apply the mutation
                mutated_seq[pos] = new_amino_acid
                # print('mutated_seq',mutated_seq)

            # Convert tokenized mutated sequence back to amino acid string
            mutated_seq = ''.join(mutated_seq)
            mutated_seqs.append(mutated_seq)

        # print(mutated_seqs)

    return mutated_seqs

def generate_and_evaluate_mutants_p_sampling(WT, reward_models, model, model_identifier, tokenizer, filepath, ep, version, num_designs=5, num_muts=5, cum_prob_threshold=0.25, high_conf_threshold=0.9, seed=None):
    # Set models to evaluation mode
    model.eval()

    # Seed is set during generation after training but not during training
    if seed is not None:
        # Set seeds
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    # Generate single mutant probability space by masking one position at a time of WT,
    single_mutant_logits = get_logits_for_all_positions(model, WT, tokenizer, model_identifier) # get logits for each position
    probabilities = F.softmax(single_mutant_logits, dim=-1) # convert logits to probabilities
    generate_heatmap(WT, probabilities, model_identifier, WT, filepath, ep, version, tokenizer) # generate heatmap and save heatmap as svg and png
    print('Generated heatmap for single mutant space from WT')

    # Mutate WT with high confidence mutations using p-sampling
    sequence_with_high_confidence_mutations = generate_high_confidence_mutant_sequences(WT, probabilities, tokenizer, model_identifier, num_designs, high_conf_threshold)
    print('Mutated WT with high confidence mutations')

    # Generate single mutant probability space by masking one position at a time of sequence with high confidence mutations,
    single_mutant_logits = get_logits_for_all_positions(model, sequence_with_high_confidence_mutations, tokenizer, model_identifier) # get logits for each position
    probabilities = F.softmax(single_mutant_logits, dim=-1) # convert logits to probabilities
    generate_heatmap(WT, probabilities, model_identifier, sequence_with_high_confidence_mutations, filepath, ep, version, tokenizer) # generate heatmap and save heatmap as svg and png
    print('Generated heatmap for single mutant space from sequence with high confidence mutations')

    # Duplicate the mutated sequence to match the number of designs
    sequences_with_high_confidence_mutations = [sequence_with_high_confidence_mutations] * num_designs

    # Add mutations until num_muts of mutations relative to WT sequence are obtained for all sequences_with_high_confidence_mutations
    mutated_seqs = generate_mutated_sequences(WT, sequences_with_high_confidence_mutations, cum_prob_threshold, probabilities, model, tokenizer, num_muts, model_identifier)
    print('Mutated sequences with 5 mutations')
    # print(mutated_seqs)
 
    # Score mutants
    batch_size = num_designs
    scores_tensor = torch.zeros((len(reward_models), batch_size), dtype=torch.float32).to(device)
    for i, reward_model in enumerate(reward_models):
        reward_model.to(device)
        reward_model.eval()
        with torch.no_grad():
            for j, seq in enumerate(mutated_seqs):
                sequence = torch.tensor(aa2ind(list(seq))).to(device)
                score = reward_model.predict(sequence)[0][0]
                scores_tensor[i, j] = score

    # Convert PyTorch tensors to NumPy arrays
    scores_np = scores_tensor.cpu().numpy()

    return mutated_seqs, scores_np

# Function to identify mutations and count them
def identify_mutations_and_count(WT, seq):
    mutations = []
    for i, (wt_aa, seq_aa) in enumerate(zip(WT, seq), start=1):
        if wt_aa != seq_aa:
            mutations.append(f"{wt_aa}{i}{seq_aa}")
    mutation_str = ', '.join(mutations)
    num_mutations = len(mutations)
    return mutation_str, num_mutations

def generate_and_evaluate_mutants_max_sampling(num_designs, num_muts, WT, reward_models, model, seed, model_identifier=None):
    # Set models to evaluation mode
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if model_identifier != "esmc-6b-2024-12":
        model.to(device)
        model.eval()

    # Mutate sequences using different models
    mutated_seqs = mutate_sequences_after_training_esm2_max_sampling(model, num_designs, num_muts, WT, model_identifier)
    
    # Score mutants
    batch_size = num_designs
    scores_tensor = torch.zeros((len(reward_models), batch_size), dtype=torch.float32).to(device)
    for i, reward_model in enumerate(reward_models):
        reward_model.to(device)
        reward_model.eval()
        with torch.no_grad():
            for j, seq in enumerate(mutated_seqs):
                sequence = torch.tensor(aa2ind(list(seq))).to(device)
                score = reward_model.predict(sequence)[0][0]
                scores_tensor[i, j] = score

    # Convert PyTorch tensors to NumPy arrays
    scores_np = scores_tensor.cpu().numpy()

    return mutated_seqs, scores_np

def mutate_sequences_after_training_esm2_max_sampling(model, num_seqs, num_muts, WT, model_identifier=None):
    """ Generate mutated sequences based on the model """
    mutated_seqs = []
    masked_pos = torch.zeros((num_seqs, num_muts), dtype=torch.long).to(device)
    
    # # ESMC 6B cannot be moved to device
    # if model_identifier != "esmc-6b-2024-12":
    #     model.to(device)
    #     model.eval()
    # else:
    #     import time
    #     from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

    #     @retry(stop=stop_after_attempt(10), wait=wait_fixed(1.2))  # Retry up to 10 times with a 1.2s wait
    #     def encode_protein_with_retry(protein):
    #         return model.encode(protein)

    # # ESMC models use difference tokenizer than ESM2
    # if model_identifier == 'esmc_300m' or 'esmc_600m' or 'esmc-6b-2024-12':
    #     from esm.sdk.api import ESMProtein, LogitsConfig
    #     print('imported ESMProtein to prepare sequences for ESMC')

    #     # copy and pasted from https://github.com/evolutionaryscale/esm/blob/main/esm/utils/constants/esm3.py
    #     SEQUENCE_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P",
    #     "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "|", "<mask>" ]

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
    #             protein = ESMProtein(sequence=''.join(masked_seq))

    #             # Throttle requests to avoid hitting API limits
    #             if model_identifier == "esmc-6b-2024-12":
    #                 time.sleep(1.2)  # Adjust time based on your rate limit (e.g., 50 requests/minute = 1.2 seconds/request)

    #             # Encode the sequence and process logits
    #             with torch.no_grad():
    
    #                 # Throttle requests to avoid hitting API limits
    #                 if model_identifier == "esmc-6b-2024-12":
    #                     try:
    #                         protein_tensor = encode_protein_with_retry(protein)
    #                     except RetryError as e:
    #                         print(f"RetryError: Failed to encode protein after multiple attempts: {e}")
    #                         break  # Skip this sequence and continue to the next
    #                     except Exception as e:
    #                         print(f"Unexpected error encoding protein: {e}")
    #                         time.sleep(60)  # Wait 1 minute to reset the rate limit
    #                         continue
                    
    #                 else:
    #                     protein_tensor = model.encode(protein)

    #                 logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True))

    #             sequence_logits = logits_output.logits.sequence.squeeze(0)  # Remove batch dimension
                
    #             # Process each masked position
    #             for pos in positions:
    #                 logits = sequence_logits[pos+1, 4:24]  # Extract logits for amino acids
    #                 log_probs = F.log_softmax(logits, dim=-1)
    #                 probs = torch.exp(log_probs)
    #                 sampled_idx = torch.argmax(probs).item()  # Replace with `torch.multinomial` if sampling is needed
    #                 sampled_aa = SEQUENCE_VOCAB[sampled_idx + 4]  # Correct for offset for amino acid tokens
    #                 final_seq = list(final_seq)
    #                 final_seq[pos] = sampled_aa
    #                 final_seq = ''.join(final_seq)
    #             muts_added = hamming_distance(WT, final_seq)  # Calculate mutations added
    #             if muts_added >= num_muts:
    #                 mutated_seqs.append(final_seq)
    #                 break

    # else:
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D") # esm2 tokenizer, works for all ESM2 model sizes

    for seq_idx in range(num_seqs):
        final_seq = WT # Begin with CreiLOV for first timestep of epoch
        muts_added = 0
        
        while muts_added < num_muts:
            # Mask multiple positions at once
            positions = random.sample(range(1, len(WT)), k=num_muts - muts_added)
            masked_seq = list(final_seq)
            masked_pos[seq_idx, :len(positions)] = torch.tensor(positions).to(device)
            for pos in positions:
                masked_seq[pos] = '<mask>'
            inputs = tokenizer(''.join(masked_seq), return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            
            # Process each masked position
            for mut_idx, pos in enumerate(positions):
                mut_idx = mut_idx + muts_added
                logits = out.logits[0, pos+1, 4:24]  # Adjust index if needed
                log_probs = F.log_softmax(logits, dim=-1)
                probs = torch.exp(log_probs)
                # sampled_idx = torch.multinomial(probs, 1).item()
                sampled_idx = torch.argmax(probs).item()
                sampled_aa = tokenizer.convert_ids_to_tokens(sampled_idx+4) # Correct for offset for amino acid tokens
                final_seq = list(final_seq)
                final_seq[pos] = sampled_aa
                final_seq = ''.join(final_seq)
            muts_added = hamming_distance(WT, final_seq)
            if muts_added >= num_muts:
                mutated_seqs.append(''.join(final_seq))
                break
    
    return mutated_seqs




