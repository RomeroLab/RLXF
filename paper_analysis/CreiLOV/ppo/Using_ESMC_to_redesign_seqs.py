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
from functions import (load_reward_model, identify_mutations_and_count, generate_df, generate_and_evaluate_mutants, generate_and_evaluate_mutants_max_sampling,
    mutate_sequences_after_training, mutate_sequences_after_training_esm2_max_sampling, get_sft_version_file)
from dataloading_RLXF_ESM2 import (ProtDataModuleESM2, ProtRepDatasetESM2)
from PPO_ESM2_650M_with_model_saving_DDP import RLXF_PPO_ESM2
from transformers import AutoModelForMaskedLM, AutoTokenizer
from MLP import MLP
import itertools
import copy
import warnings
import optuna
import logging
import sys
from optuna.exceptions import TrialPruned
from pytorch_lightning.callbacks import Callback
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.sdk.forge import ESM3ForgeInferenceClient

# Define amino acid dictionary for tokenization, define WT for length of context window
AAs = 'ACDEFGHIKLMNPQRSTVWY' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # CreiLOV
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
sequence_length = len(WT)

# Parameters
num_EnsMLPs = 100  # We have 100 reward models
num_designs = 1000
seed = 7028

# Load reward models
reward_models = []
for i in range(num_EnsMLPs):
    model_name = f"best_model_v{i}.ckpt"
    checkpoint_path = f"./MLP_Reward_Models/{model_name}"
    reward_model = load_reward_model(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_models.append(reward_model)


# Load pretrained model
model_identifier ='esmc_300m' # esmc_600m
ESMC_model = ESMC.from_pretrained(model_identifier)

# Generate and evaluate 1000 designs with 5 mutants
fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_max_sampling(num_designs=num_designs,
                                                                    num_muts=5,
                                                                    WT=WT,
                                                                    reward_models=reward_models,
                                                                    model=ESMC_model,
                                                                    seed=seed,
                                                                    model_identifier=model_identifier)

print(f"Status: finished generating sequences with {model_identifier}")

# Save mutants from ESMC
base_path = f'./logs/'
np.save(base_path + f'fixed_scores_{model_identifier}.npy', fixed_scores_np)
with open(base_path + f'fixed_mutated_seqs_{model_identifier}.txt', 'w') as file:
    for seq in fixed_mutated_seqs:
        file.write(seq + '\n')

#### Function used:

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
    
    # ESMC 6B cannot be moved to device
    if model_identifier != "esmc-6b-2024-12":
        model.to(device)
        model.eval()
    else:
        import time
        from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

        @retry(stop=stop_after_attempt(10), wait=wait_fixed(1.2))  # Retry up to 10 times with a 1.2s wait
        def encode_protein_with_retry(protein):
            return model.encode(protein)

    # ESMC models use difference tokenizer than ESM2
    if model_identifier == 'esmc_300m' or 'esmc_600m' or 'esmc-6b-2024-12':
        from esm.sdk.api import ESMProtein, LogitsConfig
        print('imported ESMProtein to prepare sequences for ESMC')

        # copy and pasted from https://github.com/evolutionaryscale/esm/blob/main/esm/utils/constants/esm3.py
        SEQUENCE_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P",
        "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "|", "<mask>" ]

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
                protein = ESMProtein(sequence=''.join(masked_seq))

                # Throttle requests to avoid hitting API limits
                if model_identifier == "esmc-6b-2024-12":
                    time.sleep(1.2)  # Adjust time based on your rate limit (e.g., 50 requests/minute = 1.2 seconds/request)

                # Encode the sequence and process logits
                with torch.no_grad():
    
                    # Throttle requests to avoid hitting API limits
                    if model_identifier == "esmc-6b-2024-12":
                        try:
                            protein_tensor = encode_protein_with_retry(protein)
                        except RetryError as e:
                            print(f"RetryError: Failed to encode protein after multiple attempts: {e}")
                            break  # Skip this sequence and continue to the next
                        except Exception as e:
                            print(f"Unexpected error encoding protein: {e}")
                            time.sleep(60)  # Wait 1 minute to reset the rate limit
                            continue
                    
                    else:
                        protein_tensor = model.encode(protein)

                    logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True))

                sequence_logits = logits_output.logits.sequence.squeeze(0)  # Remove batch dimension
                
                # Process each masked position
                for pos in positions:
                    logits = sequence_logits[pos+1, 4:24]  # Extract logits for amino acids
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = torch.exp(log_probs)
                    sampled_idx = torch.argmax(probs).item()  # Replace with `torch.multinomial` if sampling is needed
                    sampled_aa = SEQUENCE_VOCAB[sampled_idx + 4]  # Correct for offset for amino acid tokens
                    final_seq = list(final_seq)
                    final_seq[pos] = sampled_aa
                    final_seq = ''.join(final_seq)
                muts_added = hamming_distance(WT, final_seq)  # Calculate mutations added
                if muts_added >= num_muts:
                    mutated_seqs.append(final_seq)
                    break

    else:
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




