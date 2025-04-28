#!/usr/bin/env python
# coding: utf-8

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from transformers import AutoTokenizer
import pickle
import os
from torchtext import vocab # This package can give problems sometimes, it may be necessary to downgrade to a specific version
from collections import OrderedDict
import torch
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from scipy.special import softmax
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM
from MLP import (SeqFcnDataset, ProtDataModule, MLP)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Plotting parameters
WT_name = "avGFP"
WT = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK' # avGFP
model_identifier ='esm2_t6_8M_UR50D' # 'esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D'
num_models = 10
ppo_version = 8
ppo_ep = 5
sft_version = 0

# Define amino acid dictionary for tokenization, define WT for length of context window
AAs = 'ACDEFGHIKLMNPQRSTVWY' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
sequence_length = len(WT)
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_identifier}")
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
Magma_r = plt.cm.magma_r(np.linspace(0, 1, 256))
Magma_r[0] = [0, 0, 0, 0.03]  # fade zero to white
cmap = LinearSegmentedColormap.from_list("Modified_Magma_r", Magma_r, N=256)
esm2_token_start = 4
esm2_token_end = 24
output_dir = './logs/figures/'
os.makedirs(output_dir, exist_ok=True)


####################################### Create one-hot encoding of WT ########################################

# Create empty 20 x L matrix
wt_matrix = np.zeros((20, len(WT)))

# Fill in 1 at the WT amino acid at each position
for pos, aa in enumerate(WT):
    if aa in AAs:  # Only place a 1 if the AA is valid
        aa_idx = AAs.index(aa)
        wt_matrix[aa_idx, pos] = 1

# Save the WT one-hot matrix
np.save('./logs/figures/WT.npy', wt_matrix)

# Plot heatmap
plt.figure(figsize=(len(WT) // 3, 6))
sns.heatmap(wt_matrix, cmap=cmap, square=True, linewidths=0.003, linecolor='0.7', vmin=0, vmax=1)

plt.yticks(np.arange(len(AAs)) + 0.5, list(AAs), fontsize=8, rotation=0)
plt.xlabel("Position in WT", fontsize=18)
plt.ylabel("Amino Acid", fontsize=18)
plt.title(f"{WT_name} One-Hot Amino Acid Map")
plt.tight_layout()

# Save plot
plt.savefig('./logs/figures/WT.png', dpi=300)
plt.savefig('./logs/figures/WT.svg')
plt.show()

####################################### Create SM probability maps from pre-trained, SFT, and PPO ########################################

# Load pre-trained, SFT, and PPO ESM-2 models
# Pre-trained
pretrained_ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
pretrained_ESM2 = pretrained_ESM2.to(device).eval()

# SFT
logger_name = f'SFT_{model_identifier}'
sft_ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
sft_state_dict = torch.load(f'./logs/{logger_name}/version_{sft_version}/SFT_{model_identifier}_v{sft_version}.pt')
sft_ESM2.load_state_dict(sft_state_dict)
sft_ESM2 = sft_ESM2.to(device).eval()

# PPO
ppo_logger_name = f'./logs/PPO_{model_identifier}'
ppo_ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
ppo_state_dict = torch.load(f'{ppo_logger_name}/version_{ppo_version}/ema_aligned_{model_identifier}_v{ppo_version}_ep{ppo_ep}.pt')
ppo_ESM2.load_state_dict(ppo_state_dict)
ppo_ESM2 = ppo_ESM2.to(device).eval()

models = {
    'Pretrained': pretrained_ESM2,
    'SFT': sft_ESM2,
    'PPO': ppo_ESM2
}

# Helper to mask a sequence at a position
def mask_sequence(sequence, mask_pos):
    """Return a sequence with a single masked position"""
    masked_seq = list(sequence)
    masked_seq[mask_pos] = "<mask>"
    return " ".join(masked_seq)

# Function to get log probabilities
def get_single_mut_log_probs(model, sequence):
    new_log_states = torch.zeros((len(sequence), 20), dtype=torch.float32).to(device)

    with torch.no_grad():
        for mask_pos in range(len(sequence)):
            masked_seq = mask_sequence(sequence, mask_pos)
            inputs = tokenizer(masked_seq, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits[:, :, esm2_token_start:esm2_token_end]
            log_probs = F.log_softmax(logits[0, mask_pos + 1], dim=-1)
            new_log_states[mask_pos] = log_probs

    return new_log_states

# Function to plot and save heatmap
def plot_heatmap(log_probabilities, model_name):
    probs = torch.exp(log_probabilities).to(torch.float32).cpu()
    all_tokens = list(tokenizer.get_vocab().keys())[4:24]

    plt.figure(figsize=(sequence_length // 3, 6))
    sns.heatmap(probs.T, cmap=cmap, square=True, linewidths=0.003, linecolor='0.7', vmin=0, vmax=1)

    plt.xticks(np.arange(sequence_length) + 0.5, range(1, sequence_length + 1), fontsize=8, rotation=90)
    plt.yticks(np.arange(20) + 0.5, AAs, fontsize=10, rotation=0)
    plt.xlabel('Position', fontsize=16)
    plt.ylabel('Amino Acid', fontsize=16)
    plt.title(f'Single Mutation Probabilities – {model_name}', fontsize=18)

    # Overlay WT residues
    for pos, aa in enumerate(WT):
        if aa in AAs:
            aa_idx = AAs.index(aa)
            plt.scatter(pos + 0.5, aa_idx + 0.5, color='black', s=30)

    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='WT')],
        loc='upper right', fontsize=10
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_single_mut_heatmap.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'{model_name}_single_mut_heatmap.svg'))
    np.save(f'./logs/figures/{model_name}_single_mut_heatmap.npy', probs.cpu().detach().numpy())
    plt.close()

for model_name, model in models.items():
    print(f"Processing model: {model_name}")
    log_probs = get_single_mut_log_probs(model, WT)
    plot_heatmap(log_probs, model_name)

print("finished generating single mutation maps")

####################################### Create SM probability maps from reward model predictions ########################################

# Load models
models = []
for i in range(num_models):
    model_name = f"reward_model_v{i}.ckpt"
    checkpoint_path = f"./reward_models/{model_name}"
    reward_model = MLP.load_from_checkpoint(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    models.append(reward_model)
    print('loaded reward model')

# Score the WT sequence as a whole
WT_tensor = torch.tensor(aa2ind(list(WT)))
wt_scores = [model.predict(WT_tensor).item() for model in models]
WT_score = np.percentile(wt_scores, 5)
print('scored WT', WT_score)

# Step 2: Score all single mutants of WT
heatmap_matrix = np.zeros((20, sequence_length))

for pos in range(sequence_length):
    for aa_idx, aa in enumerate(AAs):

        mutant_seq = list(WT)
        mutant_seq[pos] = aa
        ind_seq = torch.tensor(aa2ind(mutant_seq))

        scores = []
        for model in models:
            score = model.predict(ind_seq).item()
            scores.append(score)

        functional_score = np.percentile(scores, 5)
        heatmap_matrix[aa_idx, pos] = functional_score - WT_score

# # Subtract WT score from all entries and normalize
heatmap_matrix = softmax(heatmap_matrix, axis=0)  # AA-wise softmax per position

# Plot normalized heatmap
plt.figure(figsize=(sequence_length // 3, 6))
sns.heatmap(heatmap_matrix, cmap=cmap, square=True, linewidths=0.003, linecolor='0.7')

# Overlay WT residues
for pos, aa in enumerate(WT):
    if aa in AAs:
        aa_index = AAs.index(aa)
        plt.scatter(pos + 0.5, aa_index + 0.5, color='black', s=30)
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='WT')],
    loc='upper right'
)

plt.yticks(np.arange(20) + 0.5, list(AAs))
plt.xlabel('Position in WT', fontsize=18)
plt.ylabel('Mutant Amino Acid', fontsize=18)
plt.title('Predicted Function for All Single Mutants')
plt.tight_layout()
plt.savefig("./logs/figures/single_mutant_function_predictions.png", dpi=300)
plt.show()

# Save matrix
np.save("./logs/figures/single_mutant_function_predictions.npy", heatmap_matrix)

####################################### Perform Linear Regression ########################################

# Matrix filenames
matrix_filenames = [
    'WT.npy',
    'Pretrained_single_mut_heatmap.npy',
    'SFT_single_mut_heatmap.npy',
    'PPO_single_mut_heatmap.npy',
    'single_mutant_function_predictions.npy'
]

# Load and flatten all matrices
matrix_data = [np.load(os.path.join(output_dir, fname)).reshape(-1) for fname in matrix_filenames]
matrix_shape = np.load(os.path.join(output_dir, matrix_filenames[0])).shape

# Identify target
target_name = 'Pretrained_single_mut_heatmap.npy'
target_idx = matrix_filenames.index(target_name)
y = matrix_data[target_idx]

# Store regression results and predicted matrices
results = {}
predicted_matrices = []

for i, (fname, x) in enumerate(zip(matrix_filenames, matrix_data)):
    
    # Perform linear regression: y ≈ m * x + b
    m, b = np.polyfit(x, y, deg=1)
    y_pred = m * x + b
    y_pred_matrix = y_pred.reshape(matrix_shape)
    predicted_matrices.append(y_pred_matrix)

    # Compute R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    results[fname] = {'slope': m, 'intercept': b, 'r2': r2}

# Print regression metrics
print(f"Linear regression results for predicting {target_name} from each matrix:")
for fname, stats in results.items():
    print(f"{fname}: slope = {stats['slope']:.4f}, intercept = {stats['intercept']:.4f}, R² = {stats['r2']:.4f}")

####################################### Perform MDS ########################################

# Compute distances between predicted matrices
flattened = [mat.reshape(-1) for mat in predicted_matrices]
distance_matrix = pairwise_distances(flattened, metric='euclidean')

# Run MDS on distance matrix
RS = 9
embedding = MDS(n_init=10, dissimilarity="precomputed", random_state=RS, normalized_stress='auto')
X = embedding.fit_transform(distance_matrix)

# Save MDS coordinates
np.save('./logs/figures/MDS_coordinates_for_SM_matrices.npy', X)

# Plot MDS
labels = [fname.replace('.npy', '') for fname in matrix_filenames]
colors = plt.cm.plasma(np.linspace(0, 1, len(labels)))

plt.figure(figsize=(6, 6))
for i, label in enumerate(labels):
    plt.scatter(X[i, 0], X[i, 1], color=colors[i], s=50)
    plt.text(X[i, 0] + 0.3, X[i, 1], label, fontsize=8)

plt.xlabel("MDS dimension 1")
plt.ylabel("MDS dimension 2")
plt.title("MDS of Linear Regression–Predicted Matrices")
# plt.xlim([-8, 8])
# plt.ylim([-8, 8])
# plt.tight_layout()

# Save and/or show plot
plt.savefig('./logs/figures/MDS_plot_from_predicted_matrices.svg')
plt.savefig('./logs/figures/MDS_plot_from_predicted_matrices.png')
plt.show()





