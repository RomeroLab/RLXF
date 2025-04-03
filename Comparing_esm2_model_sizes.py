#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from transformers import AutoModelForMaskedLM, AutoTokenizer
from MLP import MLP
import itertools
import copy
import warnings
import optuna
import logging
import sys
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define sequence and parameters
WT_name = "avGFP"
WT = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
slen = len(WT)
num_reward_models = 10
rl_version = 8
sft_version = 0
ep = 5
WT_linewidth = 2

# load ensemble of reward models
reward_models = []
for i in range(num_reward_models):
    model_name = f"reward_model_v{i}.ckpt"
    checkpoint_path = f"./reward_models/{model_name}"
    reward_model = MLP.load_from_checkpoint(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_models.append(reward_model)

# Constants for the mean and standard deviation
scores = []
for reward_model in reward_models:
    reward_model.to(device)
    reward_model.eval()
    with torch.no_grad():
        score = reward_model.predict(WT)[0][0].cpu()  # Assuming predict returns a nested list/array
        scores.append(score)
predicted_wt_score = np.median(np.array(scores, dtype=np.float32))

# Create histogram
model_identifiers = ['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D']
all_fixed_scores = {}  # Dictionary to store scores for each model
all_sft_scores = {}  # Dictionary to store scores for each model
all_rl_scores = {}  # Dictionary to store scores for each model

# load scores
for huggingface_identifier in model_identifiers:
    dir_filepath = f'./logs/PPO_{huggingface_identifier}' # ! update

    # Load mutants from pretrained ESM2 650M, sft, and aligned models
    fixed_scores_np = np.load(f'{dir_filepath}/version_{rl_version}/fixed_{huggingface_identifier}_scores.npy') # ! update
    sft_scores_np = np.load(f'{dir_filepath}/version_{rl_version}/sft_{huggingface_identifier}_scores.npy') # ! update
    rl_scores_np = np.load(f'{dir_filepath}/version_{rl_version}/ema_aligned_{huggingface_identifier}_scores.npy') # ! update
    all_fixed_scores[huggingface_identifier] = np.median(fixed_scores_np, axis=0)
    all_sft_scores[huggingface_identifier] = np.median(sft_scores_np, axis=0)
    all_rl_scores[huggingface_identifier] = np.median(rl_scores_np, axis=0)

    

# Define the conditions (in desired order)
conditions = [
    "Pre-trained",
    "SFT",
    "PPO"
    ]

# Build a dictionary mapping each model to its condition scores.
all_scores = {}
for model in model_identifiers:
    all_scores[model] = {
        "Pre-trained": all_fixed_scores[model],
        "SFT": all_sft_scores[model],
        "PPO": all_rl_scores[model]
        }

# Define a color mapping for the conditions (adjust colors as desired)
color_map = {
    "Pre-trained": "#7570b3",
    "SFT": "#d95f02",
    "PPO": "#e6ab02"
    }

model_labels = {
    'esm2_t6_8M_UR50D': 'ESM-2 (8M)',
    'esm2_t12_35M_UR50D': 'ESM-2 (35M)',
    'esm2_t30_150M_UR50D': 'ESM-2 (150M)',
    'esm2_t33_650M_UR50D': 'ESM-2 (650M)'
}


# Parameters for plotting
n = 100
n_win_rate = 1000

# ---------------------------
# Load data into dataframe
# ---------------------------
df_list = []
for model in model_identifiers:
    for cond in conditions:
        arr = np.array(all_scores[model][cond])[:n]
        np.random.seed(42)
        sample = arr[:n]
        temp = pd.DataFrame({'Score': sample})
        temp['Model'] = model
        temp['Condition'] = cond
        df_list.append(temp)
df_scores = pd.concat(df_list, ignore_index=True)

# Create figure and subplots with a custom width ratio: strip plot larger than win rate plot.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 1]})

# -------------------------------
# Plot 1: Scatter plot of scores.
# -------------------------------
# Set up dodge offsets for conditions
width = 0.8
offsets = np.linspace(-width/2, width/2, len(conditions))
offset_dict = dict(zip(conditions, offsets))

# Prepare a dictionary to hold one scatter handle per condition (for the legend)
scatter_handles = {cond: None for cond in conditions}

# Loop over all data points and plot them with dodge and jitter.
for i, row in df_scores.iterrows():
    model = row["Model"]
    cond = row["Condition"]
    model_index = model_identifiers.index(model)
    
    # Dodge offset plus small random jitter.
    jitter = np.random.uniform(-0.035, 0.035)
    x_val = model_index + offset_dict[cond] + jitter
    color = color_map[cond]
    sc = ax1.scatter(x_val, row["Score"], color=color, edgecolors='none')
    
    # Save the first encountered scatter handle for this condition (for the legend)
    if scatter_handles[cond] is None:
        scatter_handles[cond] = sc

# Draw the predicted wild-type score line.
ax1.axhline(predicted_wt_score, color='black', linestyle='--', linewidth=WT_linewidth,
            label='Predicted CreiLOV Log Fluorescence')

# Replace x-tick labels with custom model labels.
ax1.set_xticks(np.arange(len(model_identifiers)))
ax1.set_xticklabels([model_labels[m] for m in model_identifiers], fontsize=10)
ax1.set_xlabel("ESM-2", fontsize=12)
ax1.set_ylabel("Predicted Log Fluorescence", fontsize=12)
ax1.set_ylim(3.8, 4.25)
ax1.text(0.9, 0.05, f"n = {n}", transform=ax1.transAxes, fontsize=10,
         color='black', va='bottom')

# Overlay median horizontal markers.
means = df_scores.groupby(["Model", "Condition"])["Score"].mean().reset_index()
for _, row in means.iterrows():
    i = model_identifiers.index(row["Model"])
    # x-position based on model index plus the dodge offset for this condition
    x_pos = i + offset_dict[row["Condition"]]
    # Draw a short horizontal line for the mean (adjust the length as desired)
    ax1.hlines(row["Score"], x_pos - 0.05, x_pos + 0.05,
               color='black', lw=WT_linewidth, zorder=10)

# Annotate maximum values above the scores.
y_offset = (df_scores["Score"].max() - df_scores["Score"].min()) * 0.02  # 2% of overall range
max_values = df_scores.groupby(["Model", "Condition"])["Score"].max().reset_index()
for _, row in max_values.iterrows():
    i = model_identifiers.index(row["Model"])
    x_pos = i + offset_dict[row["Condition"]]
    ax1.text(x_pos, np.maximum(row["Score"] + y_offset, 4.165), f"Max:\n{row['Score']:.3f}",
             ha='center', va='bottom', fontsize=10, color='black', zorder=11)

# Build custom legend combining scatter points and a mean indicator.
scatter_handles_list = [scatter_handles[cond] for cond in conditions if scatter_handles[cond] is not None]
mean_handle = Line2D([], [], color='black', lw=WT_linewidth, linestyle='-', label='Mean')
handles = scatter_handles_list + [mean_handle]
labels = conditions + ["Mean"]

ax1.legend(handles=handles, labels=labels,
           title="Training Condition", loc="upper center", bbox_to_anchor=(0.5, -0.15),
           ncol=3, fontsize=10, title_fontsize=12)

# -------------------------------------------------------------------------
# Plot 2: Win rate relative to SFT (PEFT) ESM2 150M.
# -------------------------------------------------------------------------
win_data = []
# Map conditions to x-axis positions.
cond_map = {cond: i for i, cond in enumerate(conditions)}
baseline_model = 'esm2_t33_650M_UR50D'
for model in model_identifiers:
    baseline_arr = np.array(all_scores[baseline_model]["SFT"])
    for cond in conditions:
        if model == baseline_model and cond == "SFT":
            win_rate = 50
        else:
            arr = np.array(all_scores[model][cond])
            win_rate = 100 * np.mean(arr > baseline_arr)
        win_data.append({
            "Model": model,
            "Condition": cond,
            "WinRate": win_rate,
            "x": cond_map[cond]
        })
df_win = pd.DataFrame(win_data)

# Plot a line for each model across conditions.
for model in model_identifiers:
    df_model = df_win[df_win["Model"] == model].copy()
    df_model["cond_order"] = df_model["Condition"].apply(lambda c: conditions.index(c))
    df_model = df_model.sort_values("cond_order")
    ax2.plot(df_model["x"], df_model["WinRate"], marker="o", label=model_labels[model])

# Add a horizontal tie line at 50% win rate.
ax2.axhline(50, color='gray', linestyle='--', lw=linewidth, label='Tie')
ax2.text(0.85, 0.05, f"n = {n_win_rate}", transform=ax2.transAxes, fontsize=10,
         color='black', va='bottom')

ax2.set_xticks(list(cond_map.values()))
ax2.set_xticklabels(conditions, fontsize=8)
ax2.set_xlabel("Training Condition", fontsize=12)
ax2.set_ylabel("Win Rate (%)", fontsize=12)
ax2.set_ylim(0, 100)
ax2.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.15),
           ncol=3, fontsize=10, title_fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('./log/figures/esm2_models_and_designs.svg')
plt.savefig('./log/figures/esm2_models_and_designs.png')
plt.show()





