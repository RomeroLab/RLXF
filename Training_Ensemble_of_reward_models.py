#!/usr/bin/env python
# coding: utf-8

# Importing Packages
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
import enum
import os
from sklearn import metrics
import torchmetrics
from scipy.stats import spearmanr
import csv

# import helper scripts (you can change this if you want but check out https://www.nature.com/articles/s41467-024-50712-3)
from training_MLP import (SeqFcnDataset, ProtDataModule, MLP)

# Parameters to update
data_filepath = ''
model_savepath = './trained_models/reward_models'
num_muts_threshold = 4 # variants with this number of mutations or less will be in training set
num_muts_of_val_test_splits = 5 # variants with this number of mutations will be split into validation and test sets
percent_validation_split = 0.75 # percent_validation_split*100 defines percent of variants with num_muts_of_val_test_splits mutations to be in validation set
learning_rate = 1e-6
batch_size = 128
epochs = 2000
num_models = 100 # number of models in ensemble
patience = 400 # patience for EarlyStopping, I recommend training ensemble for awhile after loss plateaus
WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA' # parent sequence
slen = len(WT) # length of parent sequence

# load data
df = pd.read_pickle(f"{data_filepath}/SeqFxnDataset.pkl") # load preprocessed data with Sequence and Score column
# df.head()

# create data splits for SeqFxnDataset
splits_path = f'.{data_filepath}/SeqFxnDataset_splits.pkl'
dm = ProtDataModule(df, num_muts_threshold, num_muts_of_val_test_splits, percent_validation_split, batch_size=None, splits_path=None)
dm.save_splits(splits_path)

# load data for model
dm = ProtDataModule(df, num_muts_threshold, num_muts_of_val_test_splits, percent_validation_split, batch_size, splits_path)

############################################################## train reward models ##############################################################
os.makedirs(model_savepath, exist_ok=True)
for i in range(num_models):
    model = MLP(learning_rate, batch_size, epochs, slen) # Resubstantiate the model for each training iteration
    logger_name = f'reward_model'
    logger = CSVLogger('logs', name=logger_name)
    checkpoint_callback = ModelCheckpoint(dirpath=model_savepath,filename=f'reward_model_v{i}',monitor='val_loss',mode='min',save_top_k=1) # Define the model checkpoint callback with version number in the filename
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min') # Use early stopping
    trainer = pl.Trainer(logger=logger, max_epochs=epochs, callbacks=[early_stopping, checkpoint_callback], enable_progress_bar=False) # Trainer with early stopping and checkpointing
    trainer.fit(model, dm) # Train the model
    metrics_file_name = f'metrics.csv'
    pt_metrics = pd.read_csv(f'logs/reward_model/version_{i}/metrics.csv')

############################################################## plot loss curves ##############################################################
train_losses = []
val_losses = []
for i in range(num_models):
    try:
        pt_metrics = pd.read_csv(f'logs/reward_model/version_{i}/metrics.csv')
        train = pt_metrics[~pt_metrics.train_loss.isna()]
        val = pt_metrics[~pt_metrics.val_loss.isna()]
        max_epochs = max(len(train), len(val))  # Update the maximum number of epochs
        train_losses.append(train.train_loss.tolist())
        val_losses.append(val.val_loss.tolist())
    except FileNotFoundError:
        print(f"Metrics file for version {i} not found.")
        continue

# Pad each loss array in train_losses and val_losses to max_length with np.nan
max_length = max(max(len(loss_array) for loss_array in train_losses), max(len(loss_array) for loss_array in val_losses))
train_losses_padded = [np.pad(loss_array, (0, max_length - len(loss_array)), constant_values=np.nan) for loss_array in train_losses]
val_losses_padded = [np.pad(loss_array, (0, max_length - len(loss_array)), constant_values=np.nan) for loss_array in val_losses]

# Calculate the mean and standard deviation
train_mean = np.nanmean(train_losses_padded, axis=0)
val_mean = np.nanmean(val_losses_padded, axis=0)
train_std = np.nanstd(train_losses_padded, axis=0)
val_std = np.nanstd(val_losses_padded, axis=0)
epochs = np.arange(max_length)

# Plot figure
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_mean, label='Training Loss')
plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2, label='±1 Standard Deviation')
plt.plot(epochs, val_mean, label='Validation Loss')
plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2, label='±1 Standard Deviation')
plt.ylabel('Loss')
plt.xlabel('Epoch')
ax = plt.gca()  # Get the current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
file_path = os.path.join(model_savepath, 'Loss_Curve.png')
plt.savefig(file_path, bbox_inches='tight')
# plt.show()

############################################################## evaluate reward models on test set ##############################################################
all_Y_values = [[] for _ in range(len(df.iloc[dm.test_idx]))]  # List of lists to store predictions for each sequence

# Scores Test Sequences for Models
for i in range(num_models):
    model = MLP(learning_rate, batch_size, epochs, slen)
    model_path = f'{model_savepath}/reward_model_v{i}.ckpt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    test_data_frame = df.iloc[list(dm.test_idx)].copy()  # Use the test index from DataModule
    Y = [model.predict(j).item() for j in test_data_frame['Sequence']]
    for j, score in enumerate(Y):
        all_Y_values[j].append(score)

# Calculate median, mean, and variance for each sequence
medians = [np.median(scores) for scores in all_Y_values]
# variances = [np.var(scores) for scores in all_Y_values]

# Calculating metrics
actual_scores = test_data_frame['functional_score'].tolist()
mse = metrics.mean_squared_error(actual_scores, medians)
r = np.corrcoef(actual_scores, medians)[0][1]
rho, _ = spearmanr(actual_scores, medians)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(actual_scores, medians, color='red', s=5)
plt.plot([min(min(actual_scores), min(medians)), max(max(actual_scores),max(medians))], [min(min(actual_scores),min(medians)), max(max(actual_scores),max(medians))], color='black')  # Diagonal line for reference
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Model Test Results")
plt.text(3.0, 4.1, f"MSE = {mse:.5f}")
plt.text(3.0, 4.05, f"R = {r :.3f}")
plt.text(3.0, 4.0, f"Rho = {rho :.3f}")
ax = plt.gca()  # Get the current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plot_file_path = f'{model_savepath}/Test_Results.png'
plt.savefig(plot_file_path)

# Save metrics
metrics_data = [['MSE', 'Pearson R', "Spearman's Rho"], [mse, r, rho]]
csv_file_path = plot_file_path.replace('.png', '.csv')
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(metrics_data)




