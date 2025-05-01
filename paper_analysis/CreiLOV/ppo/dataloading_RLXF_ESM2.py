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


class ProtDataModuleESM2(pl.LightningDataModule):
    def __init__(self, WT, batch_size, seed):
        super().__init__()
        self.wt_sequence = WT
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = ProtRepDatasetESM2(self.wt_sequence)

    def train_dataloader(self):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() %2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        return data_utils.DataLoader(
            self.train_ds,  # The dataset to load, in this case, the training dataset
            batch_size=self.batch_size,  # The number of samples in each batch to load
            shuffle=True,  # Enable shuffling to randomize the order of data before each epoch
            worker_init_fn=seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
            generator=generator,  # Specify the generator used for random number generation in shuffling
        )


class ProtRepDatasetESM2(torch.utils.data.Dataset):
    def __init__(self, wt_sequence):
        self.wt_sequence = WT

    def __len__(self):
        return 1 # 1 sequence

    def __getitem__(self, idx):
        # Return the protein sequence as a string and its length
        return self.wt_sequence, len(self.wt_sequence)
